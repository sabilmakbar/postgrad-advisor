"""CV upload mechanics (story 009): attach, hold, consume. Scripted fakes, no
live API, no network.

Every guard here carries its paired positive: interception vs. a normal answer
still reaching the interviewer, a replacement CV at cv_confirm vs. a plain "yes"
still reaching the confirmation parser.
"""

import shutil
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

import app.cv as cv_mod
import app.graph as graph_mod
from app.cv import CVConfirmation, split_upload
from app.graph import build_graph
from app.prompts import (
    CV_AFTER_CAP,
    CV_EXTRACTION_FAILED,
    CV_PARSE_FAILED,
    CV_RECEIVED,
    CV_REQUEST,
    CV_UNREADABLE,
    FIRST_QUESTION,
    FIXED_INTERVIEW_QUESTIONS,
    TURN_CAP_MESSAGE,
)
from app.model_layer import AllProvidersExhausted
from app.state import CVFacts, SlotUpdate, WorkExperience

FIXTURE = Path(__file__).parent / "fixtures" / "cv.txt"
COMPLETE = SlotUpdate(degree_intent="phd", field="machine learning", target_regions=["UK"])
FIRST_FACTS = CVFacts(degree="BSc Computer Science", gpa="3.7 / 4.0")
SECOND_FACTS = CVFacts(degree="MSc Computer Science", gpa="3.9 / 4.0")


@pytest.fixture
def cv_dir(monkeypatch, tmp_path):
    """CV_DIR pointed at a tmp dir so uploads can be made for real. Paths built
    from it are /upload-shaped — absolute, under CV_DIR — which is exactly what
    the interception predicate keys on."""
    d = tmp_path / "data" / "cv"
    d.mkdir(parents=True)
    monkeypatch.setattr(cv_mod, "CV_DIR", d)
    return d


def _upload(cv_dir, name="deadbeef.txt", readable=True):
    """The path /upload would return, with the file actually written unless the
    test wants an unreadable one."""
    dest = cv_dir / name
    if readable:
        shutil.copy(FIXTURE, dest)
    return str(dest)


def _wire(monkeypatch, questions=(), parses=(), facts=(), confirms=(), seen=None, chat_seen=None):
    """Scripted fakes for every model call the CV path can make, plus stubs for
    the shortlister and fit nodes (covered by their own tests).

    Strict iterators: an unscripted call raises StopIteration, so a test that
    scripts nothing PROVES the call never happened. Pass ``seen`` to record what
    the slot parser was actually shown.
    """
    q, p, f, c = iter(questions), iter(parses), iter(facts), iter(confirms)

    def _parse(prompt):
        if seen is not None:
            seen.append(prompt)
        return next(p)

    def _chat(prompt):
        if chat_seen is not None:
            chat_seen.append(prompt)
        return AIMessage(content=next(q))

    monkeypatch.setattr(
        graph_mod, "chat", lambda tier="heavy": RunnableLambda(_chat)
    )
    monkeypatch.setattr(
        graph_mod, "structured", lambda schema, tier="heavy": RunnableLambda(_parse)
    )
    monkeypatch.setattr(
        cv_mod,
        "structured",
        lambda schema, tier="heavy": RunnableLambda(
            lambda _: next(f if schema is CVFacts else c)
        ),
    )
    monkeypatch.setattr(
        graph_mod,
        "shortlist_node",
        lambda s: {"handed_off": True, "messages": s.messages + [("ai", "SHORTLIST")]},
    )
    monkeypatch.setattr(
        graph_mod, "fit_node", lambda s: {"messages": s.messages + [("ai", "VERDICTS")]}
    )


def _thread():
    return {"configurable": {"thread_id": "t"}}


def _ai(state):
    return [text for role, text in state["messages"] if role == "ai"]


# --- AC 2: interception at the interviewer stage --------------------------


def test_upload_predicate_matches_upload_paths_only():
    """The predicate is pinned against the constant /upload actually writes to
    (app.server imports it from app.config), not a hand-typed relative string.
    The remainder comes back with it: text and attachment are two parts of one
    Send and both have to survive the split (fix round)."""
    from app.server import CV_DIR as SERVER_CV_DIR

    assert split_upload(str(SERVER_CV_DIR / "deadbeef.pdf")) == (
        str(SERVER_CV_DIR / "deadbeef.pdf"),
        "",
    )
    assert split_upload("Europe or the UK, needs funding") == (
        None,
        "Europe or the UK, needs funding",
    )
    # a user-typed path outside CV_DIR is a normal message, not an upload
    assert split_upload("/home/me/my cv.pdf")[0] is None
    # prefix false-match guard: data/cvfoo is not inside data/cv
    assert split_upload(str(SERVER_CV_DIR.parent / "cvfoo" / "x.pdf"))[0] is None
    # the chip travels on its own line beside whatever the user typed
    assert split_upload("here it is\n" + str(SERVER_CV_DIR / "a.txt")) == (
        str(SERVER_CV_DIR / "a.txt"),
        "here it is",
    )


def test_upload_predicate_rejects_escapes_and_extra_paths():
    """Fix round 2. The containment check is lexical, so an absolute line is
    resolved first: `..` cannot walk out of CV_DIR, and the directory itself is
    not a CV. Only ONE attachment can exist per Send, so a second upload-shaped
    line is dropped rather than handed on as text — AC 2 makes a path reaching
    the LLM structurally impossible, not merely unlikely."""
    from app.server import CV_DIR as SERVER_CV_DIR

    escape = str(SERVER_CV_DIR / ".." / ".." / "etc" / "passwd")
    assert split_upload(escape) == (None, escape)  # not an upload: stays text
    assert split_upload(str(SERVER_CV_DIR))[0] is None  # the bare dir is not a CV
    assert split_upload(str(SERVER_CV_DIR) + "/")[0] is None

    first, second = str(SERVER_CV_DIR / "a.txt"), str(SERVER_CV_DIR / "b.txt")
    assert split_upload(f"{first}\n{second}") == (first, "")  # second dropped
    assert split_upload(f"note\n{first}\n{second}") == (first, "note")


def test_upload_mid_interview_is_intercepted_in_code(monkeypatch, cv_dir):
    """AC 2: the path is stored, answered with the fixed line, and the interview
    carries on. `parses=()` is the proof that no LLM saw it — a slot-parser call
    would raise StopIteration."""
    _wire(monkeypatch, questions=(), parses=(), facts=[FIRST_FACTS, SECOND_FACTS])
    path = _upload(cv_dir)
    g = build_graph()
    g.invoke({}, _thread())
    state = g.invoke(Command(resume=path), _thread())

    assert state["pending_cv"] == path
    assert state["cv_facts"].degree == "BSc Computer Science"
    # fix round 2: the acknowledgement alone — the open question is already in
    # the log right above it and is not restated
    assert state["messages"][-2:] == [("ai", FIRST_QUESTION), ("ai", CV_RECEIVED)]
    assert not any(path in text for _, text in state["messages"])  # path never in the log
    assert state["__interrupt__"]  # still interviewing
    assert not state.get("handed_off")

    # a second upload replaces the first without stacking acknowledgements
    again = _upload(cv_dir, "cafebabe.txt")
    state = g.invoke(Command(resume=again), _thread())
    assert state["pending_cv"] == again
    assert state["cv_facts"].degree == "MSc Computer Science"
    assert state["messages"][-2:] == [("ai", FIRST_QUESTION), ("ai", CV_RECEIVED)]


def test_the_answer_after_an_upload_is_parsed_against_the_open_question(
    monkeypatch, cv_dir
):
    """The acknowledgement now sits between the question and the next answer, so
    the parser's "Question asked" can no longer be messages[-2] (fix round 2)."""
    seen = []
    _wire(
        monkeypatch,
        questions=["Q2"],
        parses=[SlotUpdate(degree_intent="phd")],
        facts=[FIRST_FACTS],
        seen=seen,
    )
    g = build_graph()
    g.invoke({}, _thread())
    g.invoke(Command(resume=_upload(cv_dir)), _thread())  # attachment alone
    state = g.invoke(Command(resume="a PhD, please"), _thread())

    assert seen[0][-1][1] == f"Question asked: {FIRST_QUESTION}\nAnswer: a PhD, please"
    assert state["degree_intent"] == "phd"


def test_a_second_path_in_one_message_never_reaches_the_llm_or_the_log(
    monkeypatch, cv_dir
):
    """Fix round 2: only the first upload-shaped line is the attachment; the
    rest are dropped, not passed on as the typed remainder. `parses=()` proves
    no slot-parser call happened — a path as an 'answer' is the live incident."""
    _wire(monkeypatch, questions=(), parses=(), facts=[FIRST_FACTS])
    first, second = _upload(cv_dir, "one.txt"), _upload(cv_dir, "two.txt")
    g = build_graph()
    g.invoke({}, _thread())
    state = g.invoke(Command(resume=f"{first}\n{second}"), _thread())

    assert state["pending_cv"] == first
    assert state["cv_facts"].degree == "BSc Computer Science"
    assert not any(second in text for _, text in state["messages"])
    assert state["messages"][-1] == ("ai", CV_RECEIVED)


def _at_the_turn_cap(monkeypatch, cv_dir):
    """A graph paused one vague answer short of the turn cap: state.turns ==
    MAX_TURNS, so the NEXT answer takes the capped branch."""
    _wire(
        monkeypatch,
        questions=[f"Q{i}" for i in range(1, 10)],  # turn 0 is fixed text: 9 generated, not 10
        parses=[SlotUpdate() for _ in range(10)],
    )
    g = build_graph()
    g.invoke({}, _thread())
    for _ in range(graph_mod.MAX_TURNS - 1):
        g.invoke(Command(resume="not sure yet"), _thread())
    return g


def test_a_cv_attached_on_the_capped_turn_is_not_promised_a_read(monkeypatch, cv_dir):
    """Fix round 2: the cap routes straight to END, so a held CV is never
    consumed. Saying "I'll read it right after we finish" there would be a lie —
    the capped turn says the run is over and a fresh chat is needed instead."""
    g = _at_the_turn_cap(monkeypatch, cv_dir)
    state = g.invoke(Command(resume="still no idea\n" + _upload(cv_dir)), _thread())

    assert state["ended"] is True
    assert CV_RECEIVED not in _ai(state)  # no promise the run cannot keep
    assert _ai(state)[-2:] == [CV_AFTER_CAP, TURN_CAP_MESSAGE]
    assert "__interrupt__" not in state  # the run really is over


def test_a_cv_held_from_an_earlier_turn_is_answered_honestly_at_the_cap(
    monkeypatch, cv_dir
):
    """Same lie, one turn removed: the CV is attached alone before the cap and
    the capped answer carries no attachment. Nothing consumes it either way."""
    g = _at_the_turn_cap(monkeypatch, cv_dir)
    g.invoke(Command(resume=_upload(cv_dir)), _thread())  # attachment alone
    state = g.invoke(Command(resume="still no idea"), _thread())

    assert state["ended"] is True and state["pending_cv"]  # held, never consumed
    assert _ai(state)[-2:] == [CV_AFTER_CAP, TURN_CAP_MESSAGE]


def test_normal_answer_still_reaches_the_interviewer(monkeypatch, cv_dir):
    """Paired positive for the interception guard: a non-upload answer is parsed
    into slots and gets the next question, exactly as before."""
    chat_seen = []
    _wire(
        monkeypatch,
        questions=[],
        parses=[SlotUpdate(degree_intent="phd")],
        chat_seen=chat_seen,
    )
    g = build_graph()
    g.invoke({}, _thread())
    state = g.invoke(Command(resume="a PhD, please"), _thread())

    assert state["degree_intent"] == "phd"  # the parser ran
    assert state.get("pending_cv") is None
    assert state["messages"][-1] == ("ai", FIXED_INTERVIEW_QUESTIONS["field"])
    assert chat_seen == []


def test_text_sent_with_an_upload_is_answered_and_the_cv_held(monkeypatch, cv_dir):
    """Fix round: one Send carrying BOTH text and an attachment processes both.
    The wire format is the composer's own — the typed text, then the path on its
    own line."""
    seen = []
    _wire(
        monkeypatch,
        questions=["Q2"],
        parses=[SlotUpdate(degree_intent="phd")],
        facts=[FIRST_FACTS],
        seen=seen,
    )
    path = _upload(cv_dir)
    g = build_graph()
    g.invoke({}, _thread())
    state = g.invoke(Command(resume="a PhD, please\n" + path), _thread())

    assert state["degree_intent"] == "phd"  # the typed part counted as an answer
    assert state["pending_cv"] == path  # and the CV is still held for intake
    assert state["cv_facts"].degree == "BSc Computer Science"
    assert ("human", "a PhD, please") in state["messages"]
    assert _ai(state)[-2:] == [CV_RECEIVED, "Q2"]  # ack AND the next question
    assert not any(path in text for _, text in state["messages"])  # path stays out
    prompt = seen[0][-1][1]  # the slot parser was shown the text...
    assert "a PhD, please" in prompt and path not in prompt  # ...and not the path


# --- AC 4: the ask must be checkpointed before the interrupt fires ---------


def test_cv_request_is_checkpointed_when_the_intake_interrupt_fires(monkeypatch):
    """AC 4 pin. Pre-fix, cv_intake appended CV_REQUEST only on the resume
    re-run — after the server captured its replay snapshot — so the ask
    re-emitted one turn late as a dead turn. Verified failing on the pre-fix
    code before this landed: the interrupt carried CV_REQUEST while
    state.messages held only [Q1, answer, SHORTLIST]."""
    _wire(monkeypatch, questions=(), parses=[COMPLETE])
    g = build_graph()
    g.invoke({}, _thread())
    state = g.invoke(Command(resume="a PhD in ML in the UK"), _thread())

    assert state["__interrupt__"][0].value == CV_REQUEST
    assert ("ai", CV_REQUEST) in state["messages"]
    assert _ai(state).count(CV_REQUEST) == 1  # asked once, not once per superstep


def test_unreadable_path_re_asks_once_then_accepts(monkeypatch, cv_dir):
    """The AC-4 rule holds on the retry leg too, and the loop still terminates."""
    _wire(monkeypatch, questions=(), parses=[COMPLETE], facts=[FIRST_FACTS])
    g = build_graph()
    g.invoke({}, _thread())
    g.invoke(Command(resume="a PhD in ML in the UK"), _thread())
    state = g.invoke(Command(resume=_upload(cv_dir, "gone.txt", readable=False)), _thread())

    assert state["__interrupt__"][0].value == CV_UNREADABLE
    assert ("ai", CV_UNREADABLE) in state["messages"]
    assert _ai(state).count(CV_UNREADABLE) == 1

    state = g.invoke(Command(resume=_upload(cv_dir, "good.txt")), _thread())
    assert state["cv_facts"].degree == "BSc Computer Science"


# --- AC 3: a held upload is consumed without an ask -----------------------


def test_pending_upload_is_consumed_without_a_cv_request_turn(monkeypatch, cv_dir):
    """AC 3: upload mid-interview, finish the interview, and intake goes
    straight to parse + echo — no CV request turn anywhere in the transcript."""
    path = _upload(cv_dir)
    _wire(monkeypatch, questions=(), parses=[COMPLETE], facts=[FIRST_FACTS])
    g = build_graph()
    g.invoke({}, _thread())
    g.invoke(Command(resume=path), _thread())
    state = g.invoke(Command(resume="a PhD in ML in the UK"), _thread())

    assert CV_REQUEST not in _ai(state)
    assert state["cv_facts"].degree == "BSc Computer Science"
    assert _ai(state)[-1].startswith("Here is what I read from your CV")
    assert state["__interrupt__"][0].value == _ai(state)[-1]  # paused at cv_confirm
    assert state.get("pending_cv") is None  # consumed, not re-consumable


# --- AC 5: a second CV at cv_confirm replaces the extraction ---------------


def test_cv_path_at_confirm_replaces_the_extraction(monkeypatch, cv_dir):
    """AC 5: a new CV means 'use this instead' — re-parse, re-echo, still
    unconfirmed. `confirms=()` proves the confirmation LLM was never asked to
    interpret a file path as a yes/no reply."""
    _wire(
        monkeypatch,
        questions=(),
        parses=[COMPLETE],
        facts=[FIRST_FACTS, SECOND_FACTS],
        confirms=(),
    )
    g = build_graph()
    g.invoke({}, _thread())
    g.invoke(Command(resume="a PhD in ML in the UK"), _thread())
    g.invoke(Command(resume=_upload(cv_dir, "one.txt")), _thread())
    state = g.invoke(Command(resume=_upload(cv_dir, "two.txt")), _thread())

    assert state["cv_facts"].degree == "MSc Computer Science"  # replaced
    assert not state.get("cv_confirmed")  # never written: still awaiting a yes
    assert "MSc Computer Science" in state["messages"][-1][1]  # re-echoed
    assert state["__interrupt__"][0].value == state["messages"][-1][1]
    assert state.get("pending_cv") is None


def test_corrections_sent_with_a_replacement_cv_apply_to_the_new_extraction(
    monkeypatch, cv_dir
):
    """Fix round: at cv_confirm a replacement CV plus correction text applies
    BOTH — the corrections belong to the new extraction, not to the facts being
    replaced (3.95 is neither CV's own GPA, so both steps are proved)."""
    _wire(
        monkeypatch,
        questions=(),
        parses=[COMPLETE],
        facts=[FIRST_FACTS, SECOND_FACTS],
        confirms=[CVConfirmation(confirmed=False, corrections=CVFacts(gpa="3.95 / 4.0"))],
    )
    g = build_graph()
    g.invoke({}, _thread())
    g.invoke(Command(resume="a PhD in ML in the UK"), _thread())
    g.invoke(Command(resume=_upload(cv_dir, "one.txt")), _thread())
    wire = "my gpa is actually 3.95 / 4.0\n" + _upload(cv_dir, "two.txt")
    state = g.invoke(Command(resume=wire), _thread())

    assert state["cv_facts"].degree == "MSc Computer Science"  # the new CV parsed
    assert state["cv_facts"].gpa == "3.95 / 4.0"  # the correction applied on top
    assert not state.get("cv_confirmed")  # correcting is not confirming
    assert state.get("pending_cv") is None and state.get("pending_correction") is None
    assert "3.95 / 4.0" in state["messages"][-1][1]  # re-echoed with both


def test_text_sent_with_the_cv_at_intake_is_applied_as_a_correction(monkeypatch, cv_dir):
    """The intake ask is the third interrupt a mixed Send can land in: the path
    must still parse (not be swallowed into the filename) and the text applied."""
    _wire(
        monkeypatch,
        questions=(),
        parses=[COMPLETE],
        facts=[FIRST_FACTS],
        confirms=[
            CVConfirmation(confirmed=False, corrections=CVFacts(english_test="IELTS 7.5"))
        ],
    )
    g = build_graph()
    g.invoke({}, _thread())
    g.invoke(Command(resume="a PhD in ML in the UK"), _thread())
    state = g.invoke(Command(resume="my ielts is 7.5\n" + _upload(cv_dir)), _thread())

    assert CV_UNREADABLE not in _ai(state)  # the path was read, not the whole line
    assert state["cv_facts"].degree == "BSc Computer Science"
    assert state["cv_facts"].english_test == "IELTS 7.5"


def test_plain_confirmation_still_reaches_fit(monkeypatch, cv_dir):
    """Paired positive for the replace guard: an ordinary reply is still parsed
    by the confirmation chain and the run proceeds to fit."""
    _wire(
        monkeypatch,
        questions=(),
        parses=[COMPLETE],
        facts=[FIRST_FACTS],
        confirms=[CVConfirmation(confirmed=True)],
    )
    g = build_graph()
    g.invoke({}, _thread())
    g.invoke(Command(resume="a PhD in ML in the UK"), _thread())
    g.invoke(Command(resume=_upload(cv_dir)), _thread())
    state = g.invoke(Command(resume="yes, that's right"), _thread())

    assert state["cv_confirmed"] is True
    assert state["messages"][-2] == ("ai", "VERDICTS")
    assert "This assessment is complete" in state["messages"][-1][1]
    assert "__interrupt__" not in state  # run over: the page will see done/over


def test_confirmed_corrections_re_echo_until_a_clean_affirmation(monkeypatch, cv_dir):
    _wire(
        monkeypatch,
        questions=(),
        parses=[COMPLETE],
        facts=[CVFacts(current_location="Jakarta, Indonesia")],
        confirms=[
            CVConfirmation(
                confirmed=True,
                corrections=CVFacts(current_location="Bandung, Indonesia"),
            ),
            CVConfirmation(
                confirmed=True,
                corrections=CVFacts(current_location=None),
            ),
            CVConfirmation(confirmed=True),
        ],
    )
    g = build_graph()
    g.invoke({}, _thread())
    g.invoke(Command(resume="a PhD in ML in the UK"), _thread())
    g.invoke(Command(resume=_upload(cv_dir)), _thread())

    state = g.invoke(Command(resume="yes, but I now live in Bandung"), _thread())
    assert not state.get("cv_confirmed")
    assert "current location: Bandung, Indonesia" in state["messages"][-1][1]
    assert state["__interrupt__"][0].value == state["messages"][-1][1]

    state = g.invoke(Command(resume="yes, but clear my location"), _thread())
    assert not state.get("cv_confirmed")
    assert "current location: not found" in state["messages"][-1][1]
    assert state["__interrupt__"][0].value == state["messages"][-1][1]

    state = g.invoke(Command(resume="yes, that is correct"), _thread())
    assert state["cv_confirmed"] is True
    assert state["messages"][-2] == ("ai", "VERDICTS")
    assert "This assessment is complete" in state["messages"][-1][1]


# --- story 011: early extraction informs the interview ---------------------


def test_cv_facts_reach_runtime_context_but_not_transcript(monkeypatch, cv_dir):
    facts = CVFacts(
        degree="BSc Mathematics",
        publications=["Paper A"],
        experience=[WorkExperience(
            employer="Acme", title="Data Scientist", tenure="2 yrs", summary="Built NLP models"
        )],
    )
    chat_seen = []
    _wire(
        monkeypatch,
        questions=["Q2"],
        parses=[SlotUpdate(degree_intent="phd")],
        facts=[facts],
        chat_seen=chat_seen,
    )
    g = build_graph()
    g.invoke({}, _thread())
    g.invoke(Command(resume=_upload(cv_dir)), _thread())
    state = g.invoke(Command(resume="a PhD"), _thread())

    prompt = chat_seen[0][-1][1]
    assert len(chat_seen) == 1
    assert "BSc Mathematics" in prompt and "Built NLP models" in prompt
    assert "attribute them to the CV" in prompt
    assert "broad topic such as AI or machine learning is not" in prompt
    assert "latest explicit user clarification controls" in prompt
    assert not any("Built NLP models" in text for _, text in state["messages"])
    assert state.get("field") is None  # CV facts never fill a slot
    assert state["cv_facts"].degree == "BSc Mathematics"  # clarification parsing does not mutate it


def test_threads_without_upload_get_no_cv_context(monkeypatch):
    chat_seen = []
    _wire(
        monkeypatch,
        questions=[],
        parses=[SlotUpdate(degree_intent="phd")],
        chat_seen=chat_seen,
    )
    g = build_graph()
    g.invoke({}, _thread())
    state = g.invoke(Command(resume="a PhD"), _thread())
    assert chat_seen == []
    assert state["messages"][-1] == ("ai", FIXED_INTERVIEW_QUESTIONS["field"])
    assert state.get("field") is None
    assert state.get("cv_facts") is None


def test_user_answer_not_cv_selects_the_field(monkeypatch, cv_dir):
    facts = CVFacts(degree="BSc Mathematics")
    chat_seen = []
    _wire(
        monkeypatch,
        questions=["Q2"],
        parses=[SlotUpdate(field="machine learning")],
        facts=[facts],
        chat_seen=chat_seen,
    )
    g = build_graph()
    g.invoke({}, _thread())
    before = g.invoke(Command(resume=_upload(cv_dir)), _thread())
    assert before.get("field") is None
    after = g.invoke(Command(resume="AI and machine learning"), _thread())
    assert after["field"] == "machine learning"
    assert after["cv_facts"].degree == "BSc Mathematics"
    assert len(chat_seen) == 1


def test_parse_and_extraction_failures_name_the_correct_cause(monkeypatch, cv_dir):
    chat_seen = []
    _wire(
        monkeypatch,
        questions=[],
        parses=[SlotUpdate(degree_intent="phd")],
        chat_seen=chat_seen,
    )
    monkeypatch.setattr(graph_mod, "parse_cv", lambda path: (_ for _ in ()).throw(ValueError("bad")))
    g = build_graph()
    g.invoke({}, _thread())
    state = g.invoke(Command(resume=_upload(cv_dir)), _thread())
    assert state["messages"][-1] == ("ai", CV_PARSE_FAILED)
    assert CV_EXTRACTION_FAILED not in _ai(state)
    state = g.invoke(Command(resume="a PhD"), _thread())
    assert chat_seen == []
    assert state["messages"][-1] == ("ai", FIXED_INTERVIEW_QUESTIONS["field"])
    assert state.get("cv_facts") is None

    for thread, error in (
        ("provider", AllProvidersExhausted("down")),
        ("validation", RuntimeError("unexpected client failure")),
    ):
        _wire(monkeypatch)
        monkeypatch.setattr(graph_mod, "parse_cv", lambda path: "text")
        monkeypatch.setattr(
            graph_mod, "extract_facts",
            lambda text, error=error: (_ for _ in ()).throw(error),
        )
        g = build_graph()
        cfg = {"configurable": {"thread_id": thread}}
        g.invoke({}, cfg)
        state = g.invoke(Command(resume=_upload(cv_dir, f"{thread}.txt")), cfg)
        assert state["messages"][-1] == ("ai", CV_EXTRACTION_FAILED)
        assert CV_PARSE_FAILED not in _ai(state)


def test_second_upload_always_gets_its_own_message(monkeypatch, cv_dir):
    calls = iter([ValueError("bad"), "text"])

    def parse(path):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    _wire(monkeypatch, facts=[FIRST_FACTS])
    monkeypatch.setattr(graph_mod, "parse_cv", parse)
    g = build_graph()
    g.invoke({}, _thread())
    g.invoke(Command(resume=_upload(cv_dir, "bad.txt")), _thread())
    state = g.invoke(Command(resume=_upload(cv_dir, "good.txt")), _thread())
    assert _ai(state)[-2:] == [CV_PARSE_FAILED, CV_RECEIVED]


def test_attachment_only_at_cap_extracts_but_mixed_capped_turn_does_not(monkeypatch, cv_dir):
    calls = []
    g = _at_the_turn_cap(monkeypatch, cv_dir)
    monkeypatch.setattr(graph_mod, "extract_facts", lambda text: calls.append(text) or FIRST_FACTS)
    state = g.invoke(Command(resume=_upload(cv_dir, "alone.txt")), _thread())
    assert len(calls) == 1 and state["messages"][-1] == ("ai", CV_RECEIVED)

    calls.clear()
    g = _at_the_turn_cap(monkeypatch, cv_dir)
    monkeypatch.setattr(graph_mod, "extract_facts", lambda text: calls.append(text) or FIRST_FACTS)
    state = g.invoke(Command(resume="still vague\n" + _upload(cv_dir, "mixed.txt")), _thread())
    assert calls == []
    assert _ai(state)[-2:] == [CV_AFTER_CAP, TURN_CAP_MESSAGE]
    assert CV_RECEIVED not in _ai(state)


def test_mid_interview_parse_skips_intake_then_replacement_reparses(monkeypatch, cv_dir):
    _wire(
        monkeypatch,
        parses=[COMPLETE],
        facts=[FIRST_FACTS, SECOND_FACTS],
        confirms=(),
    )
    g = build_graph()
    g.invoke({}, _thread())
    first = _upload(cv_dir, "first.txt")
    g.invoke(Command(resume=first), _thread())
    state = g.invoke(Command(resume="a PhD in ML in the UK"), _thread())
    assert state["cv_facts"].degree == "BSc Computer Science"
    assert state.get("pending_cv") is None
    assert "BSc Computer Science" in state["messages"][-1][1]

    second = _upload(cv_dir, "second.txt")
    state = g.invoke(Command(resume=second), _thread())
    assert state["cv_facts"].degree == "MSc Computer Science"
    assert "MSc Computer Science" in state["messages"][-1][1]


# --- AC 1 + AC 5, page side -----------------------------------------------


def test_page_holds_the_upload_and_owns_the_post_over_line():
    """The page has no DOM harness in this repo (story 008 round 1), so only the
    contract is pinned here: the auto-send call is gone, a chip element exists,
    the upload control is not disabled by `over`, and the one page-side fixed
    line is present. Behaviour itself is a human check at the done checkpoint."""
    page = (Path(__file__).parents[1] / "static" / "index.html").read_text()

    assert "send((await res.json()).path)" not in page  # AC 1: auto-send removed
    assert 'id="chip"' in page and "attachment = { path:" in page
    assert "cv.disabled = busy;" in page  # AC 5: upload stays live once over
    assert "This chat has already finished, so a new CV can't change anything in it." in page
    # Both paths to `over` must read true, so the line may not name the verdicts:
    # the turn cap also ends the run, with no verdicts and possibly no earlier CV.
    assert "fit verdicts above" not in page
