"""CV fit (story 005): fakes for search + LLM, no live calls.

Covers: local parse of the fixture CVs (AC 1); verdict schema, heavy tier and
code-grounded source URLs (AC 3); unknown on empty search with the searched
query shown and NO judge call (AC 4); the <=12 search budget enforced in code
(AC 2); probability scrub proven against a seeded BAD output in the same run
(AC 5, zero-results rule); and graph wiring shortlist -> CV upload -> confirm
(with a correction round) -> fit loop -> final summary.
"""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

import app.cv as cv_mod
import app.fit as fit_mod
import app.graph as graph_mod
import app.model_layer as model_layer
from app.cv import CVConfirmation, parse_cv
from app.fit import _PROB_RE, MAX_SEARCHES, FitVerdict, fit_node
from app.graph import build_graph
from app.prompts import CV_REQUEST, CV_UNREADABLE
from app.requirements_search import SearchResult
from app.state import CVFacts, InterviewState, ShortlistEntry, SlotUpdate


@pytest.fixture(autouse=True)
def _no_live_probes(monkeypatch):
    """Belt and braces: if anything falls through to real chain building, the
    id health probe must not hit the network."""
    monkeypatch.setattr(model_layer, "_probe_cache", {})
    monkeypatch.setattr(model_layer, "_probe_status", lambda provider, model_id: 200)


FIXTURES = Path(__file__).parent / "fixtures"

FACTS = CVFacts(
    degree="BSc Computer Science, University of Testland",
    gpa="3.7 / 4.0",
    publications=['"A Study of Entirely Synthetic Data" — SynthConf 2025'],
    english_test="IELTS 7.5",
)


def _entry(i):
    return ShortlistEntry(
        university=f"Uni-{i}",
        country="Testland",
        rank_context=f"QS world rank {i + 1}",
        degree_type="phd",
        why_it_fits="fits",
    )


def _state(n=2, **over):
    base = dict(
        degree_intent="phd",
        field="computer science",
        target_regions=["Asia"],
        cv_facts=FACTS,
        cv_confirmed=True,
        shortlist=[_entry(i) for i in range(n)],
        messages=[("ai", "facts echo")],
    )
    return InterviewState(**{**base, **over})


def _url(query):
    return f"https://{query.split()[0].lower()}.example/requirements"


def _wire(monkeypatch, verdicts, found=None):
    """Fake search (records queries; `found` maps query->SearchResult, default
    one hit per query) + fake judge (scripted verdicts in call order; records
    tier and prompts). Returns (queries, judge_calls, seen)."""
    queries, judge_calls, seen = [], [], {}

    def fake_search(query):
        queries.append(query)
        if found is not None:
            return found(query)
        return SearchResult(
            query=query,
            text=f"[{_url(query)}] Minimum GPA 3.5 of 4.0; IELTS 7.0 required.",
            sources=[_url(query)],
        )

    def fake_structured(schema, tier="heavy"):
        assert schema is FitVerdict
        seen["tier"] = tier

        def _run(msgs):
            judge_calls.append(msgs)
            return verdicts[len(judge_calls) - 1]  # IndexError = unexpected call

        return RunnableLambda(_run)

    monkeypatch.setattr(fit_mod, "search_requirements", fake_search)
    monkeypatch.setattr(fit_mod, "structured", fake_structured)
    return queries, judge_calls, seen


# --- local parse (AC 1) ------------------------------------------------------


def test_parse_cv_txt_and_pdf():
    assert "GPA: 3.7 / 4.0" in parse_cv(str(FIXTURES / "cv.txt"))
    assert "GPA 3.7/4.0" in parse_cv(str(FIXTURES / "cv.pdf"))


def test_parse_cv_rejects_other_types():
    with pytest.raises(ValueError):
        parse_cv(str(FIXTURES / "cv.docx"))


# --- verdicts (AC 3) ---------------------------------------------------------


def test_verdict_schema_sources_and_tier(monkeypatch):
    # AC 3: valid label, reasons referencing CV + requirements, >=1 source per
    # non-unknown verdict — and the LLM's source is kept only because it exists
    # in the fetched results; heavy tier per FR-011.
    good = FitVerdict(
        label="strong",
        reasons=["GPA 3.7/4.0 meets the stated 3.5 minimum"],
        sources=[_url("Uni-0")],
    )
    invented = FitVerdict(
        label="reach",
        reasons=["IELTS 7.5 is below the stated 8.0 requirement"],
        sources=["https://invented.example/nowhere"],
    )
    queries, judge_calls, seen = _wire(monkeypatch, [good, invented])

    out = fit_node(_state(2))
    v0, v1 = out["fit_verdicts"]
    assert seen["tier"] == "heavy"
    assert len(judge_calls) == 2
    assert v0.label == "strong" and v0.sources == [_url("Uni-0")]
    # invented URL replaced by the real fetched source (zero-results
    # counterpart: the valid URL above survived exactly as given)
    assert v1.label == "reach" and v1.sources == [_url("Uni-1")]
    assert out["searches_used"] == 2
    assert queries[0] == "Uni-0 computer science admission requirements"
    assert all("phd" not in query and "masters_" not in query for query in queries)
    prompt = judge_calls[0][-1][1]
    assert "; PhD." in prompt
    assert "masters_course" not in prompt and "masters_research" not in prompt

    rendered = out["messages"][-1][1]
    for v in (v0, v1):
        assert v.university in rendered and v.label in rendered
        assert v.reasons[0] in rendered and v.sources[0] in rendered
    assert not _PROB_RE.search(rendered)  # AC 5 scan on the full output


def test_canonical_degree_token_never_enters_heavy_fit_prompt_or_reply(monkeypatch):
    bad = FitVerdict(
        label="borderline",
        reasons=["Matches your masters_research goal"],
        sources=[_url("Uni-0")],
    )
    queries, judge_calls, _ = _wire(monkeypatch, [bad])
    out = fit_node(_state(1))

    prompt = "\n".join(text for _, text in judge_calls[0])
    rendered = out["messages"][-1][1]
    assert "masters_research" not in prompt
    assert "masters_coursework" not in prompt
    assert "masters_research" not in out["fit_verdicts"][0].reasons[0]
    assert "masters_research" not in rendered
    assert "Matches your research master's goal" in rendered
    assert queries == ["Uni-0 computer science admission requirements"]


# --- unknown paths (AC 4) ----------------------------------------------------


def test_unknown_when_search_empty(monkeypatch):
    # Empty search -> unknown, searched query shown, judge NEVER called.
    queries, judge_calls, _ = _wire(
        monkeypatch,
        [],
        found=lambda q: SearchResult(query=q, text="", sources=[]),
    )
    out = fit_node(_state(1))
    (v,) = out["fit_verdicts"]
    assert v.label == "unknown" and v.sources == []
    assert queries[0] in v.reasons[0]  # the searched query is shown
    assert judge_calls == []  # never guessed
    assert queries[0] in out["messages"][-1][1]


def test_llm_unknown_still_shows_query(monkeypatch):
    # The judge itself may find no requirements in the fetched text.
    verdict = FitVerdict(label="unknown", reasons=["page was a news article"], sources=[])
    queries, _, _ = _wire(monkeypatch, [verdict])
    out = fit_node(_state(1))
    (v,) = out["fit_verdicts"]
    assert v.label == "unknown" and v.sources == []
    assert queries[0] in v.reasons[0]


# --- search budget (AC 2) ----------------------------------------------------


def test_search_budget_enforced_in_code(monkeypatch):
    # 15 programs, budget 12: exactly 12 searches; the rest are unknown with
    # the budget named — counterpart in the same run: the first 12 DID search.
    verdicts = [
        FitVerdict(label="borderline", reasons=[f"r{i}"], sources=[_url(f"Uni-{i}")])
        for i in range(MAX_SEARCHES)
    ]
    queries, judge_calls, _ = _wire(monkeypatch, verdicts)

    out = fit_node(_state(15))
    assert len(queries) == MAX_SEARCHES == len(judge_calls)
    assert out["searches_used"] == MAX_SEARCHES
    over = out["fit_verdicts"][MAX_SEARCHES:]
    assert [v.label for v in over] == ["unknown"] * 3
    assert all("search budget" in v.reasons[0] for v in over)
    assert [v.label for v in out["fit_verdicts"][:MAX_SEARCHES]] == ["borderline"] * 12


def test_budget_counter_carries_prior_use(monkeypatch):
    # The counter lives in graph state: 11 already used leaves room for 1.
    verdicts = [FitVerdict(label="strong", reasons=["r"], sources=[_url("Uni-0")])]
    queries, _, _ = _wire(monkeypatch, verdicts)
    out = fit_node(_state(3, searches_used=MAX_SEARCHES - 1))
    assert len(queries) == 1
    assert out["searches_used"] == MAX_SEARCHES
    assert [v.label for v in out["fit_verdicts"]] == ["strong", "unknown", "unknown"]


# --- no probabilities (AC 5, AD-4) -------------------------------------------


def test_probability_scrub_catches_seeded_bad_output(monkeypatch):
    # Zero-results rule: seed BAD reasons and prove, in the same run, that the
    # regex catches them (raw text matches) AND the output no longer does.
    bad = [
        "you have a 70% chance of admission",
        "roughly 60 percent odds",
        "chances of 7 in 10",
    ]
    for text in bad:
        assert _PROB_RE.search(text), text  # the regex DOES catch the seeds
    seeded = FitVerdict(
        label="borderline",
        reasons=["GPA 3.7/4.0 meets the 3.5 minimum", *bad],
        sources=[_url("Uni-0")],
    )
    _wire(monkeypatch, [seeded])

    out = fit_node(_state(1))
    (v,) = out["fit_verdicts"]
    rendered = out["messages"][-1][1]
    assert not _PROB_RE.search(rendered)
    assert v.reasons == ["GPA 3.7/4.0 meets the 3.5 minimum"]  # clean one kept


def test_probability_scrub_when_all_reasons_bad(monkeypatch):
    # Every reason seeded bad: verdict still ships without odds, never empty.
    seeded = FitVerdict(
        label="strong", reasons=["a 90% shot"], sources=[_url("Uni-0")]
    )
    _wire(monkeypatch, [seeded])
    out = fit_node(_state(1))
    (v,) = out["fit_verdicts"]
    assert v.reasons and not _PROB_RE.search(out["messages"][-1][1])
    assert v.label == "strong" and v.sources == [_url("Uni-0")]


# --- graph wiring (task 4) ---------------------------------------------------


def test_graph_wiring_shortlist_to_fit_summary(monkeypatch):
    # shortlist -> await CV upload (bad path re-asks) -> echo -> correction
    # round -> confirm -> fit loop -> final summary, in one graph run.
    monkeypatch.setattr(
        graph_mod, "chat", lambda tier="heavy": RunnableLambda(lambda _: AIMessage(content="Q"))
    )
    monkeypatch.setattr(
        graph_mod,
        "structured",
        lambda schema, tier="heavy": RunnableLambda(
            lambda _: SlotUpdate(
                degree_intent="phd", field="computer science", target_regions=["Asia"]
            )
        ),
    )
    monkeypatch.setattr(
        graph_mod,
        "shortlist_node",
        lambda state: {
            "handed_off": True,
            "shortlist": [_entry(0), _entry(1)],
            "messages": state.messages + [("ai", "shortlist")],
        },
    )
    monkeypatch.setattr(cv_mod, "extract_facts", lambda text: FACTS)
    confirmations = [
        CVConfirmation(confirmed=False, corrections=CVFacts(gpa="3.8 / 4.0")),
        CVConfirmation(confirmed=True),
    ]
    confirm_calls = []

    def fake_confirm(msgs):
        confirm_calls.append(msgs)
        return confirmations[len(confirm_calls) - 1]

    monkeypatch.setattr(
        cv_mod,
        "structured",
        lambda schema, tier="heavy": RunnableLambda(fake_confirm),
    )
    verdicts = [
        FitVerdict(label="strong", reasons=["r0"], sources=[_url("Uni-0")]),
        FitVerdict(label="reach", reasons=["r1"], sources=[_url("Uni-1")]),
    ]
    _wire(monkeypatch, verdicts)

    g = build_graph()
    cfg = {"configurable": {"thread_id": "t"}}
    g.invoke({}, cfg)  # opening question
    state = g.invoke(Command(resume="a phd in CS, in Asia"), cfg)
    assert state["__interrupt__"][0].value == CV_REQUEST  # awaiting upload

    state = g.invoke(Command(resume="/nope/missing.txt"), cfg)
    assert state["__interrupt__"][0].value == CV_UNREADABLE  # re-ask, no crash

    state = g.invoke(Command(resume=str(FIXTURES / "cv.txt")), cfg)
    echo = state["__interrupt__"][0].value  # facts echoed for confirmation
    assert "3.7 / 4.0" in echo and "Is this correct?" in echo

    state = g.invoke(Command(resume="GPA is actually 3.8 / 4.0"), cfg)
    assert "3.8 / 4.0" in state["__interrupt__"][0].value  # corrected re-echo

    state = g.invoke(Command(resume="yes, correct"), cfg)
    assert "__interrupt__" not in state
    assert [v.label for v in state["fit_verdicts"]] == ["strong", "reach"]
    assert state["searches_used"] == 2
    role, summary = state["messages"][-1]
    assert role == "ai" and "Uni-0: strong" in summary and _url("Uni-1") in summary
    assert not _PROB_RE.search(summary)
