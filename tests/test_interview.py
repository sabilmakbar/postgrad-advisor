"""Interview graph (story 003): scripted fake models, no live API.

Covers: first turn targets degree intent and nothing is handed off before the
required slots fill (AC 1), progressive fill within the turn cap (AC 2),
postdoc decline (AC 3), typed completed state + handoff (AC 4), and the cap
firing at exactly 10 agent turns without handoff.
"""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

import app.graph as graph_mod
from app.graph import _REQUIRED_SLOTS, build_graph
from app.prompts import (
    CV_REQUEST,
    FIRST_QUESTION,
    FIXED_INTERVIEW_QUESTIONS,
    POSTDOC_DECLINE,
    TURN_CAP_MESSAGE,
)
from app.state import InterviewState, SlotUpdate


def _wire(monkeypatch, questions, parses):
    """Replace the graph's model factories with scripted fakes.

    Scripts are strict iterators: an extra ask or parse raises StopIteration,
    so tests also prove call COUNTS. Returns the tiers each factory saw."""
    tiers = {"ask": [], "parse": []}
    ask_calls = []  # prompt messages each ask received
    q_iter, p_iter = iter(questions), iter(parses)

    def fake_chat(tier="heavy"):
        tiers["ask"].append(tier)

        def _ask(msgs):
            ask_calls.append(msgs)
            return AIMessage(content=next(q_iter))

        return RunnableLambda(_ask)

    def fake_structured(schema, tier="heavy"):
        assert schema is SlotUpdate
        tiers["parse"].append(tier)
        return RunnableLambda(lambda _: next(p_iter))

    monkeypatch.setattr(graph_mod, "chat", fake_chat)
    monkeypatch.setattr(graph_mod, "structured", fake_structured)
    return tiers, ask_calls


def _thread():
    return {"configurable": {"thread_id": "t"}}


def test_first_turn_targets_degree_intent_and_pauses(monkeypatch):
    # AC 1: fresh session -> one agent question aimed at degree_intent, then the
    # graph pauses for the answer; nothing handed off / recommended.
    #
    # Turn 0 is FIXED TEXT, not a model call: its input was fully determined
    # (no messages, and the open slot is always _REQUIRED_SLOTS[0]), so the call
    # was constant-input with variable output. `questions=[]` is the load-bearing
    # part — the strict iterator makes any model call on this turn raise
    # StopIteration, so this asserts the quota is not spent, not merely that the
    # text matches.
    tiers, ask_calls = _wire(monkeypatch, questions=[], parses=[])
    state = build_graph().invoke({}, _thread())
    assert state["messages"] == [("ai", FIRST_QUESTION)]
    assert ask_calls == []  # no model was asked anything
    assert "__interrupt__" in state  # paused awaiting the user's answer
    assert not state.get("handed_off")
    # the fixed question still targets the first required slot
    assert _REQUIRED_SLOTS[0] == "degree_intent"
    assert "taught" in FIRST_QUESTION and "research" in FIRST_QUESTION


def test_interview_fills_state_and_hands_off(monkeypatch):
    # AC 2 + AC 4: progressive fill over 3 turns (vague field answer narrowed by
    # a follow-up), typed completed state, handoff only once complete. Story
    # 004's shortlister (the node completion routes to) is stubbed out here;
    # tests/test_shortlist.py covers it.
    monkeypatch.setattr(graph_mod, "shortlist_node", lambda s: {"handed_off": True})
    tiers, ask_calls = _wire(
        monkeypatch,
        questions=["Q-field-narrower"],
        parses=[
            SlotUpdate(degree_intent="phd"),
            SlotUpdate(),  # vague answer: nothing extracted -> follow-up
            SlotUpdate(field="machine learning", focus_area="NLP"),
            SlotUpdate(target_regions=["Europe", "UK"], constraints=["needs funding"]),
        ],
    )
    g = build_graph()
    state = g.invoke({}, _thread())
    for answer in ["a PhD", "something with computers?", "ML, mostly NLP", "Europe or the UK, needs funding"]:
        assert not state.get("handed_off")  # never handed off while incomplete
        state = g.invoke(Command(resume=answer), _thread())

    assert state["handed_off"] is True
    # story 005: after handoff the run pauses at CV intake instead of ending
    assert state["__interrupt__"][0].value == CV_REQUEST
    typed = InterviewState(**{k: v for k, v in state.items() if k in InterviewState.model_fields})
    assert typed.is_complete()
    assert typed.degree_intent == "phd"
    assert typed.field == "machine learning"
    assert typed.focus_area == "NLP"
    assert typed.target_regions == ["Europe", "United Kingdom"]
    assert typed.constraints == ["needs funding"]
    assert typed.turns == 4 and typed.turns <= 10
    assert len(ask_calls) == 1
    assert "Just captured this turn" not in ask_calls[0][-1][1]
    ai_messages = [text for role, text in state["messages"] if role == "ai"]
    assert FIXED_INTERVIEW_QUESTIONS["field"] in ai_messages
    assert FIXED_INTERVIEW_QUESTIONS["target_regions"] in ai_messages
    # AD-7: explicit tiers — heavy for user-facing prose, light for slot parsing
    assert tiers["ask"] == ["heavy"]
    assert set(tiers["parse"]) == {"light"}


@pytest.mark.parametrize(
    ("answer", "parsed"),
    [
        ("near my current location", ["Indonesia"]),
        ("wherever has the best programs", ["Europe"]),
    ],
)
def test_vague_region_answer_rejects_model_invention(monkeypatch, answer, parsed):
    _wire(
        monkeypatch,
        questions=[],
        parses=[
            SlotUpdate(degree_intent="phd", field="computer science"),
            SlotUpdate(target_regions=parsed),
        ],
    )
    g = build_graph()
    g.invoke({}, _thread())
    state = g.invoke(Command(resume="a PhD in computer science"), _thread())
    state = g.invoke(Command(resume=answer), _thread())
    assert state.get("target_regions") in (None, [])
    assert not state.get("handed_off")
    assert state["messages"][-1] == ("ai", FIXED_INTERVIEW_QUESTIONS["target_regions"])


def test_mixed_region_answer_keeps_only_explicit_supported_values(monkeypatch):
    monkeypatch.setattr(graph_mod, "shortlist_node", lambda s: {"handed_off": True})
    _wire(
        monkeypatch,
        questions=[],
        parses=[
            SlotUpdate(degree_intent="phd", field="computer science"),
            SlotUpdate(target_regions=["Europe", "United States", "Australia"]),
        ],
    )
    g = build_graph()
    g.invoke({}, _thread())
    g.invoke(Command(resume="a PhD in computer science"), _thread())
    state = g.invoke(Command(resume="Europe or the US, or wherever is best"), _thread())
    assert state["target_regions"] == ["Europe", "United States"]
    assert state["handed_off"] is True


def test_bare_degree_is_rejected_once_then_accepted(monkeypatch):
    tiers, ask_calls = _wire(
        monkeypatch,
        questions=["Q-clarify-degree"],
        parses=[
            SlotUpdate(degree_intent="masters_research"),
            SlotUpdate(degree_intent="masters_research"),
        ],
    )
    g = build_graph()
    state = g.invoke({}, _thread())
    state = g.invoke(Command(resume="MSc"), _thread())
    assert state.get("degree_intent") is None
    assert state["degree_intent_blocked"] is True
    assert state["messages"][-1] == ("ai", "Q-clarify-degree")

    state = g.invoke(Command(resume="MSc"), _thread())
    assert state["degree_intent"] == "masters_research"
    assert state["messages"][-1] == ("ai", FIXED_INTERVIEW_QUESTIONS["field"])
    assert len(ask_calls) == 1
    assert tiers["parse"] == ["light", "light"]


@pytest.mark.parametrize(
    ("answer", "intent"),
    [
        ("a research master's", "masters_research"),
        ("I want a PhD", "phd"),
        ("Ph.D.", "phd"),
        ("a taught programme", "masters_coursework"),
        ("MBA", "masters_coursework"),
    ],
)
def test_discriminating_degree_terms_fill_slot(monkeypatch, answer, intent):
    _wire(
        monkeypatch,
        questions=[],
        parses=[SlotUpdate(degree_intent=intent)],
    )
    g = build_graph()
    g.invoke({}, _thread())
    state = g.invoke(Command(resume=answer), _thread())
    assert state["degree_intent"] == intent
    assert not state.get("degree_intent_blocked")


def test_captured_degree_uses_fixed_copy_without_heavy_call(monkeypatch):
    heavy_calls = []

    monkeypatch.setattr(
        graph_mod,
        "structured",
        lambda schema, tier="light": RunnableLambda(
            lambda _: SlotUpdate(degree_intent="masters_research")
        ),
    )

    def fake_chat(tier="heavy"):
        def answer(msgs):
            heavy_calls.append(msgs)
            return AIMessage(content="unexpected")
        return RunnableLambda(answer)

    monkeypatch.setattr(graph_mod, "chat", fake_chat)
    g = build_graph()
    g.invoke({}, _thread())
    state = g.invoke(Command(resume="a research master's"), _thread())

    assert state["degree_intent"] == "masters_research"  # canonical state survives
    assert heavy_calls == []
    assert "masters_research" not in state["messages"][-1][1]
    assert state["messages"][-1] == ("ai", FIXED_INTERVIEW_QUESTIONS["field"])


def test_fixed_route_preserves_user_authored_canonical_degree_literal(monkeypatch):
    heavy_calls = []
    monkeypatch.setattr(
        graph_mod,
        "structured",
        lambda schema, tier="light": RunnableLambda(
            lambda _: SlotUpdate(degree_intent="masters_research")
        ),
    )

    def fake_chat(tier="heavy"):
        assert tier == "heavy"

        def answer(msgs):
            heavy_calls.append(msgs)
            return AIMessage(content="Which field interests you?")

        return RunnableLambda(answer)

    monkeypatch.setattr(graph_mod, "chat", fake_chat)
    literal = "I literally wrote masters_research"
    g = build_graph()
    g.invoke({}, _thread())
    state = g.invoke(Command(resume=literal), _thread())

    assert ("human", literal) in state["messages"]
    assert heavy_calls == []
    assert "masters_research" not in state["messages"][-1][1]


def test_fixed_field_copy_keeps_distinct_programme_directions(monkeypatch):
    _, ask_calls = _wire(
        monkeypatch,
        questions=[],
        parses=[SlotUpdate(degree_intent="phd")],
    )
    g = build_graph()
    g.invoke({}, _thread())
    state = g.invoke(Command(resume="a PhD"), _thread())

    question = state["messages"][-1][1]
    assert ask_calls == []
    assert "computer-science foundations" in question
    assert "mathematical or statistical foundations" in question
    assert "application-led or hybrid" in question


def test_postdoc_answer_gets_scope_decline(monkeypatch):
    # AC 3 / FR-002: postdoc interest -> deterministic supported-scope decline,
    # no slots filled, interview continues (no handoff).
    _wire(
        monkeypatch,
        questions=["Q-intent-again"],  # turn 0 is fixed text
        parses=[SlotUpdate(postdoc_interest=True)],
    )
    g = build_graph()
    g.invoke({}, _thread())
    state = g.invoke(Command(resume="actually I'm after a postdoc"), _thread())
    role, content = state["messages"][-1]
    assert role == "ai"
    assert content.startswith(POSTDOC_DECLINE)
    assert state.get("degree_intent") is None
    assert not state.get("handed_off")


def test_turn_cap_ends_gracefully_without_handoff(monkeypatch):
    # Dev-note requirement: cap triggers at 10 agent turns. Exactly 9 questions
    # are scripted, because turn 0 is fixed text and spends no model call — a
    # 10th ask would StopIteration. Counterpart to the
    # handoff test above: incomplete state ends at END, never at handoff.
    _wire(
        monkeypatch,
        questions=[f"Q{i}" for i in range(1, 10)],  # 9 generated: turn 0 is fixed
        parses=[SlotUpdate() for _ in range(10)],
    )
    g = build_graph()
    state = g.invoke({}, _thread())
    for _ in range(10):
        assert "__interrupt__" in state  # still interviewing, not ended early
        state = g.invoke(Command(resume="hmm, not sure"), _thread())

    assert "__interrupt__" not in state  # run ended, no 11th question
    assert state["turns"] == 10
    assert state["ended"] is True
    assert not state.get("handed_off")
    assert state["messages"][-1] == ("ai", TURN_CAP_MESSAGE)
    assert sum(1 for role, _ in state["messages"] if role == "ai") == 11  # 10 Qs + cap notice
