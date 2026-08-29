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
import app.model_layer as model_layer
from app.graph import build_graph
from app.prompts import CV_REQUEST, POSTDOC_DECLINE, TURN_CAP_MESSAGE
from app.state import InterviewState, SlotUpdate


@pytest.fixture(autouse=True)
def _no_live_probes(monkeypatch):
    """Belt and braces: if anything falls through to real chain building, the
    id health probe must not hit the network."""
    monkeypatch.setattr(model_layer, "_probe_cache", {})
    monkeypatch.setattr(model_layer, "_probe_status", lambda provider, model_id: 200)


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
    tiers, ask_calls = _wire(monkeypatch, questions=["Q-intent"], parses=[])
    state = build_graph().invoke({}, _thread())
    assert state["messages"] == [("ai", "Q-intent")]
    assert "__interrupt__" in state  # paused awaiting the user's answer
    assert not state.get("handed_off")
    # the slot context the model was steered with names degree_intent first
    assert "degree_intent" in ask_calls[0][-1][1]


def test_interview_fills_state_and_hands_off(monkeypatch):
    # AC 2 + AC 4: progressive fill over 3 turns (vague field answer narrowed by
    # a follow-up), typed completed state, handoff only once complete. Story
    # 004's shortlister (the node completion routes to) is stubbed out here;
    # tests/test_shortlist.py covers it.
    monkeypatch.setattr(graph_mod, "shortlist_node", lambda s: {"handed_off": True})
    tiers, _ = _wire(
        monkeypatch,
        questions=["Q-intent", "Q-field", "Q-field-narrower", "Q-region"],
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
    assert typed.target_regions == ["Europe", "UK"]
    assert typed.constraints == ["needs funding"]
    assert typed.turns == 4 and typed.turns <= 10
    # AD-7: explicit tiers — heavy for user-facing prose, light for slot parsing
    assert set(tiers["ask"]) == {"heavy"}
    assert set(tiers["parse"]) == {"light"}


def test_postdoc_answer_gets_scope_decline(monkeypatch):
    # AC 3 / FR-002: postdoc interest -> deterministic supported-scope decline,
    # no slots filled, interview continues (no handoff).
    _wire(
        monkeypatch,
        questions=["Q-intent", "Q-intent-again"],
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
    # Dev-note requirement: cap triggers at 10 agent turns. Exactly 10 questions
    # are scripted — an 11th ask would StopIteration. Counterpart to the
    # handoff test above: incomplete state ends at END, never at handoff.
    _wire(
        monkeypatch,
        questions=[f"Q{i}" for i in range(1, 11)],
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
