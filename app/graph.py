"""LangGraph interview graph (story 003): one interviewer node revisited via a
conditional edge until the state is complete, then the shortlister node
(story 004), then CV intake -> confirm -> per-program fit summary (story 005).

Each interviewer visit is one agent turn: receive the user's answer via
``interrupt()`` (so the graph pauses between turns), parse it into slots on the
light tier, then ask the next question on the heavy tier. Drive it with
``graph.invoke({}, config)`` to get the opening question, then
``graph.invoke(Command(resume=<answer>), config)`` per user answer.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.cv import cv_ask_node, cv_confirm_node, cv_intake_node, split_upload
from app.fit import fit_node
from app.model_layer import chat, structured
from app.prompts import (
    CV_AFTER_CAP,
    CV_RECEIVED,
    INTERVIEWER_INSTRUCTIONS,
    PARSER_INSTRUCTIONS,
    PERSONA,
    POSTDOC_DECLINE,
    TURN_CAP_MESSAGE,
)
from app.shortlist import shortlist_node
from app.state import InterviewState, SlotUpdate

MAX_TURNS = 10  # AC 2: cap enforced here, not in the prompt

_REQUIRED_SLOTS = ("degree_intent", "field", "target_regions")


def _next_open_slot(state: InterviewState) -> str | None:
    for name in _REQUIRED_SLOTS:
        if not getattr(state, name):
            return name
    return None


def _interviewer(state: InterviewState) -> dict:
    """One agent turn. On revisits, ``interrupt()`` is the first effect, so the
    node's re-run on resume never repeats a model call."""
    messages = list(state.messages)
    updates: dict = {}
    decline = ""
    upload = None

    if state.turns > 0:  # a question is pending: receive and parse the answer
        answer = interrupt(messages[-1][1])
        # Story 009 AC 2: a CV arriving mid-interview is handled in code. The
        # path never reaches the LLM — it answers no slot, and improvising about
        # a file it cannot open was the live incident.
        upload, answer = split_upload(answer)
        if upload:
            updates["pending_cv"] = upload
        if upload and not answer:
            # The attachment came alone, so it answers nothing: acknowledge it
            # and leave the pending question standing. It is still in the log
            # right above, so restating it only made the user read it twice
            # (fix round 2). Two uploads in a row get one acknowledgement each,
            # never a stacked one.
            if messages[-1] != ("ai", CV_RECEIVED):
                messages.append(("ai", CV_RECEIVED))
            return {"pending_cv": upload, "messages": messages}
        # The question being answered is not always messages[-2]: an
        # acknowledgement can sit between it and this answer (fix round 2).
        question = next(t for r, t in reversed(messages) if r == "ai" and t != CV_RECEIVED)
        messages.append(("human", answer))
        parsed = structured(SlotUpdate, tier="light").invoke(
            [
                ("system", PARSER_INSTRUCTIONS),
                ("human", f"Question asked: {question}\nAnswer: {answer}"),
            ]
        )
        for name in ("degree_intent", "field", "focus_area"):
            if getattr(parsed, name):
                updates[name] = getattr(parsed, name)
        for name in ("target_regions", "constraints"):
            new = [v for v in getattr(parsed, name) if v not in getattr(state, name)]
            if new:
                updates[name] = getattr(state, name) + new
        if parsed.postdoc_interest:  # FR-002: decline is code-enforced, not prompt hope
            decline = POSTDOC_DECLINE + "\n\n"

    merged = state.model_copy(update=updates)
    capped = not merged.is_complete() and state.turns >= MAX_TURNS
    # one Send, two parts: the typed answer counted above, the CV is held and
    # acknowledged here, before the normal next question (fix round). The cap
    # ends the run at END, so nothing ever consumes a held CV — whether it
    # arrived on this turn or an earlier one, say so rather than promise a read
    # that will never happen (fix round 2).
    if capped and merged.pending_cv:
        messages.append(("ai", CV_AFTER_CAP))
    elif upload:
        messages.append(("ai", CV_RECEIVED))
    if merged.is_complete():
        return {**updates, "messages": messages}
    if capped:
        messages.append(("ai", TURN_CAP_MESSAGE))
        return {**updates, "messages": messages, "ended": True}

    known = {
        k: v
        for k, v in merged.model_dump(include=set(SlotUpdate.model_fields) - {"postdoc_interest"}).items()
        if v
    }
    context = f"Captured so far: {known or 'nothing yet'}. Slot to fill next: {_next_open_slot(merged)}."
    reply = chat(tier="heavy").invoke(
        [("system", f"{PERSONA}\n\n{INTERVIEWER_INSTRUCTIONS}"), *messages, ("system", context)]
    )
    # .text, not .content: under the web chat's stream_mode="messages" the model
    # streams internally and the aggregated content can be a block list (story 006)
    messages.append(("ai", decline + reply.text))
    return {**updates, "messages": messages, "turns": state.turns + 1}


def _route(state: InterviewState) -> str:
    if state.is_complete():
        return "shortlister"
    if state.ended:
        return END
    return "interviewer"  # loop: awaits the next answer at interrupt()


def build_graph(checkpointer=None):
    """Compiled interview graph. ``interrupt()`` needs a checkpointer; defaults
    to in-memory (the web-chat story may pass its own)."""
    g = StateGraph(InterviewState)
    g.add_node("interviewer", _interviewer)
    g.add_node("shortlister", shortlist_node)
    g.add_node("cv_ask", cv_ask_node)
    g.add_node("cv_intake", cv_intake_node)
    g.add_node("cv_confirm", cv_confirm_node)
    g.add_node("fit", fit_node)
    g.add_edge(START, "interviewer")
    g.add_conditional_edges("interviewer", _route, ["interviewer", "shortlister", END])
    # story 009 AC 4: the ask is its own superstep, so it is checkpointed before
    # cv_intake's interrupt fires and the server's replay never re-emits it late
    g.add_edge("shortlister", "cv_ask")
    g.add_edge("cv_ask", "cv_intake")
    # story 005: loop until a CV parses, then until the user confirms the facts
    g.add_conditional_edges(
        "cv_intake",
        lambda s: "cv_confirm" if s.cv_facts else "cv_intake",
        ["cv_intake", "cv_confirm"],
    )
    g.add_conditional_edges(  # story 009 AC 5: a new CV re-enters intake
        "cv_confirm",
        lambda s: "cv_intake" if s.pending_cv else "fit" if s.cv_confirmed else "cv_confirm",
        ["cv_intake", "cv_confirm", "fit"],
    )
    g.add_edge("fit", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
