"""LangGraph interview graph (story 003): one interviewer node revisited via a
conditional edge until the state is complete, then the shortlister node
(story 004), then CV intake -> confirm -> per-program fit summary (story 005).

Each interviewer visit is one agent turn: receive the user's answer via
``interrupt()`` (so the graph pauses between turns), parse it into slots on the
light tier, then select fixed routine copy or ask the heavy tier for a tailored
follow-up. Drive it with
``graph.invoke({}, config)`` to get the opening question, then
``graph.invoke(Command(resume=<answer>), config)`` per user answer.
"""

import re

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.cv import (
    cv_ask_node,
    cv_confirm_node,
    cv_intake_node,
    extract_facts,
    parse_cv,
    render_interview_facts,
    render_location_hint,
    select_location_hint,
    split_upload,
)
from app.dataset import canonical_region, load_dataset, regions_in_text
from app.fit import FIT_BATCH_SIZE, fit_node, fit_summary, search_limit
from app.model_layer import chat, structured
from app.prompts import (
    CV_AFTER_CAP,
    CV_EXTRACTION_FAILED,
    CV_FIXED_ACKS,
    CV_PARSE_FAILED,
    CV_RECEIVED,
    DEGREE_INTENT_CLARIFICATION,
    FIRST_QUESTION,
    FIT_MORE_PROMPT,
    fit_budget_prompt,
    FIXED_INTERVIEW_QUESTIONS,
    INTERVIEWER_INSTRUCTIONS,
    LOCATION_INFERENCE_UNAVAILABLE,
    LOCATION_NAME_REQUIRED,
    PARSER_INSTRUCTIONS,
    PERSONA,
    POSTDOC_DECLINE,
    REGION_PARSER_INSTRUCTIONS,
    REGION_PROPOSAL_UNAVAILABLE,
    REGION_SCOPE_UNSUPPORTED,
    TURN_CAP_MESSAGE,
    current_location_region_question,
    region_proposal_question,
)
from app.shortlist import degree_intent_label, humanize_degree_text, shortlist_node
from app.state import (
    CVFacts,
    InterviewState,
    LocationEvidence,
    ProgramFit,
    ShortlistEntry,
    SlotUpdate,
    WorkExperience,
)

MAX_TURNS = 10  # AC 2: cap enforced here, not in the prompt

# LangGraph 1.2 warns on deserializing unregistered types from a checkpoint and
# will block them outright in a later release. Declaring ours keeps the upgrade
# from breaking us. Every app.state model is listed, not just the ones on the
# state today: an omitted type does NOT raise — it silently comes back as a
# plain dict, so a partial allowlist trades a loud warning for a quiet type
# downgrade. tests/test_graph_serde.py fails if a model is added and not listed.
CHECKPOINT_TYPES = (
    CVFacts,
    InterviewState,
    LocationEvidence,
    ProgramFit,
    ShortlistEntry,
    SlotUpdate,
    WorkExperience,
)
_SERDE = JsonPlusSerializer(allowed_msgpack_modules=list(CHECKPOINT_TYPES))

_REQUIRED_SLOTS = ("degree_intent", "field", "target_regions")


_DEGREE_TERMS = {
    "taught",
    "coursework",
    "course based",
    "research",
    "thesis",
    "dissertation",
    "phd",
    "doctoral",
    "doctorate",
    "doctor of philosophy",
    "dphil",
    "mres",
    "mphil",
    "mba",
}


def _has_degree_term(answer: str) -> bool:
    text = re.sub(r"\bph\s*[.-]?\s*d\b", "phd", answer.lower())
    normalised = " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())
    padded = f" {normalised} "
    return any(f" {term} " in padded for term in _DEGREE_TERMS)


def _next_open_slot(state: InterviewState) -> str | None:
    for name in _REQUIRED_SLOTS:
        if not getattr(state, name):
            return name
    return None


def _fixed_interview_question(state: InterviewState) -> str:
    return FIXED_INTERVIEW_QUESTIONS[_next_open_slot(state)]


def _read_upload(path: str) -> tuple[dict, str]:
    try:
        text = parse_cv(path)
        if not text.strip():
            raise ValueError("no extractable text")
    except Exception:
        return {
            "pending_cv": path,
            "cv_facts": None,
            "cv_parsed_path": None,
        }, CV_PARSE_FAILED
    try:
        facts = extract_facts(text)
    except Exception:
        return {
            "pending_cv": path,
            "cv_facts": None,
            "cv_parsed_path": None,
        }, CV_EXTRACTION_FAILED
    return {
        "pending_cv": path,
        "cv_facts": facts,
        "cv_parsed_path": path,
    }, CV_RECEIVED


def _interviewer(state: InterviewState) -> dict:
    """One agent turn. On revisits, ``interrupt()`` is the first effect, so the
    node's re-run on resume never repeats a model call."""
    # Turn 0 is deterministic — nothing captured, so the slot is always the
    # first required one and the model would be asked a constant question. Fixed
    # text costs no quota and removes the rate-limiter wait from the landing.
    # The route sends us straight back here, where turns > 0 now interrupts on
    # this question exactly as it would on a generated one.
    if state.turns == 0:
        return {"messages": [("ai", FIRST_QUESTION)], "turns": 1}

    messages = list(state.messages)
    updates: dict = {}
    decline = ""
    upload = None

    # turns >= 1 here: turn 0 returned above. A question is pending, so
    # receive and parse the answer to it.
    answer = interrupt(messages[-1][1])
    if state.location_hint_pending:
        normalized = " ".join(str(answer).casefold().strip().split())
        hint = select_location_hint(state.cv_facts) if state.cv_facts else None
        affirmative = {"yes", "y", "yeah", "correct", "still correct", "that's right"}
        negative = {"no", "n", "nope", "not anymore"}
        corrected = regions_in_text(answer)
        is_correction = bool(corrected) or bool(
            re.search(r"\b(?:live|living|based|reside|residing|moved)\b", normalized)
        )
        is_negative = normalized in negative or any(
            re.match(rf"^{re.escape(prefix)}[,.;:]", normalized)
            or (is_correction and normalized.startswith(prefix + " "))
            for prefix in negative
        )
        if normalized not in affirmative and not is_negative:
            return {
                "messages": messages + [("ai", render_location_hint(hint))],
            }
        facts = state.cv_facts
        if normalized in affirmative and hint:
            facts = facts.model_copy(update={"current_location": hint.location})
        elif is_negative:
            if corrected:
                facts = facts.model_copy(update={"current_location": corrected[0]})
        region_question = (
            current_location_region_question(facts.current_location)
            if facts.current_location
            else FIXED_INTERVIEW_QUESTIONS["target_regions"]
        )
        return {
            "cv_facts": facts,
            "location_hint_pending": False,
            "location_hint_rejected": is_negative,
            "messages": messages + [
                ("human", answer),
                ("ai", region_question),
            ],
            "turns": state.turns + 1,
        }
    # Story 009 AC 2: a CV arriving mid-interview is handled in code. The
    # path never reaches the LLM — it answers no slot, and improvising about
    # a file it cannot open was the live incident.
    upload, answer = split_upload(answer)
    if upload:
        updates["pending_cv"] = upload
        updates["pending_region_countries"] = []
        updates["pending_region_label"] = None
    if upload and not answer:
        # The attachment came alone, so it answers nothing: acknowledge it
        # and leave the pending question standing. It is still in the log
        # right above, so restating it only made the user read it twice
        # (fix round 2). Two uploads in a row get one acknowledgement each,
        # never a stacked one.
        extracted, note = _read_upload(upload)
        updates.update(extracted)
        if messages[-1] != ("ai", note):
            messages.append(("ai", note))
        return {**updates, "messages": messages}
    # The question being answered is not always messages[-2]: an
    # acknowledgement can sit between it and this answer (fix round 2).
    question = next(t for r, t in reversed(messages) if r == "ai" and t not in CV_FIXED_ACKS)
    messages.append(("human", answer))
    countries = {u.country for u in load_dataset().universities}
    parser_messages = [("system", PARSER_INSTRUCTIONS)]
    if _next_open_slot(state) == "target_regions":
        parser_messages.append(("system", (
            f"{REGION_PARSER_INSTRUCTIONS}\n"
            f"Confirmed current location: {getattr(state.cv_facts, 'current_location', None) if not upload else None}.\n"
            f"Pending proposal: {state.pending_region_label}: {state.pending_region_countries if not upload else []}.\n"
            f"Available dataset countries: {sorted(countries)}.\n"
            f"Recent conversation: {messages[-6:]!r}"
        )))
    parser_messages.append(("human", f"Question asked: {question}\nAnswer: {answer}"))
    parsed = structured(SlotUpdate, tier="light").invoke(parser_messages)
    if (
        parsed.degree_intent
        and not state.degree_intent
        and not state.degree_intent_blocked
        and not _has_degree_term(answer)
    ):
        parsed = parsed.model_copy(update={"degree_intent": None})
        updates["degree_intent_blocked"] = True
    captured: dict[str, object] = {}
    for name in ("degree_intent", "field", "focus_area"):
        if getattr(parsed, name):
            updates[name] = getattr(parsed, name)
            captured[name] = getattr(parsed, name)
    supported_regions = set(regions_in_text(answer))
    literal_regions = [
        canonical
        for value in parsed.target_regions
        if (canonical := canonical_region(value)) in supported_regions
    ]
    region_turn = _next_open_slot(state) == "target_regions"
    if parsed.region_scope_unsupported or parsed.region_proposal:
        literal_regions = []
    confirmed_countries = []
    if region_turn and state.pending_region_countries and not upload:
        if (
            parsed.region_confirmation == "accept"
            and not parsed.region_scope_unsupported and not literal_regions
        ):
            confirmed_countries = [c for c in state.pending_region_countries if c in countries]
        if parsed.region_confirmation or literal_regions or parsed.region_proposal or parsed.region_label:
            updates.update(pending_region_countries=[], pending_region_label=None)
    parsed_regions = list(dict.fromkeys(literal_regions + confirmed_countries))
    for name in ("target_regions", "constraints"):
        values = parsed_regions if name == "target_regions" else getattr(parsed, name)
        new = [v for v in values if v not in getattr(state, name)]
        if new:
            updates[name] = getattr(state, name) + new
            captured[name] = new
    if parsed.postdoc_interest:  # FR-002: decline is code-enforced, not prompt hope
        decline = POSTDOC_DECLINE + "\n\n"

    merged = state.model_copy(update=updates)
    capped = not merged.is_complete() and state.turns >= MAX_TURNS
    upload_note = None
    if upload and (merged.is_complete() or not capped):
        extracted, upload_note = _read_upload(upload)
        updates.update(extracted)
        merged = state.model_copy(update=updates)
    # one Send, two parts: the typed answer counted above, the CV is held and
    # acknowledged here, before the normal next question (fix round). The cap
    # ends the run at END, so nothing ever consumes a held CV — whether it
    # arrived on this turn or an earlier one, say so rather than promise a read
    # that will never happen (fix round 2).
    if capped and merged.pending_cv:
        messages.append(("ai", CV_AFTER_CAP))
    elif upload:
        messages.append(("ai", upload_note))
    if merged.is_complete():
        return {**updates, "messages": messages}
    if capped:
        messages.append(("ai", TURN_CAP_MESSAGE))
        return {**updates, "messages": messages, "ended": True}

    reconcile_focus = (
        "focus_area" in captured
        and "field" not in captured
        and bool(merged.field)
        and _next_open_slot(merged) == "target_regions"
    )
    has_profile_evidence = bool(
        merged.cv_facts
        and (
            merged.cv_facts.degree
            or merged.cv_facts.publications
            or merged.cv_facts.experience
        )
    )
    if (
        captured and not has_profile_evidence and not reconcile_focus
        and _next_open_slot(merged) != "target_regions"
    ):
        messages.append(("ai", decline + _fixed_interview_question(merged)))
        return {**updates, "messages": messages, "turns": state.turns + 1}

    if not reconcile_focus and _next_open_slot(merged) == "target_regions":
        current_location = getattr(merged.cv_facts, "current_location", None)
        if parsed.region_scope_unsupported:
            messages.append(("ai", REGION_SCOPE_UNSUPPORTED))
            return {**updates, "messages": messages, "turns": state.turns + 1,
                    "pending_region_countries": [], "pending_region_label": None}
        label = parsed.region_label or "your preferred study area"
        normalized_label = " ".join(re.findall(r"[a-z0-9]+", (parsed.region_label or "").casefold()))
        named_scope = bool(normalized_label) and (
            f" {normalized_label} " in f" {' '.join(re.findall(r'[a-z0-9]+', answer.casefold()))} "
        )
        proposed = list(dict.fromkeys(
            canonical for value in parsed.region_proposal
            if (canonical := canonical_region(value, countries)) in countries
        ))
        if proposed and (current_location or named_scope) and not upload:
            messages.append(("ai", region_proposal_question(label, proposed)))
            return {**updates, "messages": messages, "turns": state.turns + 1,
                    "pending_region_countries": proposed, "pending_region_label": label}
        if not proposed and (parsed.region_proposal or named_scope):
            messages.append(("ai", REGION_PROPOSAL_UNAVAILABLE))
            return {**updates, "messages": messages, "turns": state.turns + 1}
        if merged.pending_region_countries:
            messages.append(("ai", region_proposal_question(
                merged.pending_region_label or "your preferred study area",
                merged.pending_region_countries,
            )))
            return {**updates, "messages": messages, "turns": state.turns + 1}
        if current_location:
            messages.append(("ai", decline + current_location_region_question(current_location)))
            return {**updates, "messages": messages, "turns": state.turns + 1}
        hint = select_location_hint(merged.cv_facts) if merged.cv_facts else None
        if hint and not merged.location_hint_rejected:
            messages.append(("ai", decline + render_location_hint(hint)))
            return {**updates, "messages": messages, "location_hint_pending": True}
        answer_text = answer.casefold()
        asked_to_check_cv = bool(
            re.search(r"\b(?:cv|resume)\b", answer_text)
            and re.search(r"\b(?:check|look|read|find|infer|guess|identify|tell|see)\b", answer_text)
        ) or bool(re.search(r"\bbased on (?:my |the )?(?:cv|resume)\b", answer_text))
        prefers_nearby = bool(
            re.search(r"\b(?:near|nearby|close|closer|around)\b", answer_text)
        )
        if question in {LOCATION_INFERENCE_UNAVAILABLE, LOCATION_NAME_REQUIRED}:
            next_question = LOCATION_NAME_REQUIRED
        elif asked_to_check_cv or prefers_nearby:
            next_question = (
                LOCATION_INFERENCE_UNAVAILABLE if merged.cv_facts else LOCATION_NAME_REQUIRED
            )
        else:
            next_question = _fixed_interview_question(merged)
        messages.append(("ai", decline + next_question))
        return {**updates, "messages": messages, "turns": state.turns + 1}

    known = {
        k: v
        for k, v in merged.model_dump(include=set(SlotUpdate.model_fields) - {"postdoc_interest"}).items()
        if v
    }
    if "degree_intent" in known:
        known["degree_intent"] = degree_intent_label(known["degree_intent"])
    if "degree_intent" in captured:
        captured["degree_intent"] = degree_intent_label(captured["degree_intent"])
    next_step = _next_open_slot(merged)
    if reconcile_focus:
        next_step = (
            "before asking target regions, reconcile the newly stated focus area "
            "with the previously captured field; ask whether to keep or revise that field"
        )
    context = f"Captured so far: {known or 'nothing yet'}. Slot to fill next: {next_step}."
    if captured:
        context += (
            f" Just captured this turn: {captured}. Treat this as an "
            "interpretation the user can correct."
        )
    if merged.cv_facts:
        context += (
            " " + render_interview_facts(merged.cv_facts) +
            " These are provisional facts from the CV: attribute them to the CV, never use them "
            "to fill or overwrite a slot, and ask a direct clarification only for a concrete factual "
            "conflict. The latest explicit user clarification controls interview prose and slots, but "
            "tell them factual CV corrections must also be repeated at the later CV confirmation gate. "
            "If the field slot is open, propose plausible directions from the CV rather than asking "
            "blindly. A broad topic such as AI or machine learning is not itself a chosen programme "
            "direction; when several are plausible, offer computer-science foundations, mathematical/"
            "statistical foundations, and application-led study as distinct choices. "
            "If the target-regions slot is open and the CV states a current location, you may suggest "
            "nearby concrete dataset countries or continents, but label them as suggestions and require "
            "the user to choose one. The CV has been read and acknowledged already: do not comment on "
            "the attachment or say you cannot open files."
        )
    elif merged.pending_cv:
        context += (
            " A CV arrived but is not readable or has not been read. The app already explained why. "
            "Do not repeat it, apologise, offer to open the file, claim to have read it, or say you cannot "
            "view files."
        )
    reply = chat(tier="heavy").invoke(
        [("system", f"{PERSONA}\n\n{INTERVIEWER_INSTRUCTIONS}"), *messages, ("system", context)]
    )
    # .text, not .content: under the web chat's stream_mode="messages" the model
    # streams internally and the aggregated content can be a block list (story 006)
    reply_text = humanize_degree_text(reply.text)
    # A vague answer can leave degree_intent open. Do not let the model repeat
    # the pending opening question verbatim; narrow the decision instead.
    if (
        _next_open_slot(merged) == "degree_intent"
        and " ".join(reply_text.casefold().split())
        == " ".join(question.casefold().split())
    ):
        reply_text = DEGREE_INTENT_CLARIFICATION
    messages.append(("ai", decline + reply_text))
    return {**updates, "messages": messages, "turns": state.turns + 1}


def _route(state: InterviewState) -> str:
    if state.is_complete():
        return "shortlister"
    if state.ended:
        return END
    return "interviewer"  # loop: awaits the next answer at interrupt()


def _fit_batch_route(state: InterviewState) -> str:
    if state.fit_cursor >= len(state.shortlist):
        return "fit_summary"
    return "fit_more_ask"


def _fit_more_ask(state: InterviewState) -> dict:
    remaining = min(FIT_BATCH_SIZE, len(state.shortlist) - state.fit_cursor)
    prompt = (
        fit_budget_prompt(state.searches_used, remaining * 2, remaining)
        if state.searches_used >= search_limit(state) else FIT_MORE_PROMPT
    )
    return {"messages": state.messages + [("ai", prompt)]}


def _fit_more_decide(state: InterviewState) -> dict:
    answer = interrupt(state.messages[-1][1])
    normalized = " ".join(str(answer).lower().strip().split())
    if normalized in {"yes", "y", "continue", "please continue"}:
        extra = state.additional_searches
        if state.searches_used >= search_limit(state):
            extra += 2 * min(FIT_BATCH_SIZE, len(state.shortlist) - state.fit_cursor)
        return {"fit_continue": True, "additional_searches": extra,
                "messages": state.messages + [("human", answer)]}
    if normalized in {"no", "n", "stop", "no thanks", "no thank you", "skip", "skip it", "finish"}:
        return {"fit_continue": False, "messages": state.messages + [("human", answer)]}
    return {
        "fit_continue": None,
        "messages": state.messages + [("human", answer), ("ai", state.messages[-1][1])],
    }


def _fit_more_route(state: InterviewState) -> str:
    if state.fit_continue is True:
        return "fit_batch"
    if state.fit_continue is False:
        return "fit_summary"
    return "fit_more_decide"


def build_graph(checkpointer=None):
    """Compiled interview graph. ``interrupt()`` needs a checkpointer; defaults
    to in-memory (the web-chat story may pass its own)."""
    g = StateGraph(InterviewState)
    g.add_node("interviewer", _interviewer)
    g.add_node("shortlister", shortlist_node)
    g.add_node("cv_ask", cv_ask_node)
    g.add_node("cv_intake", cv_intake_node)
    g.add_node("cv_confirm", cv_confirm_node)
    g.add_node("fit_batch", fit_node)
    g.add_node("fit_more_ask", _fit_more_ask)
    g.add_node("fit_more_decide", _fit_more_decide)
    g.add_node("fit_summary", fit_summary)
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
        lambda s: "cv_intake" if s.pending_cv else "fit_batch" if s.cv_confirmed else "cv_confirm",
        ["cv_intake", "cv_confirm", "fit_batch"],
    )
    g.add_conditional_edges("fit_batch", _fit_batch_route, ["fit_more_ask", "fit_summary"])
    g.add_edge("fit_more_ask", "fit_more_decide")
    g.add_conditional_edges(
        "fit_more_decide", _fit_more_route, ["fit_batch", "fit_more_decide", "fit_summary"]
    )
    g.add_edge("fit_summary", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver(serde=_SERDE))
