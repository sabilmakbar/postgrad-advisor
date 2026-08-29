"""CV intake (story 005): local parse (pypdf / plain read), light-tier fact
extraction, then an echo-confirm turn before any judging (AC 1).

The CV file is read in place and never copied; if a later story saves uploads,
they go under data/cv/ (gitignored) only.

TODO (ponytail-style, dev-note privacy flag): extracted CV text DOES enter
free-tier model prompts — acceptable for the client's own CV per the spec
assumption, but a BLOCKER before real users.
"""

from pathlib import Path

from langgraph.types import interrupt
from pydantic import BaseModel, Field
from pypdf import PdfReader

from app.model_layer import structured
from app.prompts import (
    CV_CONFIRM_INSTRUCTIONS,
    CV_EXTRACT_INSTRUCTIONS,
    CV_REQUEST,
    CV_UNREADABLE,
)
from app.state import CVFacts, InterviewState


def parse_cv(path: str) -> str:
    """Local parse — pypdf for .pdf, plain read for .txt (AC 1). Anything else
    is unsupported: the intake node re-asks rather than guessing a format."""
    p = Path(path.strip()).expanduser()
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(p).pages)
    if suffix == ".txt":
        return p.read_text()
    raise ValueError(f"unsupported CV file type {suffix!r}")


def extract_facts(cv_text: str) -> CVFacts:
    """Slot-parsing, so light tier (AD-7)."""
    return structured(CVFacts, tier="light").invoke(
        [("system", CV_EXTRACT_INSTRUCTIONS), ("human", cv_text)]
    )


def render_facts(facts: CVFacts) -> str:
    """Code-rendered echo of the extracted facts + the confirmation question."""
    pubs = "; ".join(facts.publications) or "none found"
    return (
        "Here is what I read from your CV — please check it before I judge fit:\n"
        f"- degree: {facts.degree or 'not found'}\n"
        f"- GPA: {facts.gpa or 'not found'}\n"
        f"- publications: {pubs}\n"
        f"- English test: {facts.english_test or 'not found'}\n"
        "Is this correct? Reply yes, or tell me what to fix."
    )


class CVConfirmation(BaseModel):
    """Light-tier structured output for parsing the user's confirmation reply."""

    confirmed: bool = Field(
        description="True only if the user confirms the shown facts are correct."
    )
    corrections: CVFacts | None = Field(
        None, description="Only the fields the user corrected or added, if any."
    )


def cv_intake_node(state: InterviewState) -> dict:
    """Ask for the CV path, parse locally, extract facts, echo them. A path
    that can't be parsed (or yields no text) re-asks instead of failing —
    the route loops back here while cv_facts is unset."""
    messages = list(state.messages)
    retry = bool(messages) and messages[-1] == ("ai", CV_UNREADABLE)
    path = interrupt(CV_UNREADABLE if retry else CV_REQUEST)
    if not retry:
        messages.append(("ai", CV_REQUEST))
    messages.append(("human", path))
    try:
        text = parse_cv(path)
    except Exception:  # trust boundary: any unreadable user path -> re-ask
        text = ""
    if not text.strip():
        messages.append(("ai", CV_UNREADABLE))
        return {"messages": messages}
    facts = extract_facts(text)
    messages.append(("ai", render_facts(facts)))
    return {"cv_facts": facts, "messages": messages}


def cv_confirm_node(state: InterviewState) -> dict:
    """One confirmation turn (AC 1): fit only runs on user-confirmed facts.
    A correcting reply merges the corrected fields and re-echoes (the route
    loops back here until cv_confirmed)."""
    answer = interrupt(state.messages[-1][1])  # the rendered facts echo
    messages = state.messages + [("human", answer)]
    parsed = structured(CVConfirmation, tier="light").invoke(
        [
            ("system", CV_CONFIRM_INSTRUCTIONS),
            ("human", f"Facts shown:\n{render_facts(state.cv_facts)}\n\nUser reply: {answer}"),
        ]
    )
    if parsed.confirmed:
        return {"cv_confirmed": True, "messages": messages}
    facts = state.cv_facts
    if parsed.corrections:
        updates = {k: v for k, v in parsed.corrections.model_dump().items() if v}
        facts = facts.model_copy(update=updates)
    messages.append(("ai", render_facts(facts)))
    return {"cv_facts": facts, "messages": messages}
