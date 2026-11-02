"""CV intake (story 005): local parse (pypdf / plain read), light-tier fact
extraction, then an echo-confirm turn before any judging (AC 1).

The CV file is read in place and never copied; if a later story saves uploads,
they go under data/cv/ (gitignored) only.

TODO (ponytail-style, dev-note privacy flag): extracted CV text DOES enter
free-tier model prompts — acceptable for the client's own CV per the spec
assumption, but a BLOCKER before real users.
"""

import re
from datetime import date
from pathlib import Path

from langgraph.types import interrupt
from pydantic import BaseModel, Field
from pypdf import PdfReader

from app.config import CV_DIR
from app.model_layer import structured
from app.prompts import (
    CV_CONFIRM_INSTRUCTIONS,
    CV_EXTRACT_INSTRUCTIONS,
    CV_REQUEST,
    CV_UNREADABLE,
)
from app.state import CVFacts, InterviewState, LocationEvidence, WorkExperience

_MONTHS = {name.lower(): number for number, name in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1
)}
_OPEN_ENDS = {"present", "current", "now", "ongoing"}
_CURRENT_LOCATION_EVIDENCE = re.compile(
    r"\b(?:current\s+(?:location|residence)|(?:currently\s+)?(?:live|living|reside|residing|based)\s+in)\b",
    re.I,
)


def _today() -> date:
    return date.today()


def _month_index(value: str) -> tuple[int, bool] | None:
    value = value.strip().lower()
    if value in _OPEN_ENDS:
        now = _today()
        return now.year * 12 + now.month - 1, False
    match = re.fullmatch(r"([a-z]{3,9})\s+(\d{4})", value)
    if match:
        month = _MONTHS.get(match.group(1)[:3])
        if month:
            return int(match.group(2)) * 12 + month - 1, False
    match = re.fullmatch(r"(\d{4})", value)
    if match:
        return int(match.group(1)) * 12, True
    return None


def _parse_tenure(tenure: str) -> tuple[int | None, tuple[int, int] | None, bool, bool]:
    """Return months, dated interval, approximate, and counted-to-today."""
    duration = re.fullmatch(r"\s*(\d+)\s*(?:yrs?|years?)\s*", tenure, re.I)
    if duration:
        return int(duration.group(1)) * 12, None, True, False
    parts = re.split(r"\s*(?:-|–|—|\bto\b)\s*", tenure.strip(), maxsplit=1, flags=re.I)
    if len(parts) != 2:
        return None, None, False, False
    start, finish = _month_index(parts[0]), _month_index(parts[1])
    if not start or not finish or finish[0] < start[0]:
        return None, None, False, False
    open_end = parts[1].strip().lower() in _OPEN_ENDS
    return finish[0] - start[0], (start[0], finish[0]), start[1] or finish[1], open_end


def derive_experience(facts: CVFacts) -> CVFacts:
    intervals: list[tuple[int, int]] = []
    duration_only = 0
    experience = []
    for position in facts.experience:
        months, interval, _, _ = _parse_tenure(position.tenure)
        experience.append(position.model_copy(update={"months": months}))
        if interval:
            intervals.append(interval)
        elif months is not None:
            duration_only += months
    intervals.sort()
    merged: list[list[int]] = []
    for start, end in intervals:
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    total = sum(end - start for start, end in merged) + duration_only
    return facts.model_copy(
        update={
            "experience": experience,
            "experience_months": total if intervals or duration_only else None,
        }
    )


def select_location_hint(facts: CVFacts) -> LocationEvidence | None:
    """Select the uniquely newest dated location without using source order."""
    if facts.current_location:
        return None
    dated = []
    for item in facts.location_evidence:
        _, interval, _, _ = _parse_tenure(item.tenure)
        if interval:
            location = " ".join(item.location.casefold().split())
            dated.append((interval[1], interval[0], location, item))
    if not dated:
        return None
    newest_end = max(row[0] for row in dated)
    newest = [row for row in dated if row[0] == newest_end]
    if len({row[2] for row in newest}) != 1:
        return None
    newest_start = max(row[1] for row in newest)
    winners = [row[3] for row in newest if row[1] == newest_start]
    return winners[0] if len(winners) == 1 else None


def render_location_hint(item: LocationEvidence) -> str:
    return (
        f"Your latest CV entry suggests {item.location} based on {item.kind}: "
        f"{item.label} ({item.tenure}). Is that still your current location?"
    )


def render_interview_facts(facts: CVFacts) -> str:
    dated, other = [], []
    for order, position in enumerate(facts.experience):
        months, interval, _, _ = _parse_tenure(position.tenure)
        item = (position, months)
        if interval:
            dated.append((interval[1], order, item))
        else:
            other.append((order, item))
    ordered = [item for _, _, item in sorted(dated, key=lambda row: (-row[0], row[1]))]
    ordered += [item for _, item in other]
    positions = "; ".join(
        f"{position.title} — {position.tenure} ({months if months is not None else 'unknown'} months): "
        f"{position.summary}"
        for position, months in ordered[:3]
    ) or "none determinable"
    return (
        "Your CV provisionally reads: "
        f"degree={facts.degree or 'not found'}; publications={len(facts.publications)}; "
        f"current location={facts.current_location or 'not found'}; "
        f"derived total experience={facts.experience_months if facts.experience_months is not None else 'unknown'} months; "
        f"recent positions={positions}."
    )


def split_upload(message: str) -> tuple[str | None, str]:
    """Split a message into (uploaded-CV path, the rest of what the user typed).

    /upload returns ABSOLUTE paths under CV_DIR, so only those count: a path the
    user types by hand stays a normal answer, and ``is_relative_to`` (not a
    string prefix) keeps ``data/cvfoo/x.pdf`` from matching. Lines are checked
    one by one because the composer sends the chip's path on its own line,
    alongside whatever the user typed with it — and that remainder is a real
    message that every stage must still answer (story 009 fix round), so it
    comes back rather than being dropped with the path.

    Trust boundary (fix round 2): an absolute line is resolved before the
    containment check, so ``<CV_DIR>/../../etc/passwd`` no longer counts as an
    upload, and CV_DIR itself is not a CV. A relative line stays relative and so
    never matches — only /upload's absolute paths are intercepted. At most ONE
    attachment exists per Send, so any FURTHER upload-shaped line is dropped
    rather than kept: a path must never reach the slot parser or the log (AC 2).
    """
    root = CV_DIR.resolve()
    path = None
    kept = []
    for line in message.splitlines():
        candidate = Path(line.strip())
        if candidate.is_absolute():
            candidate = candidate.resolve()
        if candidate != root and candidate.is_relative_to(root):
            path = path or str(candidate)
        else:
            kept.append(line)
    return path, "\n".join(kept).strip()


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
    facts = structured(CVFacts, tier="light").invoke(
        [("system", CV_EXTRACT_INSTRUCTIONS), ("human", cv_text)]
    )
    if facts.current_location and not _location_is_grounded(
        cv_text, facts.current_location
    ):
        facts = facts.model_copy(update={"current_location": None})
    facts = facts.model_copy(update={
        "location_evidence": [
            item for item in facts.location_evidence
            if _location_evidence_is_grounded(cv_text, item)
        ]
    })
    return derive_experience(facts)


def _location_is_grounded(cv_text: str, value: str) -> bool:
    """Require the extracted value beside explicit current-location evidence."""
    wanted = " ".join(re.findall(r"[a-z0-9]+", value.casefold()))
    lines = [" ".join(re.findall(r"[a-z0-9]+", line.casefold())) for line in cv_text.splitlines()]
    for index, line in enumerate(lines):
        if _CURRENT_LOCATION_EVIDENCE.search(line) and wanted in line:
            return True
        if line in {"current location", "current residence"}:
            following = next((item for item in lines[index + 1:] if item), "")
            if wanted in following:
                return True
    return False


def _location_evidence_is_grounded(cv_text: str, item: LocationEvidence) -> bool:
    normalize = lambda value: " ".join(re.findall(r"[a-z0-9]+", value.casefold()))
    text, excerpt = normalize(cv_text), normalize(item.source_excerpt)
    excerpt_tokens = set(excerpt.split())
    return bool(
        excerpt
        and excerpt in text
        and all(
            set(normalize(value).split()) <= excerpt_tokens
            for value in (item.label, item.location, item.tenure)
        )
        and not re.search(r"\bremote\b", excerpt)
    )


def render_facts(facts: CVFacts) -> str:
    """Code-rendered echo of the extracted facts + the confirmation question."""
    pubs = "; ".join(facts.publications) or "none found"
    positions = []
    total_approximate = False
    for position in facts.experience:
        months, interval, approximate, open_end = _parse_tenure(position.tenure)
        total_approximate |= approximate or interval is None
        derived = "not derived" if months is None else f"{months} months"
        if open_end:
            derived += ", counted to today"
        elif approximate:
            derived += ", approximate"
        positions.append(
            f"  - {position.title}, {position.employer} — {position.tenure} ({derived})"
        )
    experience = "\n".join(positions) or "  - none found"
    if facts.experience_months is None:
        total = "not derived"
    else:
        total = f"{facts.experience_months} months; overlapping dated roles counted once"
        if total_approximate:
            total += "; approximate, and duration-only roles may overlap dated roles"
    return (
        "Here is what I read from your CV — please check it before I judge fit:\n"
        f"- degree: {facts.degree or 'not found'}\n"
        f"- GPA: {facts.gpa or 'not found'}\n"
        f"- publications: {pubs}\n"
        f"- English test: {facts.english_test or 'not found'}\n"
        f"- current location: {facts.current_location or 'not found'}\n"
        f"- work experience:\n{experience}\n"
        f"- total experience: {total}\n"
        "Is this correct? Reply yes, or tell me what to fix. For work-history "
        "corrections, provide the complete corrected list."
    )


class CVConfirmation(BaseModel):
    """Light-tier structured output for parsing the user's confirmation reply."""

    confirmed: bool = Field(
        description="True only if the user confirms the shown facts are correct."
    )
    corrections: CVFacts | None = Field(
        None, description="Only the fields the user corrected or added, if any."
    )


def review_facts(facts: CVFacts, reply: str) -> tuple[bool, CVFacts]:
    """Light-tier read of a reply to the facts echo: did the user confirm, and
    the facts with any corrections merged in. Shared by cv_confirm_node and by
    cv_intake_node, which applies corrections typed alongside a new CV."""
    parsed = structured(CVConfirmation, tier="light").invoke(
        [
            ("system", CV_CONFIRM_INSTRUCTIONS),
            ("human", f"Facts shown:\n{render_facts(facts)}\n\nUser reply: {reply}"),
        ]
    )
    has_corrections = bool(
        parsed.corrections and parsed.corrections.model_fields_set
    )
    if has_corrections:
        updates = {
            name: getattr(parsed.corrections, name)
            for name in parsed.corrections.model_fields_set
        }
        updates.pop("experience_months", None)
        facts = derive_experience(facts.model_copy(update=updates))
    return parsed.confirmed and not has_corrections, facts


def cv_ask_node(state: InterviewState) -> dict:
    """Write the CV ask to state — and nothing else (story 009, AC 4).

    The ask used to be appended inside cv_intake_node, next to its
    ``interrupt()``. State writes only commit when a node RETURNS, so on the
    first run the interrupt escaped before the append ever happened, and the
    append landed on the resume re-run — after the server had captured its
    replay snapshot. The ask therefore re-emitted one turn late as a dead turn.
    Swapping the two lines cannot fix that; only a PRIOR superstep can, which is
    this node. A pending upload needs no ask at all (AC 3).

    The re-ask after an unreadable path is NOT written here: cv_intake_node
    returns it itself, so it is already committed for the same reason.
    """
    if state.pending_cv:
        return {}
    return {"messages": state.messages + [("ai", CV_REQUEST)]}


def cv_intake_node(state: InterviewState) -> dict:
    """Parse the CV locally, extract facts, echo them. The path comes from a
    held upload (AC 3, AC 5) or from the answer to cv_ask's question. A path
    that can't be parsed (or yields no text) re-asks here and clears cv_facts,
    so the route loops straight back instead of failing.

    Text sent WITH the CV (held from an earlier stage or typed here) is a
    correction and is applied to the fresh extraction (story 009 fix round)."""
    path, extra = state.pending_cv, state.pending_correction
    if path and state.cv_parsed_path == path:
        return {
            "pending_cv": None,
            "pending_correction": None,
            "messages": state.messages + [("ai", render_facts(state.cv_facts))],
        }
    if not path:
        path, extra = split_upload(interrupt(state.messages[-1][1]))
        if not path:  # the ask also accepts a path the user types by hand
            path, extra = extra, None
    messages = state.messages + [("human", path)]
    try:
        text = parse_cv(path)
    except Exception:  # trust boundary: any unreadable user path -> re-ask
        text = ""
    if not text.strip():
        messages.append(("ai", CV_UNREADABLE))
        return {"cv_facts": None, "pending_cv": None, "pending_correction": None, "messages": messages}
    facts = extract_facts(text)
    if extra:  # corrections belong to the NEW facts, so they are applied here
        _, facts = review_facts(facts, extra)
    messages.append(("ai", render_facts(facts)))
    return {"cv_facts": facts, "pending_cv": None, "pending_correction": None, "messages": messages}


def cv_confirm_node(state: InterviewState) -> dict:
    """One confirmation turn (AC 1): fit only runs on user-confirmed facts.
    A correcting reply merges the corrected fields and re-echoes (the route
    loops back here until cv_confirmed)."""
    answer = interrupt(state.messages[-1][1])  # the rendered facts echo
    path, answer = split_upload(answer)
    if path:  # story 009 AC 5: another CV here means "use this one instead"
        # Corrections typed with it apply to the REPLACEMENT extraction, not to
        # the facts being thrown away, so they travel with the path to intake.
        return {"pending_cv": path, "pending_correction": answer or None}
    messages = state.messages + [("human", answer)]
    confirmed, facts = review_facts(state.cv_facts, answer)
    if confirmed:
        return {"cv_confirmed": True, "messages": messages}
    messages.append(("ai", render_facts(facts)))
    return {"cv_facts": facts, "messages": messages}
