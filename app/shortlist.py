"""Shortlister node: code filters and ranks; the LLM only annotates."""

import re

from pydantic import BaseModel, ConfigDict, Field

from app.dataset import University, load_dataset
from app.model_layer import structured
from app.prompts import (
    FIELD_NOT_IN_REGIONS,
    FIELD_TOO_FEW,
    FIELD_UNRESOLVED,
    PERSONA,
    SHORTLIST_INSTRUCTIONS,
)
from app.state import InterviewState, ShortlistEntry

MIN_SIZE, MAX_SIZE = 8, 12
TILT_WEIGHT = 15.0
MISSING_ANNOTATION = "No additional model annotation was returned."

_DEGREE_MAP = {  # interview intent -> dataset degree type
    "masters_coursework": "masters_course",
    "masters_research": "masters_research",
    "phd": "phd",
}

_DEGREE_LABEL = {
    "masters_course": "taught master's",
    "masters_research": "research master's",
    "phd": "PhD",
}

_DEGREE_TEXT_LABEL = {
    **{intent: _DEGREE_LABEL[degree] for intent, degree in _DEGREE_MAP.items()},
    **_DEGREE_LABEL,
}
CANONICAL_DEGREE_TOKENS = tuple(
    sorted(_DEGREE_TEXT_LABEL, key=len, reverse=True)
)

FIELD_BUCKETS = {
    "arts_humanities",
    "business",
    "computer_science",
    "economics",
    "engineering",
    "law",
    "life_sciences",
    "medicine",
    "natural_sciences",
    "social_sciences",
}

FIELD_ALIASES = {
    "ai": "computer_science",
    "artificial_intelligence": "computer_science",
    "ml": "computer_science",
    "machine_learning": "computer_science",
    "deep_learning": "computer_science",
    "nlp": "computer_science",
    "computer_vision": "computer_science",
    "robotics": "computer_science",
    "data_science": "computer_science",
    "software_engineering": "computer_science",
    "hci": "computer_science",
    "cybersecurity": "computer_science",
    "statistics": "natural_sciences",
    "mathematics": "natural_sciences",
    "physics": "natural_sciences",
    "chemistry": "natural_sciences",
    "bioinformatics": "life_sciences",
    "genomics": "life_sciences",
    "neuroscience": "life_sciences",
    "finance": "business",
    "marketing": "business",
    "mba": "business",
}


def degree_label(degree_type: str) -> str:
    return _DEGREE_LABEL[degree_type]


def degree_intent_label(degree_intent: str) -> str:
    return degree_label(_DEGREE_MAP[degree_intent])


def humanize_degree_text(text: str) -> str:
    for token in CANONICAL_DEGREE_TOKENS:
        text = re.sub(
            rf"(?<!\w){re.escape(token)}(?!\w)",
            _DEGREE_TEXT_LABEL[token],
            text,
            flags=re.IGNORECASE,
        )
    return text


def _resolve_field(field: str) -> tuple[str, bool]:
    normalised = field.strip().lower().replace(" ", "_")
    if normalised in FIELD_ALIASES:
        return FIELD_ALIASES[normalised], True
    for token in normalised.split("_"):
        if token in FIELD_ALIASES:
            return FIELD_ALIASES[token], True
    return normalised, False


def _canonical_field_match(field: str, u: University) -> bool:
    return any(field == t or field in t or t in field for t in u.fields)


def _field_match(field: str, u: University) -> bool:
    resolved, _ = _resolve_field(field)
    return _canonical_field_match(resolved, u)


def _region_match(regions: list[str], u: University) -> bool:
    # ponytail: exact/containment match only — "UK" won't hit "United Kingdom";
    # misses fall through to the explicit widening path. Add an alias map if
    # real interviews miss too often.
    places = (u.country.lower(), u.continent.lower())
    for r in (x.strip().lower() for x in regions):
        if any(r == p or r in p or p in r for p in places):
            return True
    return False


def _rank_key(state: InterviewState, u: University) -> tuple[bool, bool, bool, float]:
    # This is a deliberate bounded heuristic, not evidence that a degree is
    # offered. This dataset's tilt spans -64.7..93.5 (30.8 SD); limiting it to
    # +/-15 keeps it from becoming an unbounded primary sort key. There is no
    # labelled programme-fit data to calibrate 15; changing it needs a new
    # measured decision.
    direction = 1 if state.degree_intent == "masters_coursework" else -1
    adjusted = (
        u.base_score
        + direction
        * TILT_WEIGHT
        * (u.tilt if u.tilt is not None else 0.0)
        / 100.0
        if u.base_score is not None
        else 0.0
    )
    return (
        not _region_match(state.target_regions, u),
        not _field_match(state.field, u),
        u.base_score is None,
        -adjusted,
    )


def _rank_context(u: University) -> str:
    if u.rank is None:  # ranking context comes from the note, not a number
        return u.note or "unranked in this QS cycle"
    return f"QS {u.rank_scope} rank {u.rank}"


def prefilter(
    state: InterviewState, universities: list[University]
) -> tuple[list[University], list[str]]:
    """Deterministic pre-filter: degree type + region + field over the dataset.
    Widens explicitly when fewer than MIN_SIZE match — field relaxed first,
    then region (AC 3). Returns (rank-sorted candidates, widening notes)."""
    degree = _DEGREE_MAP[state.degree_intent]
    pool = [u for u in universities if degree in u.degree_types]
    in_region = [u for u in pool if _region_match(state.target_regions, u)]
    exact = [u for u in in_region if _field_match(state.field, u)]
    canonical, aliased = _resolve_field(state.field)
    exists_in_dataset = aliased or any(
        _canonical_field_match(canonical, u) for u in pool
    )

    candidates, notes = exact, []
    if len(exact) < MIN_SIZE:
        if not exists_in_dataset:
            notes.append(FIELD_UNRESOLVED)
        elif not exact:
            notes.append(FIELD_NOT_IN_REGIONS)
        else:
            notes.append(FIELD_TOO_FEW.format(count=len(exact)))
        candidates = in_region
    if len(candidates) < MIN_SIZE:
        # Fix round (QA finding 2): the count names the post-region-widening
        # total, not the field-relaxed in-region count.
        notes.append(
            "widened to nearby regions because too few matched in your "
            f"target regions; {len(pool)} candidates after widening"
        )
        candidates = pool
    return sorted(candidates, key=lambda u: _rank_key(state, u)), notes


class Pick(BaseModel):
    """One annotation for the corresponding code-selected row."""

    model_config = ConfigDict(extra="forbid")

    why_it_fits: str = Field(
        description="One line: why this program fits the candidate's interview "
        "answers, with a plain caveat where fit is weak."
    )


class Picks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    picks: list[Pick] = Field(
        description="One annotation per selected university, in the given order."
    )


def _render(entries: list[ShortlistEntry], notes: list[str]) -> str:
    header = "Here is your shortlist:"
    if notes:
        header += " (" + "; ".join(notes) + ")"
    body = "\n".join(
        f"- {e.university}, {e.country} — {e.rank_context}; "
        f"{degree_label(e.degree_type)}. "
        f"{humanize_degree_text(e.why_it_fits)}"
        for e in entries
    )
    return f"{header}\n{body}"


def shortlist_node(state: InterviewState) -> dict:
    """Filter and rank in code, then ask the LLM to annotate selected rows."""
    candidates, notes = prefilter(state, load_dataset().universities)
    selected = candidates[:MAX_SIZE]

    lines = "\n".join(
        f"- {u.university} — {u.country} ({u.continent}); {_rank_context(u)};"
        f" fields: {', '.join(u.fields)}"
        for u in selected
    )
    profile = (
        f"Candidate profile — degree: {degree_intent_label(state.degree_intent)}; "
        f"field: {state.field}; focus: {state.focus_area or 'not given'}; "
        f"regions: {', '.join(state.target_regions)}; "
        f"constraints: {', '.join(state.constraints) or 'none stated'}."
    )
    if notes:
        profile += " Note: " + "; ".join(notes) + "."
    chain = structured(Picks, tier="heavy")
    msgs = [
        ("system", f"{PERSONA}\n\n{SHORTLIST_INSTRUCTIONS}"),
        ("human", f"{profile}\n\nCandidates:\n{lines}"),
    ]
    degree = _DEGREE_MAP[state.degree_intent]
    annotations = chain.invoke(msgs).picks[: len(selected)]
    entries = [
        ShortlistEntry(
            university=u.university,
            country=u.country,
            rank_context=_rank_context(u),
            degree_type=degree,
            why_it_fits=humanize_degree_text(
                annotations[i].why_it_fits
                if i < len(annotations)
                else MISSING_ANNOTATION
            ),
        )
        for i, u in enumerate(selected)
    ]
    rendered = _render(entries, notes)
    return {
        "handed_off": True,
        "shortlist": entries,
        "messages": state.messages + [("ai", rendered)],
    }
