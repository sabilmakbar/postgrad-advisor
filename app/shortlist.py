"""Shortlister node (story 004): code filters the dataset, the LLM only ranks
and annotates the filtered candidates by index. Output maps indices back to
dataset rows, so a made-up university is impossible by construction (AD-3).
"""

import re

from pydantic import BaseModel, Field

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


def _rank_key(u: University) -> float:
    # Null rank = unranked this cycle: never sorted as if top-ranked (dev note).
    if u.rank is None:
        return float("inf")
    if isinstance(u.rank, str):  # QS band like "801-850" or "1401+"
        return float(u.rank.split("-")[0].rstrip("+"))
    return float(u.rank)


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
    return sorted(candidates, key=_rank_key), notes


class Pick(BaseModel):
    """One ranked pick; index refers to the numbered candidate list."""

    index: int = Field(description="Index of the pick in the candidate list.")
    university: str = Field(
        description="University name, copied EXACTLY from that candidate line."
    )
    why_it_fits: str = Field(
        description="One line: why this program fits the candidate's interview "
        "answers, with a plain caveat where fit is weak."
    )


class Picks(BaseModel):
    picks: list[Pick] = Field(
        description=f"{MIN_SIZE}-{MAX_SIZE} picks, best fit first."
    )


def _ground(
    picks: list[Pick],
    candidates: list[University],
    degree: str,
    seen: set[int],
    entries: list[ShortlistEntry],
) -> None:
    """The structural guard: an index outside the filtered list, a repeat,
    or a name that doesn't match the dataset row is dropped, not trusted."""
    for p in picks:
        if p.index in seen or not 0 <= p.index < len(candidates):
            continue
        u = candidates[p.index]
        if p.university != u.university:
            continue
        seen.add(p.index)
        entries.append(
            ShortlistEntry(
                university=u.university,
                country=u.country,
                rank_context=_rank_context(u),
                degree_type=degree,
                why_it_fits=humanize_degree_text(p.why_it_fits),
            )
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
    """Runs once the interview completes: filter, LLM-rank by index, map
    indices back to dataset rows, render with any widening named (AC 1-3)."""
    candidates, notes = prefilter(state, load_dataset().universities)

    lines = "\n".join(
        f"{i}. {u.university} — {u.country} ({u.continent}); {_rank_context(u)};"
        f" fields: {', '.join(u.fields)}"
        for i, u in enumerate(candidates)
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
    entries: list[ShortlistEntry] = []
    seen: set[int] = set()
    _ground(chain.invoke(msgs).picks, candidates, degree, seen, entries)

    grounding_notice = None
    if len(entries) < MIN_SIZE <= len(candidates):
        # Floor guard (fix round): enough valid candidates exist but the model
        # under-returned or the guard dropped picks — retry exactly once, then
        # ship short WITH a notice, never silently. A thin widened pool
        # (<MIN_SIZE rows) is already covered by the AC-3 widening notes.
        need = MIN_SIZE - len(entries)
        retry = msgs + [
            (
                "human",
                f"Only {len(entries)} of your picks could be grounded against "
                f"the candidate list. Give at least {need} more grounded picks "
                f"from the same numbered candidates, skipping indices "
                f"{sorted(seen)}.",
            )
        ]
        _ground(chain.invoke(retry).picks, candidates, degree, seen, entries)
        if len(entries) < MIN_SIZE:
            grounding_notice = (
                f"only {len(entries)} picks could be grounded against the dataset"
            )

    entries = entries[:MAX_SIZE]  # upper bound enforced in code, not the prompt
    rendered = _render(entries, notes)
    if grounding_notice:
        rendered += f"\n{grounding_notice}"
    return {
        "handed_off": True,
        "shortlist": entries,
        "messages": state.messages + [("ai", rendered)],
    }
