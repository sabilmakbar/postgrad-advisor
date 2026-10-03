"""Per-program fit judgment (story 005): heavy-tier structured verdicts whose
sources are code-grounded in the fetched results (AC 3); `unknown` with the
searched query when nothing is found — never guessed (AC 4); numeric
probabilities scrubbed in code, not just discouraged in the prompt (AD-4).
"""

import re

from pydantic import BaseModel, Field

from app.model_layer import structured
from app.prompts import FIT_INSTRUCTIONS, PERSONA
from app.requirements_search import search_requirements
from app.shortlist import degree_label, humanize_degree_text
from app.state import FitLabel, InterviewState, ProgramFit

MAX_SEARCHES = 12  # AD-3: per-conversation runtime-search cap, enforced here

# AD-4 / SC-005: numeric admission odds in any spelling — "70%", "70 percent",
# "chance(s) of 7 in 10".
_PROB_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|percent)|\bchances?\s+of\s+\d", re.I)


class FitVerdict(BaseModel):
    """LLM structured output for ONE program (AC 3)."""

    label: FitLabel = Field(
        description="strong, borderline, or reach; unknown ONLY if the fetched "
        "text contains no actual admission requirements."
    )
    reasons: list[str] = Field(
        description="Each reason ties a specific CV fact to a specific fetched "
        "requirement. Never a numeric probability."
    )
    sources: list[str] = Field(
        description="URLs copied EXACTLY from the fetched source list that "
        "support the reasons."
    )


def _unknown(university: str, query: str, why: str) -> ProgramFit:
    return ProgramFit(
        university=university,
        label="unknown",
        reasons=[f'{why} (searched: "{query}")'],
        sources=[],
    )


def _scrub(reasons: list[str]) -> list[str]:
    """Drop any reason carrying a numeric probability (AD-4, code-enforced)."""
    kept = [r for r in reasons if not _PROB_RE.search(r)]
    return kept or [
        "the stated reasons were withheld because they gave numeric admission "
        "odds, which this service never does — see the cited sources"
    ]


def _ground(university: str, judged: FitVerdict, found) -> ProgramFit:
    """Sources must exist in the fetched results; invented URLs are replaced by
    the real ones, and an LLM `unknown` still shows the searched query."""
    if judged.label == "unknown":
        return _unknown(
            university, found.query, "the pages found gave no admission requirements"
        )
    sources = [s for s in judged.sources if s in found.sources] or found.sources[:3]
    return ProgramFit(
        university=university,
        label=judged.label,
        reasons=_scrub([humanize_degree_text(reason) for reason in judged.reasons]),
        sources=sources,
    )


def _render(verdicts: list[ProgramFit]) -> str:
    lines = []
    for v in verdicts:
        reasons = "; ".join(humanize_degree_text(reason) for reason in v.reasons)
        line = f"- {v.university}: {v.label} — {reasons}"
        if v.sources:
            line += f" (sources: {', '.join(v.sources)})"
        lines.append(line)
    return (
        "Here is how your confirmed CV facts stack up against each shortlisted "
        "program:\n" + "\n".join(lines)
    )


def fit_node(state: InterviewState) -> dict:
    """One search + one heavy-tier judgment per shortlisted program, within the
    state's search budget; final summary rendered in code (AC 2-5)."""
    chain = structured(FitVerdict, tier="heavy")
    cv = {k: v for k, v in state.cv_facts.model_dump().items() if v}
    used = state.searches_used
    verdicts: list[ProgramFit] = []
    for entry in state.shortlist:
        query = f"{entry.university} {state.field} admission requirements"
        if used >= MAX_SEARCHES:
            verdicts.append(
                _unknown(
                    entry.university,
                    query,
                    "this conversation's search budget is used up, so I won't guess",
                )
            )
            continue
        used += 1
        found = search_requirements(query)
        if not found.text.strip():
            verdicts.append(
                _unknown(entry.university, query, "no admission requirements found")
            )
            continue
        judged = chain.invoke(
            [
                ("system", f"{PERSONA}\n\n{FIT_INSTRUCTIONS}"),
                (
                    "human",
                    f"CV facts (user-confirmed): {cv}\n\n"
                    f"Program: {entry.university}, {entry.country} — "
                    f"{entry.rank_context}; {degree_label(entry.degree_type)}.\n\n"
                    f"Fetched requirements:\n{found.text}\n\n"
                    f"Source URLs: {found.sources}",
                ),
            ]
        )
        verdicts.append(_ground(entry.university, judged, found))
    return {
        "fit_verdicts": verdicts,
        "searches_used": used,
        "messages": state.messages + [("ai", _render(verdicts))],
    }
