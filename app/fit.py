"""Per-program fit judgment (story 005): heavy-tier structured verdicts whose
sources are code-grounded in the fetched results (AC 3); `unknown` with the
searched query when nothing is found — never guessed (AC 4); numeric
probabilities scrubbed in code, not just discouraged in the prompt (AD-4).
"""

import re

from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field

from app.model_layer import structured
from app.prompts import FIT_INSTRUCTIONS, PERSONA
from app.requirements_search import SearchResult, search_requirements
from app.shortlist import degree_label, humanize_degree_text
from app.state import FitLabel, InterviewState, ProgramFit

MAX_SEARCHES = 12  # AD-3: per-conversation runtime-search cap, enforced here
FIT_BATCH_SIZE = 5

# AD-4 / SC-005: numeric admission odds in any spelling — "70%", "70 percent",
# "chance(s) of 7 in 10".
_PROB_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|percent)|\bchances?\s+of\s+\d", re.I)


class FitVerdict(BaseModel):
    """LLM structured output for ONE program (AC 3)."""

    label: FitLabel = Field(
        description="strong, borderline, or reach when evidence is sufficient; "
        "unknown when evidence is incomplete."
    )
    reasons: list[str] = Field(
        description="Each reason ties a specific CV fact to a specific fetched "
        "requirement. Never a numeric probability."
    )
    sources: list[str] = Field(
        description="URLs copied EXACTLY from the fetched source list that "
        "support the reasons."
    )
    sufficient: bool = Field(
        description="Whether the evidence is sufficient for a grounded fit verdict."
    )
    missing_evidence: str | None = Field(
        None, description="Short description of what evidence is missing."
    )
    reformulated_query: str | None = Field(
        None,
        description="One corrective query naming the same university and degree intent.",
    )


def _unknown(university: str, query: str, why: str) -> ProgramFit:
    return ProgramFit(
        university=university,
        label="unknown",
        reasons=[f'{why} (searched: "{query}")'],
        sources=[],
    )


def _unknown_attempts(university: str, queries: list[str], why: str) -> ProgramFit:
    attempted = "; ".join(f'"{query}"' for query in queries)
    return ProgramFit(
        university=university,
        label="unknown",
        reasons=[f"{why} (searched: {attempted})"],
        sources=[],
    )


def _normalized_words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _same_program_query(query: str, university: str, degree: str) -> bool:
    normalized = _normalized_words(query)
    identity = f"{_normalized_words(university)} {_normalized_words(degree)}"
    if normalized != identity and not normalized.startswith(f"{identity} "):
        return False

    degree_words = _normalized_words(degree)
    conflicting = {
        "phd": ("master", "mres", "mba"),
        "taught master s": ("phd", "doctoral", "doctorate", "research master", "mres"),
        "research master s": ("phd", "doctoral", "doctorate", "taught master", "mba"),
    }[degree_words]
    return not any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in conflicting)


def _combine(first: SearchResult, second: SearchResult) -> SearchResult:
    sources = list(dict.fromkeys([*first.sources, *second.sources]))
    sections = [
        f'Query: "{result.query}"\n{result.text or "[no results]"}'
        for result in (first, second)
    ]
    return SearchResult(query=second.query, text="\n\n".join(sections), sources=sources)


def _judge(chain, cv: dict, entry, found: SearchResult, pass_name: str) -> FitVerdict:
    return chain.invoke(
        [
            ("system", f"{PERSONA}\n\n{FIT_INSTRUCTIONS}"),
            (
                "human",
                f"Evidence pass: {pass_name}.\n"
                f"CV facts (user-confirmed): {cv}\n\n"
                f"Program: {entry.university}, {entry.country} — "
                f"{entry.rank_context}; {degree_label(entry.degree_type)}.\n\n"
                f"Fetched requirements:\n{found.text or '[no results]'}\n\n"
                f"Source URLs: {found.sources}",
            ),
        ]
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


def _render(verdicts: list[ProgramFit], pending: list[str]) -> str:
    lines = []
    for v in verdicts:
        reasons = "; ".join(humanize_degree_text(reason) for reason in v.reasons)
        line = f"- {v.university}: {v.label} — {reasons}"
        if v.sources:
            line += f" (sources: {', '.join(v.sources)})"
        lines.append(line)
    rendered = (
        "Here is how your confirmed CV facts stack up against each shortlisted "
        "program:\n" + "\n".join(lines)
    )
    if pending:
        rendered += "\n\nPending assessment: " + ", ".join(pending)
    return rendered


def fit_batch(state: InterviewState) -> dict:
    """Assess at most five programs from the cursor within the shared budget."""
    chain = structured(FitVerdict, tier="heavy")
    facts = state.cv_facts
    cv = {
        "degree": facts.degree,
        "gpa": facts.gpa,
        "publications": facts.publications,
        "english_test": facts.english_test,
        "experience": [
            {"title": position.title, "tenure": position.tenure}
            for position in facts.experience
        ],
        "experience_months": facts.experience_months,
    }
    cv = {k: v for k, v in cv.items() if v is not None and v != []}
    used = min(max(state.searches_used, 0), MAX_SEARCHES)
    verdicts = list(getattr(state, "fit_verdicts", []))
    cursor = min(max(getattr(state, "fit_cursor", 0), 0), len(state.shortlist))
    total = len(state.shortlist)
    stop = min(cursor + FIT_BATCH_SIZE, total)
    for position in range(cursor, stop):
        entry = state.shortlist[position]
        try:
            get_stream_writer()(
                f"searching requirements: {entry.university} ({position + 1}/{total})"
            )
        except Exception:
            pass
        query = f"{entry.university} {state.field} admission requirements"
        if used >= MAX_SEARCHES:
            verdicts.append(
                _unknown(
                    entry.university,
                    query,
                    "this conversation's search budget is used up, so I won't guess",
                )
            )
            cursor = position + 1
            break
        used += 1
        found = search_requirements(query)
        judged = _judge(chain, cv, entry, found, "initial")
        if judged.sufficient:
            verdicts.append(_ground(entry.university, judged, found))
            cursor = position + 1
            continue

        missing = judged.missing_evidence or "admission evidence remains incomplete"
        retry_query = judged.reformulated_query
        if used >= MAX_SEARCHES:
            verdicts.append(_unknown_attempts(entry.university, [query], missing))
            cursor = position + 1
            continue
        degree = degree_label(entry.degree_type)
        if not retry_query or not _same_program_query(
            retry_query, entry.university, degree
        ):
            verdicts.append(
                _unknown_attempts(
                    entry.university,
                    [query],
                    f"{missing}; the proposed corrective query did not preserve "
                    f"{entry.university} and {degree}",
                )
            )
            cursor = position + 1
            continue

        used += 1
        recovered = search_requirements(retry_query)
        combined = _combine(found, recovered)
        final = _judge(chain, cv, entry, combined, "corrective and final")
        if final.sufficient:
            verdicts.append(_ground(entry.university, final, combined))
        else:
            verdicts.append(
                _unknown_attempts(
                    entry.university,
                    [query, retry_query],
                    final.missing_evidence or "admission evidence remains incomplete",
                )
            )
        cursor = position + 1
    pending = [entry.university for entry in state.shortlist[cursor:]]
    return {
        "fit_verdicts": verdicts,
        "searches_used": used,
        "fit_cursor": cursor,
        "fit_continue": None,
        "messages": state.messages + [("ai", _render(verdicts, pending))],
    }


# Kept for callers of the story-005 unit boundary; the graph uses ``fit_batch``.
fit_node = fit_batch
