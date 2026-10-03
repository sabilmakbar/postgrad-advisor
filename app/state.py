"""Typed interview state shared across the graph (story 003, AC 4)."""

from typing import Literal

from pydantic import BaseModel, Field

DegreeIntent = Literal["masters_coursework", "masters_research", "phd"]
FitLabel = Literal["strong", "borderline", "reach", "unknown"]  # AD-4: labels only


class WorkExperience(BaseModel):
    employer: str
    title: str
    tenure: str
    summary: str
    months: int | None = None


class CVFacts(BaseModel):
    """Provisional facts extracted from the CV until ``cv_confirmed``. Doubles
    as the light-tier structured-output schema. Unset fields were not stated."""

    degree: str | None = Field(
        None,
        description="Highest completed or in-progress degree with field and "
        "institution, e.g. 'BSc Computer Science, University of X'.",
    )
    gpa: str | None = Field(
        None,
        description="GPA WITH its scale, exactly as the CV states it, "
        "e.g. '3.7 / 4.0'. Unset if the CV gives none.",
    )
    publications: list[str] = Field(
        [], description="One entry per publication: title and venue/year."
    )
    english_test: str | None = Field(
        None, description="English test and score, e.g. 'IELTS 7.5'. Unset if none."
    )
    experience: list[WorkExperience] = []
    experience_months: int | None = None


class ProgramFit(BaseModel):
    """Grounded fit verdict for one shortlisted program (story 005). Sources
    are code-enforced to come from the fetched search results, never from
    free LLM text (AC 3); `unknown` carries the searched query (AC 4)."""

    university: str
    label: FitLabel
    reasons: list[str]
    sources: list[str]  # >=1 URL for non-unknown verdicts; empty for unknown


class ShortlistEntry(BaseModel):
    """One shortlisted program (story 004). Identity fields come from the
    dataset row the LLM's pick index maps back to — never from free LLM text."""

    university: str
    country: str
    rank_context: str  # "QS world rank 2"; for null-rank rows, the dataset note
    degree_type: str  # dataset degree type matching the interview intent
    why_it_fits: str  # one-line rationale grounded in the interview answers


class InterviewState(BaseModel):
    """Filled progressively during the interview; complete once degree intent,
    field, and at least one target region are captured."""

    degree_intent: DegreeIntent | None = None
    degree_intent_blocked: bool = False  # story 012: reject one unsupported parse
    field: str | None = None
    focus_area: str | None = None
    target_regions: list[str] = []
    constraints: list[str] = []
    messages: list[tuple[str, str]] = []  # (role, content) conversation log
    turns: int = 0  # agent question turns taken; capped in graph logic
    ended: bool = False  # turn cap hit before the state completed
    handed_off: bool = False  # set by the shortlister node (story 004)
    shortlist: list[ShortlistEntry] = []  # built by the shortlister node
    pending_cv: str | None = None  # story 009: uploaded path held until cv_intake
    pending_correction: str | None = None  # story 009: text sent with that upload
    cv_facts: CVFacts | None = None  # extracted/provisional until cv_confirmed
    cv_parsed_path: str | None = None  # upload whose facts _interviewer extracted
    cv_confirmed: bool = False  # story 005: set once the user confirms the echo
    searches_used: int = 0  # story 005: runtime-search budget counter (AD-3, <=12)
    fit_verdicts: list[ProgramFit] = []  # story 005: built by the fit node

    def is_complete(self) -> bool:
        return bool(self.degree_intent and self.field and self.target_regions)


class SlotUpdate(BaseModel):
    """Structured-output schema for parsing ONE user answer (light tier).
    Fields left unset mean the answer didn't cover them."""

    degree_intent: DegreeIntent | None = Field(
        None, description="Degree the user wants, if the answer states it."
    )
    field: str | None = Field(
        None, description="Field of study, e.g. 'computer science'."
    )
    focus_area: str | None = Field(
        None, description="Narrower specialisation within the field, if given."
    )
    target_regions: list[str] = Field(
        [], description="Countries or regions the user wants to study in."
    )
    constraints: list[str] = Field(
        [], description="Constraints the user states: budget, timing, visa, etc."
    )
    postdoc_interest: bool = Field(
        False, description="True only if the user asks about postdoc positions."
    )
