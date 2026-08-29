"""Typed interview state shared across the graph (story 003, AC 4)."""

from typing import Literal

from pydantic import BaseModel, Field

DegreeIntent = Literal["masters_coursework", "masters_research", "phd"]


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
    field: str | None = None
    focus_area: str | None = None
    target_regions: list[str] = []
    constraints: list[str] = []
    messages: list[tuple[str, str]] = []  # (role, content) conversation log
    turns: int = 0  # agent question turns taken; capped in graph logic
    ended: bool = False  # turn cap hit before the state completed
    handed_off: bool = False  # set by the shortlister node (story 004)
    shortlist: list[ShortlistEntry] = []  # built by the shortlister node

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
