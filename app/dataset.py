"""Static QS-2026 university snapshot: schema + loader (AD-3: rankings only)."""

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "programs.json"

Continent = Literal[
    "Africa", "Asia", "Europe", "North America", "Oceania", "South America"
]
DegreeType = Literal["masters_course", "masters_research", "phd"]

_RANK_BAND = re.compile(r"^\d+(-\d+|\+)$")  # e.g. "801-850" or "1401+"
CONTINENTS = (
    "Africa", "Asia", "Europe", "North America", "Oceania", "South America"
)
REGION_ALIASES = {
    "us": "United States",
    "usa": "United States",
    "uk": "United Kingdom",
}


class University(BaseModel):
    university: str
    country: str
    continent: Continent
    # point rank, a QS band string like "801-850", or None = exists but unranked
    # in this QS cycle (see `note` for why)
    rank: int | str | None
    rank_scope: Literal["world", "regional"]
    fields: list[str] = Field(min_length=1)
    degree_types: list[DegreeType] = Field(min_length=1)
    # Derived by scripts/ingest_qs.py from the QS indicator scores, never hand
    # edited. base_score is the QS weighted composite (0-100, comparable across
    # the whole list including banded ranks); tilt is employer reputation minus
    # citations per faculty. None = the institution is absent from the source CSV.
    base_score: float | None = None
    tilt: float | None = None
    verified: bool
    confidence: Literal["high", "medium", "low"]
    note: str | None = None

    @field_validator("rank")
    @classmethod
    def rank_format(cls, v: int | str | None) -> int | str | None:
        if isinstance(v, str) and not _RANK_BAND.match(v):
            raise ValueError(f"rank band must look like '801-850' or '1401+', got {v!r}")
        return v


class ProgramDataset(BaseModel):
    source: str
    retrieved: str | None  # set by the PM's web verification pass, not the executor
    universities: list[University]

    @model_validator(mode="after")
    def unique_names(self) -> "ProgramDataset":
        names = [u.university for u in self.universities]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate university names: {sorted(dupes)}")
        return self


def load_dataset(path: Path = DATASET_PATH) -> ProgramDataset:
    return ProgramDataset.model_validate_json(path.read_text())


def _normalise_region(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _region_lookup(countries=None) -> dict[str, str]:
    countries = countries or {u.country for u in load_dataset().universities}
    values = [*countries, *CONTINENTS]
    return {_normalise_region(value): value for value in values} | REGION_ALIASES


def canonical_region(value: str, countries=None) -> str | None:
    """Resolve one exact country, continent, or declared alias."""
    return _region_lookup(countries).get(_normalise_region(value))


def regions_in_text(text: str, countries=None) -> list[str]:
    """Return canonical regions supported by whole normalized tokens in text."""
    words = f" {_normalise_region(text)} "
    found = []
    for token, canonical in _region_lookup(countries).items():
        if f" {token} " in words and canonical not in found:
            found.append(canonical)
    return found
