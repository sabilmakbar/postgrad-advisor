"""Story 002: schema, uniqueness, and coverage of data/programs.json."""

from collections import Counter

import pytest
from pydantic import ValidationError

from app.dataset import ProgramDataset, University, canonical_region, load_dataset, regions_in_text

CONTINENTS = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]

VALID_ENTRY = {
    "university": "Test University",
    "country": "Testland",
    "continent": "Europe",
    "rank": 1,
    "rank_scope": "world",
    "fields": ["engineering"],
    "degree_types": ["phd"],
    "verified": False,
    "confidence": "high",
}


def test_dataset_loads_and_validates():
    ds = load_dataset()
    assert ds.source
    assert "retrieved" in ds.model_fields_set  # field present (PM fills the date)
    assert len(ds.universities) >= 120


def test_per_continent_coverage():
    ds = load_dataset()
    counts = Counter(u.continent for u in ds.universities)
    assert set(counts) == set(CONTINENTS)
    for continent in CONTINENTS:
        assert counts[continent] >= 15, f"{continent}: {counts[continent]} < 15"


def test_university_names_unique():
    ds = load_dataset()
    names = [u.university for u in ds.universities]
    assert len(names) == len(set(names))


def test_null_rank_validates():
    u = University.model_validate({**VALID_ENTRY, "rank": None})
    assert u.rank is None


def test_note_is_optional():
    assert University.model_validate(VALID_ENTRY).note is None
    noted = University.model_validate({**VALID_ENTRY, "note": "excluded from QS 2026"})
    assert noted.note == "excluded from QS 2026"


# --- broken fixtures: prove the validator can fail (zero-results rule) ---


def test_rejects_bad_continent():
    with pytest.raises(ValidationError):
        University.model_validate({**VALID_ENTRY, "continent": "Atlantis"})


def test_rejects_empty_fields():
    with pytest.raises(ValidationError):
        University.model_validate({**VALID_ENTRY, "fields": []})


def test_rejects_malformed_rank_band():
    with pytest.raises(ValidationError):
        University.model_validate({**VALID_ENTRY, "rank": "about 800"})


def test_rejects_duplicate_names():
    with pytest.raises(ValidationError, match="duplicate"):
        ProgramDataset.model_validate(
            {"source": "x", "retrieved": None, "universities": [VALID_ENTRY, VALID_ENTRY]}
        )


def test_exact_region_resolution_and_raw_answer_tokens():
    countries = {"United States", "United Kingdom", "Australia", "Germany"}
    assert canonical_region(" usa ", countries) == "United States"
    assert canonical_region("UK", countries) == "United Kingdom"
    assert canonical_region("Europe", countries) == "Europe"
    assert canonical_region("US", {"Australia"}) == "United States"
    assert canonical_region("land", countries) is None
    assert regions_in_text("Germany, Europe, or wherever is best", countries) == [
        "Germany", "Europe"
    ]
    assert regions_in_text("near my current location", countries) == []
