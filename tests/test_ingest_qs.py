"""scripts/ingest_qs.py: name matching, QS composite arithmetic, write guard.

Uses a hand-built four-row CSV rather than the real 1501-row list, so the
expected composites are checkable by hand in the assertions below.
"""

import json

import pytest

from scripts import ingest_qs

COLUMNS = [
    "2026 Rank", "Institution Name", "Country/Territory",
    "AR SCORE", "CPF SCORE", "ER SCORE", "FSR SCORE",
    "IFR SCORE", "ISR SCORE", "IRN SCORE", "EO SCORE", "SUS SCORE",
]

ROWS = [
    # Plain name. Two blank internationalisation indicators (5% each), which
    # is the shape QS actually publishes for deep-band institutions.
    ["1", "Testfield University", "Testland", "80", "90", "40", "60", "", "", "50", "70", "30"],
    # Accented + parenthetical: the dataset stores this name ASCII-folded.
    ["701-710", "Universidade de São Paulo (USP)", "Brazil", "20", "10", "30", "40", "50", "60", "10", "20", "30"],
    # Reached only through ALIASES — normalisation cannot bridge an acronym.
    ["=17", "École Polytechnique Fédérale de Lausanne", "Switzerland", "70", "80", "50", "90", "60", "70", "80", "40", "20"],
    # Present in the CSV but absent from the dataset: must be ignored, not
    # appended. Guards against the transform growing the dataset by accident.
    ["2", "Unrelated Institute", "Testland", "99", "99", "99", "99", "99", "99", "99", "99", "99"],
]


def _csv(tmp_path, *extra):
    lines = [",".join(COLUMNS)] + [",".join(r) for r in [*ROWS, *extra]]
    path = tmp_path / "qs.csv"
    path.write_text("﻿" + "\n".join(lines), encoding="utf-8")  # BOM, as QS ships
    return path


def _dataset(tmp_path, names):
    entries = [
        {
            "university": n, "country": "Testland", "continent": "Europe",
            "rank": None, "rank_scope": "world", "fields": ["engineering"],
            "degree_types": ["phd"], "verified": True, "confidence": "low",
        }
        for n in names
    ]
    path = tmp_path / "programs.json"
    path.write_text(json.dumps({"source": "t", "retrieved": None, "universities": entries}))
    return path


def _run(tmp_path, monkeypatch, names):
    target = _dataset(tmp_path, names)
    monkeypatch.setattr(ingest_qs, "DATASET", target)
    ingest_qs.main(_csv(tmp_path))
    return {u["university"]: u for u in json.loads(target.read_text())["universities"]}


def test_matches_across_name_variants_and_scores_them(tmp_path, monkeypatch, capsys):
    rows = _run(
        tmp_path, monkeypatch,
        ["Testfield University", "Universidade de Sao Paulo (USP)", "EPFL"],
    )

    # Blank indicators contribute zero, matching QS's own arithmetic:
    # 80*.30 + 90*.20 + 40*.15 + 60*.10 + 0 + 0 + 50*.05 + 70*.05 + 30*.05
    assert rows["Testfield University"]["base_score"] == 61.5
    assert rows["Testfield University"]["tilt"] == -50.0  # ER 40 - CPF 90
    assert rows["Testfield University"]["rank"] == 1

    # Accent fold + parenthetical drop; a QS band stays a string per the schema.
    assert rows["Universidade de Sao Paulo (USP)"]["rank"] == "701-710"
    assert rows["Universidade de Sao Paulo (USP)"]["base_score"] == 25.0

    # Alias path, and "=" (joint rank) is stripped to a plain int.
    assert rows["EPFL"]["rank"] == 17
    assert rows["EPFL"]["tilt"] == -30.0

    assert len(rows) == 3  # the CSV's extra institution was not appended
    assert "rank changes: 3" in capsys.readouterr().out


def test_unmatched_row_aborts_without_writing(tmp_path, monkeypatch):
    # The forced opposite of the test above: same CSV, one dataset name the
    # CSV has no row for. A silent miss would leave a stale rank sitting next
    # to freshly written ones, so the run must abort BEFORE writing.
    target = _dataset(tmp_path, ["Testfield University", "Nowhere University"])
    before = target.read_text()
    monkeypatch.setattr(ingest_qs, "DATASET", target)

    with pytest.raises(SystemExit, match="Nowhere University"):
        ingest_qs.main(_csv(tmp_path))

    assert target.read_text() == before


def test_known_absent_name_is_tolerated(tmp_path, monkeypatch):
    # KAIST is ranked by QS but missing from the mirror. It must not abort the
    # run, and its rank must be left alone rather than nulled or guessed.
    monkeypatch.setattr(ingest_qs, "KNOWN_ABSENT", {"Nowhere University"})
    rows = _run(tmp_path, monkeypatch, ["Testfield University", "Nowhere University"])

    assert rows["Nowhere University"]["rank"] is None
    assert "base_score" not in rows["Nowhere University"]
    assert rows["Testfield University"]["base_score"] == 61.5


def test_ambiguous_name_is_not_bound_to_an_arbitrary_row(tmp_path, monkeypatch):
    # Two CSV rows folding to the same key (QS really does list three
    # "Universidad de los Andes"). Binding either would silently attach
    # another university's scores, so the key is dropped and the run aborts.
    twin = ["3", "The Testfield University", "Elsewhere", *["50"] * 9]
    target = _dataset(tmp_path, ["Testfield University"])
    monkeypatch.setattr(ingest_qs, "DATASET", target)

    with pytest.raises(SystemExit, match="Testfield University"):
        ingest_qs.main(_csv(tmp_path, twin))
