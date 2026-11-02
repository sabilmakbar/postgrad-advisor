"""scripts/ingest_qs.py: name matching, QS composite arithmetic, write guard.

Uses a hand-built four-row CSV rather than the real 1501-row list, so the
expected composites are checkable by hand in the assertions below.
"""

import json

import pytest

from scripts import ingest_qs
from scripts import refresh_university_pool as refresh
from app.dataset import load_dataset

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


def _full_csv(tmp_path):
    import csv
    path = tmp_path / "full-qs.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*COLUMNS, "Region"])
        writer.writeheader()
        for values in ROWS:
            row = dict(zip(COLUMNS, values))
            row["Region"] = "Americas" if row["Country/Territory"] == "Brazil" else "Europe"
            writer.writerow(row)
    return path


def test_refresh_previews_then_expands_pool_preserving_known_metadata(tmp_path, capsys):
    target = _dataset(tmp_path, ["Testfield University"])
    before = target.read_bytes()
    source = _full_csv(tmp_path)
    refresh.main(["--world", str(source), "--dataset", str(target)])
    assert target.read_bytes() == before
    assert json.loads(capsys.readouterr().out)["added_count"] == 3
    refresh.main(["--world", str(source), "--dataset", str(target), "--apply"])
    result = load_dataset(target)
    assert len(result.universities) == 4 and result.ranking_year == 2026
    assert result.universities[0].fields == ["engineering"]
    assert result.universities[1].fields == [] and result.universities[1].degree_types == []
    assert result.universities[1].continent == "South America"
    assert "sha256" in result.source


@pytest.mark.parametrize(("old", "new"), [("2026 Rank", "Bad Header"), ("1,Testfield", "NOT-RANKED,Testfield"), ("80,90,40", "NaN,90,40")])
def test_refresh_rejects_malformed_input_without_replacing_cache(tmp_path, old, new):
    target = _dataset(tmp_path, ["Testfield University"])
    source = _full_csv(tmp_path)
    source.write_text(source.read_text().replace(old, new, 1))
    before = target.read_bytes()
    with pytest.raises(SystemExit):
        refresh.main(["--world", str(source), "--dataset", str(target), "--apply"])
    assert target.read_bytes() == before


def test_refresh_imports_subject_evidence_and_subject_only_institution(tmp_path):
    import csv
    target = _dataset(tmp_path, ["Testfield University"])
    subjects = tmp_path / "subject.csv"
    with subjects.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Rank", "Institution Name", "Country/Territory"])
        writer.writerow(["=2", "Testfield University", "Testland"])
        writer.writerow(["51-100", "Subject Only University", "Testland"])
    refresh.main(["--world", str(_full_csv(tmp_path)), "--dataset", str(target),
                  "--subject", f"computer_science:2025={subjects}", "--apply"])
    result = {u.university: u for u in load_dataset(target).universities}
    assert result["Testfield University"].fields == ["engineering", "computer_science"]
    assert result["Subject Only University"].rank is None
    evidence = result["Subject Only University"].subject_ranks["computer_science"]
    assert evidence.rank == "51-100" and evidence.year == 2025
    assert evidence.source == "subject.csv" and len(evidence.sha256) == 64


def test_refresh_report_cannot_overwrite_live_dataset(tmp_path):
    target = _dataset(tmp_path, ["Testfield University"])
    before = target.read_bytes()
    with pytest.raises(SystemExit):
        refresh.main(["--world", str(_full_csv(tmp_path)), "--dataset", str(target), "--report", str(target)])
    assert target.read_bytes() == before


def test_refresh_refuses_a_partial_source(tmp_path):
    target = _dataset(tmp_path, [f"Existing {i}" for i in range(20)])
    before = target.read_bytes()
    with pytest.raises(SystemExit):
        refresh.main(["--world", str(_full_csv(tmp_path)), "--dataset", str(target), "--apply"])
    assert target.read_bytes() == before


def test_https_source_is_downloaded_without_model_calls(tmp_path, monkeypatch):
    import io
    source = _full_csv(tmp_path)
    class Response(io.BytesIO):
        url = "https://example.test/qs.csv"
    requests = []
    def fetch(request, timeout):
        requests.append((request.full_url, timeout))
        return Response(source.read_bytes())
    monkeypatch.setattr(refresh, "urlopen", fetch)
    rows, digest = refresh.read_source("https://example.test/qs.csv")
    assert len(rows) == 4 and len(digest) == 64
    assert requests == [("https://example.test/qs.csv", 30)]


def test_blank_subject_identity_cannot_replace_dataset(tmp_path):
    target = _dataset(tmp_path, ["Testfield University"])
    source = _full_csv(tmp_path)
    subject = tmp_path / "bad-subject.csv"
    subject.write_text("Rank,Institution Name,Country/Territory\n1,,Testland\n")
    before = target.read_bytes()
    with pytest.raises(SystemExit):
        refresh.main(["--world", str(source), "--dataset", str(target), "--subject",
                      f"computer_science:2026={subject}", "--apply"])
    assert target.read_bytes() == before


def test_subject_only_refresh_preserves_notes_and_clears_old_verification(tmp_path):
    target = _dataset(tmp_path, ["Subject Only University"])
    data = json.loads(target.read_text())
    data["universities"][0].update(rank=15, confidence="high", note="Reviewed partner-award arrangement")
    target.write_text(json.dumps(data))
    subject = tmp_path / "subject.csv"
    subject.write_text("Rank,Institution Name,Country/Territory\n3,Subject Only University,Testland\n")
    refresh.main(["--world", str(_full_csv(tmp_path)), "--dataset", str(target),
                  "--subject", f"computer_science:2026={subject}", "--apply"])
    entry = next(u for u in load_dataset(target).universities if u.university == "Subject Only University")
    assert "Reviewed partner-award arrangement" in entry.note
    assert entry.rank is None and not entry.verified and entry.confidence == "low"


def _subject_book(tmp_path, tables):
    from openpyxl import Workbook
    book = Workbook()
    book.remove(book.active)
    for label, rows in [("Engineering & Technology", []), *tables]:
        sheet = book.create_sheet(label[:31])
        sheet.append(["2026 QS World University Rankings by Subject"])
        sheet.append([label])
        sheet.append([None])
        sheet.append(["2026", "2025", "INSTITUTION", "COUNTRY/TERRITORY"])
        for rank, name, country in rows:
            sheet.append([rank, None, name, country])
    path = tmp_path / "subjects.xlsx"
    book.save(path)
    book.close()
    return path


def test_official_workbook_preserves_individual_subject_ranks(tmp_path):
    target = _dataset(tmp_path, ["Testfield University"])
    book = _subject_book(tmp_path, [
        ("Computer Science & Information Systems", [(2, "Testfield University", "Testland")]),
        ("Data Science and Artificial Intelligence", [(3, "Testfield University", "Testland")]),
    ])
    refresh.main(["--world", str(_full_csv(tmp_path)), "--subject-workbook", str(book),
                  "--dataset", str(target), "--apply"])
    university = load_dataset(target).universities[0]
    assert university.fields == ["engineering", "computer_science"]
    assert university.subject_ranks["Computer Science & Information Systems"].rank == 2
    assert university.subject_ranks["Data Science and Artificial Intelligence"].rank == 3
    assert all(r.year == 2026 and r.source == "subjects.xlsx" for r in university.subject_ranks.values())


def test_workbook_does_not_merge_distinct_parenthetical_campuses(tmp_path):
    target = _dataset(tmp_path, ["Testfield University"])
    book = _subject_book(tmp_path, [("Computer Science & Information Systems", [
        (2, "Campus University", "Testland"), (3, "Campus University (Beijing)", "Testland"),
    ])])
    refresh.main(["--world", str(_full_csv(tmp_path)), "--subject-workbook", str(book),
                  "--dataset", str(target), "--apply"])
    names = [u.university for u in load_dataset(target).universities]
    assert "Campus University" in names and "Campus University (Beijing)" in names


def test_source_names_make_repeated_refresh_idempotent(tmp_path):
    target = _dataset(tmp_path, ["Testfield University"])
    source = _full_csv(tmp_path)
    args = ["--world", str(source), "--dataset", str(target), "--apply"]
    refresh.main(args)
    before = target.read_bytes()
    refresh.main(args)
    assert target.read_bytes() == before


def test_bad_workbook_keeps_dataset_intact(tmp_path):
    target = _dataset(tmp_path, ["Testfield University"])
    source = _full_csv(tmp_path)
    book = tmp_path / "bad.xlsx"
    book.write_bytes(b"not a spreadsheet")
    before = target.read_bytes()
    with pytest.raises(SystemExit):
        refresh.main(["--world", str(source), "--subject-workbook", str(book),
                      "--dataset", str(target), "--apply"])
    assert target.read_bytes() == before


def test_repeat_refresh_does_not_count_subject_only_rows_as_missing_world_rows(tmp_path):
    target = _dataset(tmp_path, ["Testfield University"])
    source = _full_csv(tmp_path)
    book = _subject_book(tmp_path, [("Computer Science & Information Systems", [
        (i + 1, f"Subject Only-{i}", "Testland") for i in range(8)
    ])])
    args = ["--world", str(source), "--subject-workbook", str(book), "--dataset", str(target), "--apply"]
    refresh.main(args)
    assert len(load_dataset(target).universities) == 12
    before = target.read_bytes()
    refresh.main(args)
    assert target.read_bytes() == before


def test_repeated_refresh_preserves_notes_for_institutions_absent_from_all_sources(tmp_path):
    target = _dataset(tmp_path, ["Absent University"])
    data = json.loads(target.read_text())
    data["universities"][0]["note"] = "Reviewed historical exception"
    target.write_text(json.dumps(data))
    source = _full_csv(tmp_path)
    args = ["--world", str(source), "--dataset", str(target), "--apply"]
    refresh.main(args)
    before = target.read_bytes()
    refresh.main(args)
    assert target.read_bytes() == before
    entry = next(u for u in load_dataset(target).universities if u.university == "Absent University")
    assert "Reviewed historical exception" in entry.note
