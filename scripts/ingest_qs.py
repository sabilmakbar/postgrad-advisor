"""Refresh ranks and derive comparability scores in data/programs.json from a
QS WUR full-list CSV.

Offline transform, stdlib only. Run after each QS release:

    python scripts/ingest_qs.py data/qs_raw/qs-wur-2026.csv

The CSV must carry QS's nine per-indicator score columns. Two values are
derived per row and written back:

  base_score  the QS weighted composite recomputed from the nine indicators.
              QS publishes "Overall SCORE" only for the top ~700; the
              indicators are populated for the whole list, so recomputing
              gives one comparable 0-100 number for every institution
              including the banded tail. Validated against all 703 rows that
              carry a published overall: mean abs err 0.75, p95 1.24,
              max 1.30. The residual is rounding in the published indicators.
  tilt        ER minus CPF. Positive = employer recognition outruns citation
              intensity; negative = the reverse. corr(ER, CPF) is 0.397 over
              1501 rows, so this is an independent axis, not a prestige echo.

Nothing here decides whether an institution offers a given degree: QS carries
no degree-level data at all. base_score and tilt rank candidates; degree
availability is a runtime question.

Refuses to write if any dataset row fails to match, except the names in
KNOWN_ABSENT. A silent miss would leave a stale rank next to a fresh one.
"""

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "programs.json"

# QS WUR 2024+ methodology weights. Sum to 1.0; ISD is indicative and unweighted.
QS_WEIGHTS = {
    "AR SCORE": 0.30,   # academic reputation
    "CPF SCORE": 0.20,  # citations per faculty
    "ER SCORE": 0.15,   # employer reputation
    "FSR SCORE": 0.10,  # faculty/student ratio
    "IFR SCORE": 0.05,  # international faculty ratio
    "ISR SCORE": 0.05,  # international student ratio
    "IRN SCORE": 0.05,  # international research network
    "EO SCORE": 0.05,   # employment outcomes
    "SUS SCORE": 0.05,  # sustainability
}

# Dataset name -> exact QS name, for the rows normalisation cannot bridge:
# local abbreviations, native-language official names, and QS renames.
ALIASES = {
    "EPFL": "École Polytechnique Fédérale de Lausanne",
    "LMU Munich": "Ludwig-Maximilians-Universität München",
    "Heidelberg University": "Ruprecht-Karls-Universität Heidelberg",
    "Nanyang Technological University (NTU)": (
        "Nanyang Technological University, Singapore (NTU Singapore)"
    ),
    "IIT Delhi": "Indian Institute of Technology Delhi (IITD)",
    "IIT Bombay": "Indian Institute of Technology Bombay (IITB)",
    # QS 2026 lists the post-merger entity, dropping "The University of".
    "The University of Adelaide": "Adelaide University",
    "The University of Newcastle, Australia": "The University of Newcastle",
    "Universidad de los Andes": "Universidad de Los Andes Colombia",
    "Universidade Estadual Paulista (UNESP)": (
        'Universidade Estadual Paulista "Júlio de Mesquita Filho" (UNESP)'
    ),
    "Universidad de Santiago de Chile (USACH)": (
        "Universidad de Santiago de Chile - USACH"
    ),
    "Universidad Austral": "Universidad Austral - Argentina",
}

# Institutions QS ranks but this mirror omits. Keeps rank None rather than
# failing the run; the row's `note` carries the reason for readers.
KNOWN_ABSENT = {"KAIST"}


def normalise(name: str) -> str:
    """Fold a university name to a match key: accents stripped, parentheticals
    and a leading "The" dropped, punctuation flattened to single spaces."""
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c)).lower()
    folded = re.sub(r"\s*\([^)]*\)", " ", folded)
    folded = re.sub(r"^\s*the\s+", "", folded)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", folded).split())


def load_qs(path: Path) -> dict[str, dict]:
    """Index the CSV by match key. Collisions are dropped from the index so an
    ambiguous name fails the run instead of binding to an arbitrary row."""
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
    missing = sorted(set(QS_WEIGHTS) - set(rows[0]))
    if missing:
        sys.exit(f"CSV lacks indicator columns: {missing}")

    index: dict[str, dict | None] = {}
    for row in rows:
        key = normalise(row["Institution Name"])
        index[key] = None if key in index else row
    for name, qs_name in ALIASES.items():
        index[normalise(name)] = index.get(normalise(qs_name))
    return index


def indicator(row: dict, column: str) -> float | None:
    """One indicator score, or None where QS published nothing for it. Blanks
    occur only in IFR, ISR and SUS (5% each); the four heavy indicators are
    populated for all 1501 rows."""
    try:
        return float(row[column])
    except ValueError:
        return None


def base_score(row: dict) -> float:
    """QS weighted composite over the nine indicators.

    A blank indicator contributes zero rather than being excluded, because
    that is what QS itself does: on the 16 rows carrying both a published
    overall and a blank, scoring blanks as zero lands 1.07 off on average
    while renormalising over the published weights lands 3.19 off. Zero
    penalises an institution for unreported internationalisation, but it
    keeps this number faithful to the rank displayed beside it, which matters
    more than a fairness the source does not apply.
    """
    return round(
        sum((indicator(row, c) or 0.0) * w for c, w in QS_WEIGHTS.items()), 1
    )


def tilt(row: dict) -> float | None:
    """Employer reputation minus citations per faculty, or None if either is
    blank. Unlike base_score this cannot absorb a zero: a missing ER would
    read as a strong research tilt rather than as no data."""
    er, cpf = indicator(row, "ER SCORE"), indicator(row, "CPF SCORE")
    return None if er is None or cpf is None else round(er - cpf, 1)


def parse_rank(raw: str) -> int | str:
    """Point rank to int; QS bands ("701-710", "1401+") stay strings, matching
    the dataset schema. A leading "=" marks a joint rank and carries no data."""
    value = raw.strip().lstrip("=")
    return int(value) if value.isdigit() else value


def main(csv_path: Path) -> None:
    index = load_qs(csv_path)
    data = json.loads(DATASET.read_text())

    unmatched, changed = [], []
    for row in data["universities"]:
        qs = index.get(normalise(row["university"]))
        if qs is None:
            unmatched.append(row["university"])
            continue
        # First column is the rank, named for the cycle ("2026 Rank"); read it
        # positionally so a new cycle's CSV needs no edit here.
        rank = parse_rank(qs[list(qs)[0]])
        if rank != row["rank"]:
            changed.append(f"{row['university']}: {row['rank']} -> {rank}")
        row["rank"] = rank
        row["base_score"] = base_score(qs)
        row["tilt"] = tilt(qs)

    surprises = sorted(set(unmatched) - KNOWN_ABSENT)
    if surprises:
        sys.exit(
            f"{len(surprises)} rows did not match the CSV; nothing written.\n"
            "Add each to ALIASES (or KNOWN_ABSENT if QS dropped it):\n  "
            + "\n  ".join(surprises)
        )

    DATASET.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    scored = sum(1 for r in data["universities"] if r.get("base_score") is not None)
    print(f"matched {len(data['universities']) - len(unmatched)}, scored {scored}")
    print(f"absent from CSV (rank left as-is): {sorted(unmatched) or 'none'}")
    print(f"rank changes: {len(changed)}")
    for line in changed:
        print(f"  {line}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <qs-wur-full-list.csv>")
    main(Path(sys.argv[1]))
