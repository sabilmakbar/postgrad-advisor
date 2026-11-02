"""Download/import QS CSVs and rebuild the institution pool without model calls.

Preview: uv run python -m scripts.refresh_university_pool --world FILE_OR_HTTPS_URL
Apply: add --apply. Subject tables: --subject computer_science:2026=FILE_OR_URL
Subjects must use the app's field buckets. Exports must be CSV, not login/HTML pages.
"""

import argparse
import csv
import hashlib
import io
from itertools import islice
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zipfile import BadZipFile

from app.dataset import DATASET_PATH, ProgramDataset, SubjectRanking, University, load_dataset
from app.shortlist import FIELD_BUCKETS
from scripts.ingest_qs import ALIASES, QS_WEIGHTS, base_score, normalise, parse_rank, tilt

MAX_BYTES = 20_000_000
COUNTRY_ALIASES = {
    "United States of America": "United States",
    "Republic of Korea": "South Korea",
    "Hong Kong SAR, China": "Hong Kong SAR",
    "Russian Federation": "Russia",
    "Iran (Islamic Republic of)": "Iran",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
}
SOUTH_AMERICA = {
    "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Guyana",
    "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela", "French Guiana",
}
SUBJECT_ONLY_REGIONS = {
    "Algeria": "Africa", "Tanzania": "Africa", "Zimbabwe": "Africa",
    "Mauritius": "Africa", "Nepal": "Asia",
}
FACULTY_BUCKETS = {
    "Arts & Humanities": "arts_humanities", "Engineering & Technology": "engineering",
    "Life Sciences & Medicine": "life_sciences", "Natural Sciences": "natural_sciences",
    "Social Sciences & Management": "social_sciences",
}
SUBJECT_BUCKETS = {
    "Architecture / Built Environment": "engineering",
    "Computer Science & Information Systems": "computer_science",
    "Data Science and Artificial Intelligence": "computer_science",
    **dict.fromkeys(("Dentistry", "Medicine", "Nursing", "Pharmacy & Pharmacology", "Veterinary Science"), "medicine"),
    "Psychology": "social_sciences", "Economics & Econometrics": "economics", "Law": "law",
    **dict.fromkeys(("Accounting & Finance", "Business & Management Studies", "Marketing",
                     "Hospitality & Leisure Management"), "business"),
    "Sports-related Subjects": "life_sciences", "Statistics & Operational Research": "natural_sciences",
}
# Reviewed name variants in the official 2026 subject and overall exports.
# Campus-qualified Beijing/East China institutions deliberately remain distinct.
SOURCE_ALIASES = {
    **ALIASES, "UCL": "UCL (University College London)",
    'American University of Ras Al Khaimah': 'American University of Ras Al Khaimah ( AURAK )',
    'American University of Sharjah': 'American University of Sharjah (AUS)',
    'Arab Academy for Science, Technology and Maritime Transport (AASTMT)': 'Arab Academy for Science, Technology and Maritime Transport',
    'Auckland University of Technology (AUT)': 'Auckland University of Technology',
    'Australian National University (ANU)': 'Australian National University',
    'Azerbaijan State University of Economics (UNEC)': 'Azerbaijan State University of Economics',
    'Azerbaijan University of Architecture and Construction (AzUAC)': 'Azerbaijan University of Architecture and Construction',
    'Chung-Ang University (CAU)': 'Chung-Ang University',
    'City University of Hong Kong (CityUHK)': 'City University of Hong Kong',
    'ETH Zurich': 'ETH Zurich (Swiss Federal Institute of Technology)',
    'European University of Lefke': 'European University of Lefke (EUL)',
    'Gebze Technical University (GTU)': 'Gebze Technical University',
    'Hamad Bin Khalifa University (HBKU)': 'Hamad Bin Khalifa University',
    'Imam Abdulrahman Bin Faisal University (IAU)': 'Imam Abdulrahman Bin Faisal University (formerly University of Dammam)',
    'Indian Institute of Technology Kharagpur (IIT-KGP)': 'Indian Institute of Technology Kharagpur (IITKGP)',
    'Industrial University of Ho Chi Minh City (IUH)': 'Industrial University of Ho Chi Minh City',
    'Institut Teknologi Sepuluh Nopember (ITS Surabaya)': 'Institut Teknologi Sepuluh Nopember (ITS)',
    'Jeonbuk National University': 'Jeonbuk National University (formerly known as Chonbuk National University)',
    'Kazan (Volga region) Federal University': 'Kazan Federal University',
    'Kindai University (Kinki University)': 'Kindai University (previously Kinki University)',
    "King Mongkut's University of Technology North Bangkok (KMUTNB)": "King Mongkut's University of Technology North Bangkok",
    "King's College London": "King's College London (KCL)",
    'L.N. Gumilyov Eurasian National University (ENU)': 'L.N. Gumilyov Eurasian National University',
    'Macquarie University (Sydney, Australia)': 'Macquarie University',
    'National Sun Yat-sen University (NSYSU)': 'National Sun Yat-sen University',
    'National Taiwan Normal University (NTNU)': 'National Taiwan Normal University',
    'National Yang Ming Chiao Tung University (NYCU Taiwan)': 'National Yang Ming Chiao Tung University',
    'Norwegian University of Life Sciences': 'Norwegian University of Life Sciences (UMB)',
    'Pontificia Universidad Católica de Chile (UC)': 'Pontificia Universidad Católica de Chile',
    'Pontificia Universidad Católica del Ecuador (PUCE)': 'Pontificia Universidad Catolica del Ecuador',
    'Pontifícia Universidade Católica de São Paulo': 'Pontificia Universidade Católica de São Paulo (PUC -SP)',
    'Pontifícia Universidade Católica do Rio de Janeiro': 'Pontificia Universidade Católica do Rio de Janeiro (PUC - Rio)',
    'Queen Mary University of London': 'Queen Mary University of London (QMUL)',
    'Queensland University of Technology (QUT)': 'Queensland University of Technology',
    'SRM Institute of Science and Technology': 'SRM Institute of Science and Technology (f.k.a SRM University)',
    'Satbayev University': 'Satbayev University (previously Kazakh Technical University named after K.I. Satpaev)',
    'Sathyabama Institute of Science and Technology (Deemed to be University)': 'Sathyabama Institute of Science and Technology',
    'Sorbonne University': 'Sorbonne University (merged from Paris IV & UPMC)',
    'South Ural State University (National Research University)': 'South Ural State University',
    'Sungkyunkwan University (SKKU)': 'Sungkyunkwan University',
    'Széchenyi István University (University of Győr)': 'Széchenyi István University',
    'Taipei Medical University (TMU)': 'Taipei Medical University',
    'Tallinn University of Technology (TalTech)': 'Tallinn University of Technology',
    "Taylor's University": "Taylor's University (Malaysia)",
    'Technische Universität Berlin (TU Berlin)': 'Technische Universität Berlin',
    'Tecnológico de Monterrey': 'Tecnológico de Monterrey (ITESM)',
    'Telkom University': 'Telkom University (aka Telkom Institute of Technology)',
    'The Catholic University of Korea (CUK)': 'The Catholic University of Korea',
    'The Chinese University of Hong Kong (CUHK)': 'The Chinese University of Hong Kong',
    'The University of New South Wales (UNSW Sydney)': 'The University of New South Wales',
    'Tokyo University of Science': 'Tokyo University of Science (RIKADAI)',
    'UNIVERSITI TEKNIKAL MALAYSIA MELAKA': 'Universiti Teknikal Malaysia Melaka (UTeM)',
    'Ulsan National Institute of Science and Technology (UNIST)': 'Ulsan National Institute of Science and Technology',
    'Universidad Autónoma de Nuevo León': 'Universidad Autónoma de Nuevo León (UANL)',
    'Universidad Autónoma del Estado de México (UAEMex)': 'Universidad Autónoma del Estado de México',
    'Universidad Carlos III de Madrid (UC3M)': 'Universidad Carlos III de Madrid',
    'Universidad Nacional de Cuyo (UNCUYO)': 'Universidad Nacional de Cuyo',
    'Universidad Nacional de La Plata (UNLP)': 'Universidad Nacional de La Plata',
    'Universidad Nacional de Rosario (UNR)': 'Universidad Nacional de Rosario',
    'Universidad Peruana Cayetano Heredia (UPCH)': 'Universidad Peruana Cayetano Heredia',
    'Universidad Politécnica de Madrid (UPM)': 'Universidad Politécnica de Madrid',
    'Universidad Técnica Federico Santa María (USM)': 'Universidad Técnica Federico Santa María',
    'Universidad de Alcalá (Madrid)': 'Universidad de Alcalá',
    'Universidad de Buenos Aires (UBA)': 'Universidad de Buenos Aires',
    'Universidad de Monterrey (UDEM)': 'Universidad de Monterrey',
    'Universidad de Palermo (UP)': 'Universidad de Palermo',
    'Universidad del Desarrollo (UDD)': 'Universidad del Desarrollo',
    'Universidade Federal de São Paulo': 'Universidade Federal de São Paulo (UNIFESP)',
    'Universidade Federal de Viçosa (UFV)': 'Universidade Federal de Viçosa',
    'Universidade de Brasília': 'Universidade de Brasilia (UnB)',
    'Universidade de São Paulo': 'Universidade de São Paulo (USP)',
    'Universitat Pompeu Fabra (Barcelona)': 'Universitat Pompeu Fabra',
    'Universitat de Lleida (University of Lleida)': 'Universitat de Lleida',
    'Universite libre de Bruxelles': 'Université Libre de Bruxelles (ULB)',
    'Universiti Malaysia Perlis': 'Universiti Malaysia Perlis (UniMAP)',
    'Universiti Teknologi Malaysia': 'Universiti Teknologi Malaysia (UTM)',
    'University of Cyprus (UCY)': 'University of Cyprus',
    'University of Messina (UniME)': 'University of Messina',
    'University of Southern Denmark (SDU)': 'University of Southern Denmark',
    'University of Victoria (UVic)': 'University of Victoria',
    'University of Zurich': 'University of Zurich (UZH)',
    'Università degli Studi della Tuscia (University of Tuscia)': "Universita' degli Studi della Tuscia",
    'Université catholique de Louvain (UCLouvain)': 'Université Catholique de Louvain (UCL)',
    'Wroclaw University of Science and Technology (Wrocław Tech)': 'Wroclaw University of Science and Technology (WUST)',
}


def read_bytes(source: str) -> bytes:
    if urlparse(source).scheme:
        if urlparse(source).scheme != "https":
            raise ValueError("Remote sources must use HTTPS; local file paths are also supported")
        if urlparse(source).username or urlparse(source).password:
            raise ValueError("Use a public CSV URL without embedded credentials")
        with urlopen(Request(source, headers={"User-Agent": "postgrad-advisor-pool-refresh/1"}), timeout=30) as response:
            if urlparse(response.url).scheme != "https":
                raise ValueError("Source redirected away from HTTPS")
            raw = response.read(MAX_BYTES + 1)
    else:
        with Path(source).open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Source exceeds 20 MB")
    return raw


def read_source(source: str) -> tuple[list[dict], str]:
    raw = read_bytes(source)
    text = raw.decode("utf-8-sig")
    if text.lstrip().startswith("<"):
        raise ValueError("Source returned HTML, not a CSV export; download an accessible CSV first")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError("Source contains no CSV rows")
    return rows, hashlib.sha256(raw).hexdigest()


def read_subject_workbook(source: str) -> tuple[dict, str]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise ValueError("Excel import needs: uv sync --group data") from exc
    raw = read_bytes(source)
    try:
        workbook = load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    except BadZipFile as exc:
        raise ValueError("Source is not a readable Excel workbook") from exc
    tables, bucket = {}, None
    try:
        for sheet in workbook:
            if sheet.title.casefold() in {"index", "methodology"}:
                continue
            rows = sheet.iter_rows(values_only=True)
            heading = list(islice(rows, 4))
            if len(heading) != 4:
                raise ValueError(f"Incomplete subject workbook sheet: {sheet.title}")
            title, label_row, spacer, headers = heading
            label = str(label_row[0] or "").strip()
            if label in FACULTY_BUCKETS:
                bucket = FACULTY_BUCKETS[label]
                continue  # broad faculty membership does not establish a narrow subject
            if bucket is None or len(headers) < 4 or tuple(headers[2:4]) != ("INSTITUTION", "COUNTRY/TERRITORY"):
                raise ValueError(f"Unsupported subject workbook layout: {sheet.title}")
            year = int(headers[0])
            if not 2000 <= year <= 2099 or str(year) not in str(title[0]):
                raise ValueError(f"Inconsistent subject workbook year: {sheet.title}")
            records = []
            for row in rows:
                if row[0] is None or not str(row[0]).strip():
                    continue
                records.append({"Rank": str(row[0]), "Institution Name": str(row[2] or "").strip(),
                                "Country/Territory": str(row[3] or "").strip()})
            if not records or label in tables:
                raise ValueError(f"Empty or duplicate subject sheet: {sheet.title}")
            tables[label] = (year, records, SUBJECT_BUCKETS.get(label, bucket))
    finally:
        workbook.close()
    if not tables:
        raise ValueError("No subject ranking tables found")
    return tables, hashlib.sha256(raw).hexdigest()


def column(row: dict, *names: str) -> str:
    for name in names:
        if name in row and row[name] is not None:
            return row[name].strip()
    raise ValueError(f"CSV needs one of these columns: {', '.join(names)}")


def identity(name: str, country: str) -> tuple[str, str]:
    value = unicodedata.normalize("NFKD", SOURCE_ALIASES.get(name, name)).casefold()
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r"^\s*the\s+", "", value)
    # Parenthetical campuses are identity-bearing: Beijing must not merge into
    # another university just because a relaxed name normalizer drops brackets.
    return " ".join(re.findall(r"[a-z0-9]+", value)), COUNTRY_ALIASES.get(country, country)


def row_identity(row: dict) -> tuple[str, str]:
    key = identity(column(row, "Institution Name", "Institution", "University"),
                   column(row, "Country/Territory", "Country"))
    if not key[0] or not key[1]:
        raise ValueError("Every source row needs a nonempty institution name and country")
    return key


def checked_rank(value: str) -> int | str:
    rank = parse_rank(value)
    if isinstance(rank, int):
        if rank <= 0:
            raise ValueError(f"Invalid rank: {value!r}")
    elif not re.fullmatch(r"[1-9]\d*(?:-[1-9]\d*|\+)", rank):
        raise ValueError(f"Invalid rank: {value!r}")
    elif "-" in rank and int(rank.split("-")[0]) > int(rank.split("-")[1]):
        raise ValueError(f"Reversed rank band: {value!r}")
    return rank


def build_pool(world: list[dict], previous: ProgramDataset,
               subjects: dict[str, tuple[int, list[dict], str]], provenance: list[dict]) -> tuple[ProgramDataset, dict]:
    rank_columns = [k for k in world[0] if re.fullmatch(r"20\d{2} Rank", k)]
    if len(rank_columns) != 1:
        raise ValueError("World CSV needs one explicit year rank column, e.g. '2026 Rank'")
    rank_column = rank_columns[0]
    year = int(rank_column[:4])
    previous_year = previous.ranking_year
    if previous_year is None and (match := re.search(r"20\d{2}", previous.source)):
        previous_year = int(match.group())
    old, relaxed_old = {}, {}
    for u in previous.universities:
        key = identity(u.source_name or u.university, u.country)
        if key in old:
            raise ValueError(f"Ambiguous existing institution identity: {key}")
        old[key] = u
        if not u.source_name:  # migrate legacy display names once; preserve future campus identities
            relaxed = (normalise(SOURCE_ALIASES.get(u.university, u.university)), key[1])
            relaxed_old.setdefault(relaxed, []).append(key)
    entries, seen, added, changed, matched_old = [], set(), [], [], set()
    name_counts = Counter(column(r, "Institution Name", "Institution", "University") for r in world)
    for row in world:
        key = row_identity(row)
        if key in seen or not key[0] or not key[1]:
            raise ValueError(f"Missing or ambiguous source identity: {key}")
        seen.add(key)
        for indicator in QS_WEIGHTS:
            value = column(row, indicator)
            if value and (not math.isfinite(float(value)) or not 0 <= float(value) <= 100):
                raise ValueError(f"Invalid {indicator} for {key}: {value}")
        continent = column(row, "Region", "Continent")
        if continent == "Americas":
            continent = "South America" if key[1] in SOUTH_AMERICA else "North America"
        name = column(row, "Institution Name", "Institution", "University")
        old_key = key
        if key not in old:
            relaxed = (normalise(SOURCE_ALIASES.get(name, name)), key[1])
            matches = relaxed_old.get(relaxed, [])
            if len(matches) == 1:
                old_key = matches[0]
        prior = old.get(old_key)
        if prior:
            if old_key in matched_old:
                raise ValueError(f"Ambiguous metadata match: {name}")
            matched_old.add(old_key)
        if prior is None and name_counts[name] > 1:
            name += f" ({key[1]})"
        data = prior.model_dump() if prior else dict(
            university=name, fields=[], degree_types=[], verified=False, confidence="low",
        )
        data.update(source_name=column(row, "Institution Name", "Institution", "University"),
                    country=key[1], continent=continent, rank=checked_rank(row[rank_column]),
                    rank_scope="world", base_score=base_score(row), tilt=tilt(row))
        if prior and (previous_year != year or prior.rank != data["rank"]):
            data.update(verified=False, confidence="low")
        entry = University.model_validate(data)
        entries.append(entry)
        if prior is None:
            added.append(entry.university)
        elif prior.rank != entry.rank:
            changed.append({"university": entry.university, "before": prior.rank, "after": entry.rank})
    indexed = {row_identity(row): entry for row, entry in zip(world, entries)}
    country_continents = {entry.country: entry.continent for entry in entries}
    for subject, (subject_year, rows, field) in subjects.items():
        if field not in FIELD_BUCKETS:
            raise ValueError(f"Unknown subject bucket: {field}")
        subject_seen = set()
        for row in rows:
            key = row_identity(row)
            if key in subject_seen:
                raise ValueError(f"Duplicate subject institution: {key}")
            subject_seen.add(key)
            if key not in indexed:
                continent = country_continents.get(key[1]) or SUBJECT_ONLY_REGIONS.get(key[1]) or column(row, "Region", "Continent")
                if continent == "Americas":
                    continent = "South America" if key[1] in SOUTH_AMERICA else "North America"
                prior = old.get(key)
                if prior:
                    matched_old.add(key)
                data = prior.model_dump() if prior else dict(
                    university=column(row, "Institution Name", "Institution", "University"),
                    fields=[], degree_types=[], verified=False, confidence="low",
                )
                note = f"Subject-ranked; no overall rank in imported QS {year} source."
                if prior and prior.note:
                    note = prior.note if note in prior.note else note + " " + prior.note
                data.update(source_name=column(row, "Institution Name", "Institution", "University"),
                            country=key[1], continent=continent, rank=None, rank_scope="world",
                            base_score=None, tilt=None, note=note, verified=False, confidence="low")
                if prior and prior.rank is not None:
                    changed.append({"university": prior.university, "before": prior.rank, "after": None})
                entry = University.model_validate(data)
                entries.append(entry)
                indexed[key] = entry
                seen.add(key)
                if prior is None:
                    added.append(entry.university)
            rank = checked_rank(column(row, f"{subject_year} Rank", "Rank"))
            entry = indexed[key]
            source = next(p for p in provenance if p["kind"] == subject)
            entry.subject_ranks[subject] = SubjectRanking(rank=rank, year=subject_year, source=source["source"], sha256=source["sha256"])
            if field not in entry.fields:
                entry.fields.append(field)
    absent = []
    for key, prior in old.items():
        if key not in matched_old:
            note = f"Absent from imported QS {year} source; not proof the institution is closed."
            if prior.note:
                note = prior.note if note in prior.note else note + " " + prior.note
            entries.append(prior.model_copy(update={
                "rank": None, "base_score": None, "tilt": None,
                "verified": False, "confidence": "low", "note": note,
            }))
            absent.append(prior.university)
    pool = ProgramDataset.model_validate({
        "source": f"QS World University Rankings {year}; refresh inputs: " + json.dumps(provenance),
        "ranking_year": year, "retrieved": date.today().isoformat(), "universities": entries,
    })
    report = {"ranking_year": year, "before": len(previous.universities), "after": len(entries),
              "added": added, "rank_changes": changed, "absent_retained_unranked": absent,
              "unknown_subject_coverage": sum(not e.fields for e in entries),
              "unverified_degree_availability": sum(not e.degree_types for e in entries)}
    return pool, report


def write_atomic(path: Path, pool: ProgramDataset) -> None:
    payload = pool.model_dump_json(indent=2) + "\n"
    ProgramDataset.model_validate_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True, help="QS world CSV path or directly accessible HTTPS CSV URL")
    parser.add_argument("--subject", action="append", default=[], metavar="BUCKET:YEAR=CSV", help="Optional subject CSV; repeat for other buckets")
    parser.add_argument("--subject-workbook", help="Official QS subject-ranking XLSX path or HTTPS URL; requires the data dependency group")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Existing metadata to preserve")
    parser.add_argument("--output", type=Path, help="Output cache; defaults to --dataset")
    parser.add_argument("--report", type=Path, help="Optional JSON change report")
    parser.add_argument("--apply", action="store_true", help="Replace output atomically; otherwise preview only")
    parser.add_argument("--allow-shrink", action="store_true", help="Allow a source with fewer than 80% of existing institutions")
    args = parser.parse_args(argv)
    try:
        world, digest = read_source(args.world)
        def source_reference(source):
            parts = urlparse(source)
            return parts._replace(query="", fragment="").geturl() if parts.scheme else Path(source).name

        provenance = [{"kind": "world", "source": source_reference(args.world), "sha256": digest}]
        subjects = {}
        source_paths = [args.world]
        for spec in args.subject:
            descriptor, sep, source = spec.partition("=")
            bucket, year_separator, subject_year = descriptor.partition(":")
            if not sep or bucket in subjects:
                raise ValueError("Each subject must be a unique BUCKET:YEAR=CSV source")
            if not year_separator or not re.fullmatch(r"20\d{2}", subject_year):
                raise ValueError("Subject sources need an explicit year: BUCKET:YEAR=CSV")
            rows, digest = read_source(source)
            source_paths.append(source)
            subjects[bucket] = (int(subject_year), rows, bucket)
            provenance.append({"kind": bucket, "year": int(subject_year), "source": source_reference(source), "sha256": digest})
        if args.subject_workbook:
            tables, digest = read_subject_workbook(args.subject_workbook)
            source_paths.append(args.subject_workbook)
            subjects.update(tables)
            provenance.extend({"kind": label, "year": table[0], "source": source_reference(args.subject_workbook),
                               "sha256": digest} for label, table in tables.items())
        output = args.output or args.dataset
        if args.report and args.report.resolve() in {output.resolve(), args.dataset.resolve()}:
            raise ValueError("Report path must differ from dataset and output paths")
        local_inputs = {Path(source).resolve() for source in source_paths if not urlparse(source).scheme}
        if output.resolve() in local_inputs or (args.report and args.report.resolve() in local_inputs):
            raise ValueError("Output and report paths must not overwrite a raw source file")
        previous = load_dataset(args.dataset)
        previous_overall_count = sum(u.rank is not None and u.rank_scope == "world" for u in previous.universities)
        baseline_count = previous_overall_count or len(previous.universities)
        if len(world) < baseline_count * .8 and not args.allow_shrink:
            raise ValueError("Source has fewer than 80% of existing institutions; check for partial downloads or use --allow-shrink")
        pool, report = build_pool(world, previous, subjects, provenance)
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if args.apply:
            write_atomic(output, pool)
        print(json.dumps({**{k: v for k, v in report.items() if k not in {"added", "rank_changes"}},
                          "added_count": len(report["added"]), "rank_change_count": len(report["rank_changes"]),
                          "applied": args.apply}, indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Refresh failed; dataset was not replaced: {exc}\n")


if __name__ == "__main__":
    main()
