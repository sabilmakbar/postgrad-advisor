"""Offline tests for code-selected, model-annotated shortlists."""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command
from pydantic import ValidationError

import app.graph as graph_mod
import app.shortlist as shortlist_mod
from app.dataset import ProgramDataset, University, load_dataset
from app.graph import build_graph
from app.prompts import CV_REQUEST, SHORTLIST_INSTRUCTIONS
from app.shortlist import (
    FIELD_ALIASES, FIELD_BUCKETS, MAX_SIZE, MIN_SIZE, MISSING_ANNOTATION,
    TILT_WEIGHT,
    Pick, Picks, _field_match, _render, humanize_degree_text, prefilter,
    shortlist_node,
)
from app.state import InterviewState, SlotUpdate


DATASET = load_dataset().universities
DATASET_NAMES = {u.university for u in DATASET}


def _state(**over):
    base = dict(degree_intent="phd", field="computer science", target_regions=["Asia"],
                messages=[("ai", "Q-region"), ("human", "Asia")])
    return InterviewState(**{**base, **over})


def _wire(monkeypatch, annotations):
    calls = []

    def fake_structured(schema, tier="heavy"):
        assert schema is Picks and tier == "heavy"

        def answer(msgs):
            calls.append(msgs)
            return Picks(picks=[Pick(why_it_fits=text) for text in annotations])

        return RunnableLambda(answer)

    monkeypatch.setattr(shortlist_mod, "structured", fake_structured)
    return calls


def _uni(name, country, continent, rank, fields, *, base_score=50.0, tilt=0.0):
    return University(
        university=name, country=country, continent=continent, rank=rank,
        rank_scope="world", fields=fields,
        degree_types=["masters_course", "masters_research", "phd"],
        base_score=base_score, tilt=tilt, verified=True, confidence="high",
    )


def test_code_selects_rows_and_model_only_annotates(monkeypatch):
    state = _state()
    selected, notes = prefilter(state, DATASET)
    assert notes == [] and len(selected) >= MIN_SIZE
    expected = selected[:MAX_SIZE]
    calls = _wire(monkeypatch, [f"fit-{i}" for i in range(len(expected))])
    out = shortlist_node(state)
    assert len(calls) == 1
    assert [e.university for e in out["shortlist"]] == [u.university for u in expected]
    assert [e.why_it_fits for e in out["shortlist"]] == [f"fit-{i}" for i in range(len(expected))]
    assert {e.university for e in out["shortlist"]} <= DATASET_NAMES
    assert out["handed_off"] is True
    prompt = "\n".join(text for _, text in calls[0])
    assert "base_score" not in prompt and "tilt" not in prompt
    assert selected[0].university in prompt


def test_imported_unknown_coverage_is_not_excluded_or_claimed_verified(monkeypatch):
    pool = [_uni(f"Imported-{i}", "Malaysia", "Asia", i + 1, []) for i in range(8)]
    for entry in pool:
        entry.degree_types = []
    monkeypatch.setattr(shortlist_mod, "load_dataset", lambda: ProgramDataset(
        source="test import", retrieved="2026-11-02", universities=pool
    ))
    calls = _wire(monkeypatch, ["availability needs verification"] * 8)
    result = shortlist_node(_state(target_regions=["Malaysia"]))
    assert len(result["shortlist"]) == 8
    assert "unverified subject coverage or degree availability" in result["messages"][-1][1]
    prompt = "\n".join(text for _, text in calls[0])
    assert "observed subject tags: unknown" in prompt
    assert "degree availability: unverified" in prompt


def test_unknown_subject_candidate_remains_eligible_with_eight_known_matches():
    pool = [_uni(f"Known-{i}", "Malaysia", "Asia", i + 1, ["computer_science"]) for i in range(8)]
    unknown = _uni("Unknown", "Malaysia", "Asia", 1, [], base_score=100)
    unknown.degree_types = []
    selected, _ = prefilter(_state(target_regions=["Malaysia"]), [*pool, unknown])
    assert len(selected) == 9 and unknown in selected
    assert all(u.fields for u in selected[:8])  # observed field evidence controls ordering


@pytest.mark.parametrize("count", [0, 3, 9, 15])
def test_annotation_count_never_changes_selection(monkeypatch, count):
    state = _state()
    selected, _ = prefilter(state, DATASET)
    selected = selected[:MAX_SIZE]
    calls = _wire(monkeypatch, [f"annotation-{i}" for i in range(count)])
    out = shortlist_node(state)
    assert len(calls) == 1
    assert [e.university for e in out["shortlist"]] == [u.university for u in selected[:MAX_SIZE]]
    expected = [f"annotation-{i}" for i in range(min(count, len(selected)))]
    expected += [MISSING_ANNOTATION] * (len(selected) - len(expected))
    assert [e.why_it_fits for e in out["shortlist"]] == expected


def test_annotation_schema_forbids_old_identity_fields():
    assert set(Pick.model_fields) == {"why_it_fits"}
    with pytest.raises(ValidationError):
        Pick.model_validate({"index": 0, "university": "Made Up", "why_it_fits": "text"})
    with pytest.raises(ValidationError):
        Picks.model_validate({"picks": [], "unexpected": True})


def test_prompt_only_requests_ordered_annotations_and_keeps_caveat_rule():
    lowered = SHORTLIST_INSTRUCTIONS.lower()
    assert "annotate each" in lowered and "same order" in lowered
    assert "caveat plainly" in lowered
    assert "pick" not in lowered and "index" not in lowered and "numbered" not in lowered


def test_named_region_tier_beats_higher_scoring_unnamed_region():
    pool = [
        _uni("Outside-100", "Japan", "Asia", 1, ["computer_science"], base_score=100),
        _uni("Inside-20", "Germany", "Europe", 100, ["medicine"], base_score=20),
        _uni("Outside-99", "Japan", "Asia", 2, ["computer_science"], base_score=99),
        _uni("Inside-10", "France", "Europe", 101, ["medicine"], base_score=10),
    ]
    candidates, _ = prefilter(_state(target_regions=["Europe"]), pool)
    assert [u.university for u in candidates] == ["Inside-20", "Inside-10", "Outside-100", "Outside-99"]


def test_region_aliases_are_exact_and_do_not_match_country_substrings():
    pool = [
        _uni("US", "United States", "North America", 1, ["computer_science"]),
        _uni("AU", "Australia", "Oceania", 2, ["computer_science"]),
        _uni("UK", "United Kingdom", "Europe", 3, ["computer_science"]),
    ]
    us, _ = prefilter(_state(target_regions=["US"]), pool)
    uk, _ = prefilter(_state(target_regions=["UK"]), pool)
    assert us[0].university == "US"
    assert uk[0].university == "UK"


def test_field_tier_beats_higher_score_after_region_widening():
    pool = [
        _uni("Field-20", "Japan", "Asia", 100, ["computer_science"], base_score=20),
        _uni("Other-100", "Japan", "Asia", 1, ["medicine"], base_score=100),
    ]
    candidates, _ = prefilter(_state(target_regions=["Europe"]), pool)
    assert [u.university for u in candidates] == ["Field-20", "Other-100"]


def test_code_caps_selection_before_annotation_call(monkeypatch):
    pool = [
        _uni(f"U-{i}", "Japan", "Asia", i + 1, ["computer_science"], base_score=100 - i)
        for i in range(MAX_SIZE + 2)
    ]
    monkeypatch.setattr(
        shortlist_mod,
        "load_dataset",
        lambda: ProgramDataset(source="fixture", retrieved=None, universities=pool),
    )
    calls = _wire(monkeypatch, ["fit"] * (MAX_SIZE + 2))
    out = shortlist_node(_state())
    prompt = "\n".join(text for _, text in calls[0])
    assert len(out["shortlist"]) == MAX_SIZE
    assert [e.university for e in out["shortlist"]] == [f"U-{i}" for i in range(MAX_SIZE)]
    assert f"U-{MAX_SIZE}" not in prompt


def test_degree_intent_reverses_tilt_order_for_same_pool():
    assert TILT_WEIGHT == 15.0
    pool = [
        _uni("Employability", "Japan", "Asia", 2, ["computer_science"], base_score=70, tilt=80),
        _uni("Research", "Japan", "Asia", 1, ["computer_science"], base_score=71, tilt=-80),
    ]
    coursework, _ = prefilter(_state(degree_intent="masters_coursework"), pool)
    phd, _ = prefilter(_state(degree_intent="phd"), pool)
    assert [u.university for u in coursework] == ["Employability", "Research"]
    assert [u.university for u in phd] == ["Research", "Employability"]


def test_missing_base_score_sorts_last_within_tier():
    pool = [
        _uni("Missing", "Japan", "Asia", None, ["computer_science"], base_score=None, tilt=100),
        _uni("Scored", "Japan", "Asia", 1000, ["computer_science"], base_score=-10, tilt=-100),
    ]
    candidates, _ = prefilter(_state(), pool)
    assert [u.university for u in candidates] == ["Scored", "Missing"]


def test_missing_tilt_keeps_base_score_with_zero_adjustment():
    pool = [
        _uni("No-tilt", "Japan", "Asia", 1, ["computer_science"], base_score=70, tilt=None),
        _uni("Lower", "Japan", "Asia", 2, ["computer_science"], base_score=69, tilt=0),
    ]
    candidates, _ = prefilter(_state(degree_intent="masters_coursework"), pool)
    assert [u.university for u in candidates] == ["No-tilt", "Lower"]


def test_missing_score_keeps_published_rank_context(monkeypatch):
    state = _state()
    kaist_row = next(u for u in DATASET if u.university == "KAIST")
    pool = [_uni(f"Scored-{i}", "Japan", "Asia", i + 1, ["computer_science"]) for i in range(8)] + [kaist_row]
    monkeypatch.setattr(shortlist_mod, "load_dataset", lambda: ProgramDataset(
        source="score fixture", retrieved=None, universities=pool
    ))
    candidates, _ = prefilter(state, pool)
    assert candidates[-1].university == "KAIST"
    _wire(monkeypatch, ["fit"] * len(candidates))
    out = shortlist_node(state)
    kaist = next(e for e in out["shortlist"] if e.university == "KAIST")
    assert kaist.rank_context == candidates[-1].note
    assert "excluded from QS 2026" in kaist.rank_context


@pytest.mark.parametrize(("degree_type", "label"), [
    ("masters_course", "taught master's"),
    ("masters_research", "research master's"), ("phd", "PhD"),
])
def test_degree_labels_replace_canonical_tokens_at_render_boundary(degree_type, label):
    entry = shortlist_mod.ShortlistEntry(
        university="Test University", country="Testland", rank_context="QS world rank 1",
        degree_type=degree_type, why_it_fits="fixture",
    )
    rendered = _render([entry], [])
    assert f"; {label}." in rendered and f"; {degree_type}." not in rendered


@pytest.mark.parametrize(("token", "label"), [
    ("masters_coursework", "taught master's"), ("masters_course", "taught master's"),
    ("masters_research", "research master's"), ("phd", "PhD"),
])
def test_generated_degree_tokens_are_humanized(token, label):
    assert humanize_degree_text(f"Matches your {token} goal") == f"Matches your {label} goal"


def test_canonical_degree_token_never_enters_prompt_or_reply(monkeypatch):
    state = _state(degree_intent="masters_research")
    candidates, _ = prefilter(state, DATASET)
    calls = _wire(monkeypatch, ["Matches your masters_research goal"] * len(candidates))
    out = shortlist_node(state)
    prompt = "\n".join(text for _, text in calls[0])
    rendered = out["messages"][-1][1]
    assert "masters_research" not in prompt and "masters_coursework" not in prompt
    assert "masters_research" not in rendered
    assert "Matches your research master's goal" in rendered


SPARSE = (
    [_uni(f"EU-CS-{i}", "Germany", "Europe", i + 1, ["computer_science"]) for i in range(3)]
    + [_uni(f"EU-MED-{i}", "France", "Europe", 100 + i, ["medicine"]) for i in range(6)]
    + [_uni("OC-CS-0", "Australia", "Oceania", 50, ["computer_science"])]
    + [_uni("OC-MED-0", "New Zealand", "Oceania", 60, ["medicine"])]
    + [_uni(f"AS-CS-{i}", "Japan", "Asia", 200 + i, ["computer_science"]) for i in range(8)]
)


def test_widening_relaxes_field_first():
    candidates, notes = prefilter(_state(target_regions=["Europe"]), SPARSE)
    assert len(candidates) == 9 and all(u.continent == "Europe" for u in candidates)
    assert notes == ["the field was widened to related subjects because 3 matched in your target regions"]


def test_widening_relaxes_region_after_field():
    candidates, notes = prefilter(_state(target_regions=["Oceania"]), SPARSE)
    assert len(candidates) == len(SPARSE)
    assert notes == [
        "the field was widened to related subjects because 1 matched in your target regions",
        f"widened to nearby regions because too few matched in your target regions; {len(SPARSE)} candidates after widening",
    ]


def test_widening_is_stated_in_rendered_output(monkeypatch):
    monkeypatch.setattr(shortlist_mod, "load_dataset",
                        lambda: ProgramDataset(source="fixture", retrieved=None, universities=SPARSE))
    calls = _wire(monkeypatch, ["fit"] * 8)
    out = shortlist_node(_state(target_regions=["Europe"]))
    assert len(calls) == 1
    assert "the field was widened to related subjects because 3 matched" in out["messages"][-1][1]
    assert {e.university for e in out["shortlist"]} <= {u.university for u in SPARSE}


def test_thin_pool_still_gets_one_annotation_call(monkeypatch):
    tiny = [_uni(f"T-{i}", "Germany", "Europe", i + 1, ["computer_science"]) for i in range(5)]
    monkeypatch.setattr(shortlist_mod, "load_dataset",
                        lambda: ProgramDataset(source="fixture", retrieved=None, universities=tiny))
    calls = _wire(monkeypatch, [])
    out = shortlist_node(_state(target_regions=["Europe"]))
    assert len(calls) == 1 and len(out["shortlist"]) == 5
    assert all(e.why_it_fits == MISSING_ANNOTATION for e in out["shortlist"])
    assert "widened to nearby regions" in out["messages"][-1][1]


def test_field_aliases_resolve_before_matching():
    ai, ai_notes = prefilter(_state(field="AI", target_regions=["Asia"]), DATASET)
    ml, ml_notes = prefilter(_state(field="machine learning", target_regions=["Asia"]), DATASET)
    cs, cs_notes = prefilter(_state(field="computer science", target_regions=["Asia"]), DATASET)
    assert ai == ml == cs and len(ai) >= MIN_SIZE
    assert ai_notes == ml_notes == cs_notes == []
    assert all("computer_science" in u.fields or not u.fields for u in ai)


def test_alias_values_are_dataset_buckets_and_keys_are_not_substrings():
    assert set(FIELD_ALIASES.values()) <= FIELD_BUCKETS
    cs = _uni("CS-only", "Japan", "Asia", 1, ["computer_science"])
    assert not _field_match("maintenance engineering", cs)
    assert not _field_match("air quality engineering", cs)


def test_widening_distinguishes_unresolved_absent_and_few():
    _, unresolved = prefilter(_state(field="basket weaving", target_regions=["Europe"]), DATASET)
    _, absent = prefilter(_state(field="AI", target_regions=["Africa"]), SPARSE)
    _, few = prefilter(_state(field="computer science", target_regions=["Europe"]), SPARSE)
    assert "nothing in the dataset is classified" in unresolved[0]
    assert "rather than field fit" in unresolved[0]
    assert "exists in the dataset but not in your target regions" in absent[0]
    assert "because 3 matched" in few[0]


def test_canonical_fallback_counts_as_resolved():
    _, notes = prefilter(_state(field="science", target_regions=["Africa"]), DATASET)
    assert all("nothing in the dataset is classified" not in note for note in notes)


def test_graph_wiring_interview_complete_to_shortlist(monkeypatch):
    monkeypatch.setattr(graph_mod, "chat",
                        lambda tier="heavy": RunnableLambda(lambda _: AIMessage(content="Q")))
    monkeypatch.setattr(
        graph_mod, "structured",
        lambda schema, tier="heavy": RunnableLambda(lambda _: SlotUpdate(
            degree_intent="phd", field="computer science", target_regions=["Asia"])),
    )
    candidates, _ = prefilter(_state(), DATASET)
    _wire(monkeypatch, ["fit"] * len(candidates))
    graph = build_graph()
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({}, config)
    state = graph.invoke(Command(resume="a phd in CS, in Asia"), config)
    assert state["handed_off"] is True
    assert state["__interrupt__"][0].value == CV_REQUEST
    assert MIN_SIZE <= len(state["shortlist"]) <= MAX_SIZE
    assert {e.university for e in state["shortlist"]} <= DATASET_NAMES
    assert state["messages"][-1][0] == "ai"
