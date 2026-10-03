"""Shortlister (story 004): scripted fake LLM, no live API.

Covers: 8-12 entries all mapped from pick indices to real dataset rows (AC 1,
SC-002) with the zero-results counterpart — a doctored response naming a fake
university or a bad index is dropped by the index-mapping guard; entry content
incl. null-rank context rendered from the dataset note, never a number (AC 2 +
dev note); explicit field-then-region widening on a sparse fixture (AC 3); and
graph wiring interview-complete -> shortlister -> state carries the shortlist
(AC 1); floor guard — under-return with one retry, notice when still short,
thin-pool path untouched (fix round, QA finding 1).
"""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

import app.graph as graph_mod
import app.model_layer as model_layer
import app.shortlist as shortlist_mod
from app.dataset import ProgramDataset, University, load_dataset
from app.graph import build_graph
from app.prompts import CV_REQUEST
from app.shortlist import (
    FIELD_ALIASES,
    FIELD_BUCKETS,
    MAX_SIZE,
    MIN_SIZE,
    Pick,
    Picks,
    _field_match,
    _render,
    humanize_degree_text,
    prefilter,
    shortlist_node,
)
from app.state import InterviewState, SlotUpdate


@pytest.fixture(autouse=True)
def _no_live_probes(monkeypatch):
    """Belt and braces: if anything falls through to real chain building, the
    id health probe must not hit the network."""
    monkeypatch.setattr(model_layer, "_probe_cache", {})
    monkeypatch.setattr(model_layer, "_probe_status", lambda provider, model_id: 200)


DATASET = load_dataset().universities
DATASET_NAMES = {u.university for u in DATASET}


def _state(**over):
    # Asia + computer science + phd: 9 real-dataset matches (no widening),
    # including the null-rank KAIST row.
    base = dict(
        degree_intent="phd",
        field="computer science",
        target_regions=["Asia"],
        messages=[("ai", "Q-region"), ("human", "Asia")],
    )
    return InterviewState(**{**base, **over})


def _wire(monkeypatch, picks):
    """Fake the ranking model to return `picks`; records tier and prompt."""
    seen = {}

    def fake_structured(schema, tier="heavy"):
        assert schema is Picks
        seen["tier"] = tier

        def _run(msgs):
            seen["msgs"] = msgs
            return picks

        return RunnableLambda(_run)

    monkeypatch.setattr(shortlist_mod, "structured", fake_structured)
    return seen


def _wire_seq(monkeypatch, responses):
    """Fake returning one scripted response per call; returns the call log
    (one msgs entry per structured invoke), so call counts are assertable."""
    calls = []

    def fake_structured(schema, tier="heavy"):
        assert schema is Picks

        def _run(msgs):
            calls.append(msgs)
            return responses[len(calls) - 1]  # IndexError = unexpected call

        return RunnableLambda(_run)

    monkeypatch.setattr(shortlist_mod, "structured", fake_structured)
    return calls


def _picks_for(candidates, indices):
    return Picks(
        picks=[
            Pick(index=i, university=candidates[i].university, why_it_fits=f"fit-{i}")
            for i in indices
        ]
    )


def test_entries_map_indices_to_dataset_rows(monkeypatch):
    # AC 1 + AC 2 (SC-002): 8-12 entries, every one a dataset row (checked
    # against the loaded dataset, not trusted), rendered with university,
    # country, rank context, matching degree type, and the pick's rationale.
    state = _state()
    candidates, notes = prefilter(state, DATASET)
    assert notes == [] and len(candidates) == 9
    seen = _wire(monkeypatch, _picks_for(candidates, range(9)))

    out = shortlist_node(state)
    entries = out["shortlist"]
    assert MIN_SIZE <= len(entries) <= MAX_SIZE
    assert {e.university for e in entries} <= DATASET_NAMES
    assert seen["tier"] == "heavy"  # FR-011: judgment prose, not slot-parsing
    assert out["handed_off"] is True

    by_name = {u.university: u for u in DATASET}
    first = entries[0]
    row = by_name[first.university]
    assert first.country == row.country
    assert first.degree_type == "phd"  # matches degree intent
    assert first.rank_context == f"QS {row.rank_scope} rank {row.rank}"

    role, rendered = out["messages"][-1]
    assert role == "ai"
    for e in entries:
        assert e.university in rendered
        assert e.why_it_fits in rendered
    assert first.rank_context in rendered and first.country in rendered
    assert "; PhD." in rendered
    assert "masters_course" not in rendered and "masters_research" not in rendered


@pytest.mark.parametrize(
    ("degree_type", "label"),
    [
        ("masters_course", "taught master's"),
        ("masters_research", "research master's"),
        ("phd", "PhD"),
    ],
)
def test_degree_labels_replace_canonical_tokens_at_render_boundary(
    degree_type, label
):
    entry = shortlist_mod.ShortlistEntry(
        university="Test University",
        country="Testland",
        rank_context="QS world rank 1",
        degree_type=degree_type,
        why_it_fits="fixture",
    )
    rendered = _render([entry], [])
    assert f"; {label}." in rendered
    assert f"; {degree_type}." not in rendered


@pytest.mark.parametrize(
    ("token", "label"),
    [
        ("masters_coursework", "taught master's"),
        ("masters_course", "taught master's"),
        ("masters_research", "research master's"),
        ("phd", "PhD"),
    ],
)
def test_generated_degree_tokens_are_humanized(token, label):
    assert humanize_degree_text(f"Matches your {token} goal") == (
        f"Matches your {label} goal"
    )


def test_canonical_degree_token_never_enters_heavy_shortlist_prompt_or_reply(
    monkeypatch,
):
    state = _state(degree_intent="masters_research")
    candidates, _ = prefilter(state, DATASET)
    calls = []

    def fake_structured(schema, tier="heavy"):
        assert schema is Picks and tier == "heavy"

        def answer(msgs):
            calls.append(msgs)
            return Picks(
                picks=[
                    Pick(
                        index=i,
                        university=candidates[i].university,
                        why_it_fits="Matches your masters_research goal",
                    )
                    for i in range(8)
                ]
            )

        return RunnableLambda(answer)

    monkeypatch.setattr(shortlist_mod, "structured", fake_structured)
    out = shortlist_node(state)

    prompt = "\n".join(text for _, text in calls[0])
    rendered = out["messages"][-1][1]
    assert "masters_research" not in prompt
    assert "masters_coursework" not in prompt
    assert "degree: phd" not in prompt
    assert "masters_research" not in rendered
    assert "Matches your research master's goal" in rendered


def test_doctored_response_is_dropped_by_index_mapping(monkeypatch):
    # Zero-results counterpart (dev note): a fake university on a valid index,
    # an out-of-range index, and a duplicate are all dropped by the mapping,
    # while the valid pick in the SAME response survives — the guard, proven
    # against a response forced to contain both cases.
    state = _state()
    candidates, _ = prefilter(state, DATASET)
    valid = Pick(index=0, university=candidates[0].university, why_it_fits="real")
    doctored = Picks(
        picks=[
            valid,
            Pick(index=1, university="University of Atlantis", why_it_fits="fake name"),
            Pick(index=999, university=candidates[0].university, why_it_fits="bad index"),
            Pick(index=0, university=candidates[0].university, why_it_fits="dupe"),
        ]
    )
    _wire(monkeypatch, doctored)

    out = shortlist_node(state)
    assert [e.university for e in out["shortlist"]] == [candidates[0].university]
    assert [e.why_it_fits for e in out["shortlist"]] == ["real"]
    assert "Atlantis" not in out["messages"][-1][1]


def test_output_capped_at_max_size(monkeypatch):
    # AC 1 upper bound is code-enforced, not prompt-trusted: 13 valid picks in,
    # 12 entries out.
    state = _state(field="engineering", target_regions=["Asia", "Europe"])
    candidates, notes = prefilter(state, DATASET)
    assert notes == [] and len(candidates) > MAX_SIZE
    _wire(monkeypatch, _picks_for(candidates, range(MAX_SIZE + 1)))

    out = shortlist_node(state)
    assert len(out["shortlist"]) == MAX_SIZE


def test_null_rank_sorts_last_and_renders_note(monkeypatch):
    # Dev note: KAIST (rank null) is never sorted as if top-ranked — it goes
    # last among the Asia CS candidates — and its ranking context is the
    # dataset note, not a number.
    state = _state()
    candidates, _ = prefilter(state, DATASET)
    assert candidates[0].rank is not None  # ranked rows lead
    assert candidates[-1].university == "KAIST" and candidates[-1].rank is None
    _wire(monkeypatch, _picks_for(candidates, range(9)))

    out = shortlist_node(state)
    kaist = next(e for e in out["shortlist"] if e.university == "KAIST")
    assert kaist.rank_context == candidates[-1].note
    assert "excluded from QS 2026" in kaist.rank_context


# --- widening (AC 3): sparse fixture dataset -------------------------------


def _uni(name, country, continent, rank, fields):
    return University(
        university=name,
        country=country,
        continent=continent,
        rank=rank,
        rank_scope="world",
        fields=fields,
        degree_types=["phd"],
        verified=True,
        confidence="high",
    )


SPARSE = (
    [_uni(f"EU-CS-{i}", "Germany", "Europe", i + 1, ["computer_science"]) for i in range(3)]
    + [_uni(f"EU-MED-{i}", "France", "Europe", 100 + i, ["medicine"]) for i in range(6)]
    + [_uni("OC-CS-0", "Australia", "Oceania", 50, ["computer_science"])]
    + [_uni("OC-MED-0", "New Zealand", "Oceania", 60, ["medicine"])]
    + [_uni(f"AS-CS-{i}", "Japan", "Asia", 200 + i, ["computer_science"]) for i in range(8)]
)


def test_widening_relaxes_field_first(monkeypatch):
    # Only 3 field matches in Europe (<8) -> field relaxed; the 9 in-region
    # rows suffice, so the region constraint holds. Widening is named.
    candidates, notes = prefilter(_state(target_regions=["Europe"]), SPARSE)
    assert len(candidates) == 9
    assert all(u.continent == "Europe" for u in candidates)
    assert notes == [
        "the field was widened to related subjects because 3 matched in your target regions"
    ]


def test_widening_relaxes_region_after_field(monkeypatch):
    # Field relax still leaves only 2 in Oceania -> region relaxed too; both
    # widenings named.
    candidates, notes = prefilter(_state(target_regions=["Oceania"]), SPARSE)
    assert len(candidates) == len(SPARSE)  # full degree pool
    assert notes == [
        "the field was widened to related subjects because 1 matched in your target regions",
        # QA finding 2: the count is the post-region-widening total.
        f"widened to nearby regions because too few matched in your target "
        f"regions; {len(SPARSE)} candidates after widening",
    ]


def test_widening_is_stated_in_rendered_output(monkeypatch):
    # AC 3: the widening statement reaches the user-facing output, and no
    # entry was invented to fill the gap (fixture membership holds).
    monkeypatch.setattr(
        shortlist_mod,
        "load_dataset",
        lambda: ProgramDataset(source="fixture", retrieved=None, universities=SPARSE),
    )
    state = _state(target_regions=["Europe"])
    candidates, _ = prefilter(state, SPARSE)
    _wire(monkeypatch, _picks_for(candidates, range(8)))

    out = shortlist_node(state)
    rendered = out["messages"][-1][1]
    assert "the field was widened to related subjects because 3 matched" in rendered
    assert {e.university for e in out["shortlist"]} <= {u.university for u in SPARSE}


# --- floor guard (fix round, QA finding 1) ---------------------------------


def test_under_return_retry_meets_floor(monkeypatch):
    # >=8 valid candidates but the model under-returns: exactly one retry that
    # names how many more grounded picks are needed; floor met, no notice.
    state = _state()
    candidates, _ = prefilter(state, DATASET)  # 9 candidates
    calls = _wire_seq(
        monkeypatch,
        [_picks_for(candidates, range(5)), _picks_for(candidates, range(5, 9))],
    )

    out = shortlist_node(state)
    assert len(calls) == 2
    assert MIN_SIZE <= len(out["shortlist"]) <= MAX_SIZE
    retry_text = calls[1][-1][1]  # the appended human retry message
    assert "3 more grounded picks" in retry_text  # 8 - 5 = how many more
    assert "could be grounded against the dataset" not in out["messages"][-1][1]


def test_under_return_twice_ships_short_with_notice(monkeypatch):
    # Retry also under-returns (same 5 picks, dupes dropped): the short list
    # ships, but never silently — the rendered output carries the notice.
    state = _state()
    candidates, _ = prefilter(state, DATASET)
    five = _picks_for(candidates, range(5))
    calls = _wire_seq(monkeypatch, [five, five])

    out = shortlist_node(state)
    assert len(calls) == 2
    assert len(out["shortlist"]) == 5
    assert (
        "only 5 picks could be grounded against the dataset"
        in out["messages"][-1][1]
    )


def test_thin_pool_no_retry_no_notice(monkeypatch):
    # The fully-widened pool itself has <8 rows: AC-3 widening notes already
    # cover it — one call, no retry, no grounding notice.
    tiny = [_uni(f"T-{i}", "Germany", "Europe", i + 1, ["computer_science"]) for i in range(5)]
    monkeypatch.setattr(
        shortlist_mod,
        "load_dataset",
        lambda: ProgramDataset(source="fixture", retrieved=None, universities=tiny),
    )
    state = _state(target_regions=["Europe"])
    candidates, notes = prefilter(state, tiny)
    assert len(candidates) == 5 and len(notes) == 2  # widened all the way
    calls = _wire_seq(monkeypatch, [_picks_for(candidates, range(5))])

    out = shortlist_node(state)
    rendered = out["messages"][-1][1]
    assert len(calls) == 1
    assert len(out["shortlist"]) == 5
    assert "could be grounded against the dataset" not in rendered
    assert "widened to nearby regions" in rendered


def test_field_aliases_resolve_before_matching():
    ai, ai_notes = prefilter(_state(field="AI", target_regions=["Asia"]), DATASET)
    ml, ml_notes = prefilter(
        _state(field="machine learning", target_regions=["Asia"]), DATASET
    )
    assert len(ai) == len(ml) == 9
    assert ai_notes == ml_notes == []
    assert all("computer_science" in u.fields for u in ai)


def test_alias_values_are_dataset_buckets_and_keys_are_not_substrings():
    assert set(FIELD_ALIASES.values()) <= FIELD_BUCKETS
    cs = _uni("CS-only", "Japan", "Asia", 1, ["computer_science"])
    assert not _field_match("maintenance engineering", cs)
    assert not _field_match("air quality engineering", cs)


def test_widening_distinguishes_unresolved_absent_and_few():
    _, unresolved = prefilter(
        _state(field="basket weaving", target_regions=["Europe"]), DATASET
    )
    _, absent = prefilter(_state(field="AI", target_regions=["Africa"]), DATASET)
    _, few = prefilter(
        _state(field="computer science", target_regions=["Europe"]), SPARSE
    )

    assert "nothing in the dataset is classified" in unresolved[0]
    assert "rather than field fit" in unresolved[0]
    assert "exists in the dataset but not in your target regions" in absent[0]
    assert "because 3 matched" in few[0]
    assert all(
        "only 0" not in note
        for notes in (unresolved, absent, few)
        for note in notes
    )


def test_canonical_fallback_counts_as_resolved():
    _, notes = prefilter(_state(field="science", target_regions=["Africa"]), DATASET)
    assert all("nothing in the dataset is classified" not in note for note in notes)


def test_graph_wiring_interview_complete_to_shortlist(monkeypatch):
    # Task 3 / AC 1: the interview-complete edge runs the shortlister and the
    # graph state carries its shortlist.
    monkeypatch.setattr(
        graph_mod, "chat", lambda tier="heavy": RunnableLambda(lambda _: AIMessage(content="Q"))
    )
    monkeypatch.setattr(
        graph_mod,
        "structured",
        lambda schema, tier="heavy": RunnableLambda(
            lambda _: SlotUpdate(
                degree_intent="phd", field="computer science", target_regions=["Asia"]
            )
        ),
    )
    candidates, _ = prefilter(_state(), DATASET)  # the state the answer yields
    _wire(monkeypatch, _picks_for(candidates, range(9)))

    g = build_graph()
    cfg = {"configurable": {"thread_id": "t"}}
    g.invoke({}, cfg)
    state = g.invoke(Command(resume="a phd in CS, in Asia"), cfg)

    assert state["handed_off"] is True
    # story 005: after the shortlist the run pauses at CV intake
    assert state["__interrupt__"][0].value == CV_REQUEST
    shortlist = state["shortlist"]
    assert MIN_SIZE <= len(shortlist) <= MAX_SIZE
    assert {e.university for e in shortlist} <= DATASET_NAMES
    assert state["messages"][-1][0] == "ai"  # rendered shortlist is the last message
