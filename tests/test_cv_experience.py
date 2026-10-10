from datetime import date

from langchain_core.runnables import RunnableLambda

import app.cv as cv_mod
import app.fit as fit_mod
from app.cv import (
    CVConfirmation,
    derive_experience,
    extract_facts,
    render_facts,
    render_interview_facts,
    review_facts,
    select_location_hint,
)
from app.fit import FitVerdict, fit_node
from app.requirements_search import SearchResult
from app.state import CVFacts, InterviewState, LocationEvidence, ShortlistEntry, WorkExperience


def _position(tenure, title="Scientist", employer="Acme", summary="Built NLP systems"):
    return WorkExperience(employer=employer, title=title, tenure=tenure, summary=summary)


def _evidence(kind, label, location, tenure, excerpt=None):
    return LocationEvidence(
        kind=kind,
        label=label,
        location=location,
        tenure=tenure,
        source_excerpt=excerpt or f"{label} | {location} | {tenure}",
    )


def test_supported_tenures_and_unparseable_value(monkeypatch):
    monkeypatch.setattr(cv_mod, "_today", lambda: date(2022, 1, 15))
    facts = derive_experience(CVFacts(experience=[
        _position("Jan 2020 - Jan 2021"),
        _position("Jan 2020 - Feb 2020"),
        _position("2020 – 2021"),
        _position("Jan 2020 to present"),
        _position("2020 - current"),
        _position("Jan 2020 - now"),
        _position("Jan 2020 - ongoing"),
        _position("3 yrs"),
        _position("since graduation"),
    ]))
    assert [p.months for p in facts.experience] == [12, 1, 12, 24, 24, 24, 24, 36, None]


def test_overlap_merge_duration_only_and_touching_boundaries(monkeypatch):
    monkeypatch.setattr(cv_mod, "_today", lambda: date(2022, 1, 15))
    facts = derive_experience(CVFacts(experience=[
        _position("Jan 2020 - Jan 2021"),
        _position("Jun 2020 - Jun 2021"),
        _position("Jun 2021 - Dec 2021"),
        _position("2 years"),
    ]))
    assert [p.months for p in facts.experience] == [12, 12, 6, 24]
    assert facts.experience_months == 47  # Jan 2020-Dec 2021 is 23, plus 24
    echo = render_facts(facts)
    assert "overlapping dated roles counted once" in echo
    assert "duration-only roles may overlap dated roles" in echo


def test_reported_em_dash_ranges_have_exact_durations_and_total(monkeypatch):
    monkeypatch.setattr(cv_mod, "_today", lambda: date(2026, 10, 4))
    facts = derive_experience(CVFacts(experience=[
        _position("Oct 2024 — Present", "Current"),
        _position("Feb 2024 — Jul 2024", "Short"),
        _position("Sep 2021 — Jul 2024", "Long"),
        _position("Nov 2020 — Sep 2021", "Earlier"),
    ]))
    assert [position.months for position in facts.experience] == [24, 5, 34, 10]
    assert facts.experience_months == 68
    echo = render_facts(facts)
    assert all(f"{months} months" in echo for months in (24, 5, 34, 10, 68))
    assert "not derived" not in echo


def test_context_uses_three_most_recent_positions_with_summaries(monkeypatch):
    monkeypatch.setattr(cv_mod, "_today", lambda: date(2024, 1, 1))
    facts = derive_experience(CVFacts(
        degree="BSc Mathematics",
        publications=["one", "two"],
        experience=[
            _position("2018 - 2019", "Old", summary="old summary"),
            _position("2022 - present", "Newest", summary="new summary"),
            _position("2020 - 2021", "Middle", summary="middle summary"),
            _position("2 yrs", "Duration", summary="duration summary"),
        ],
    ))
    context = render_interview_facts(facts)
    assert "BSc Mathematics" in context and "publications=2" in context
    assert context.index("Newest") < context.index("Middle") < context.index("Old")
    assert "Duration" not in context
    assert all(text in context for text in ("new summary", "middle summary", "old summary"))


def test_current_location_is_echoed_and_can_be_replaced_or_cleared(monkeypatch):
    facts = CVFacts(current_location="Jakarta, Indonesia")
    assert "current location: Jakarta, Indonesia" in render_facts(facts)
    assert "current location=Jakarta, Indonesia" in render_interview_facts(facts)

    corrections = iter([
        CVFacts(current_location="Bandung, Indonesia"),
        CVFacts(current_location=None),
    ])
    monkeypatch.setattr(
        cv_mod,
        "structured",
        lambda schema, tier="light": RunnableLambda(
            lambda _: CVConfirmation(confirmed=False, corrections=next(corrections))
        ),
    )
    confirmed, facts = review_facts(facts, "I live in Bandung")
    assert confirmed is False and facts.current_location == "Bandung, Indonesia"
    assert "current location: Bandung, Indonesia" in render_facts(facts)
    confirmed, facts = review_facts(facts, "clear my location")
    assert confirmed is False and facts.current_location is None
    assert "current location: not found" in render_facts(facts)


def test_current_location_extraction_requires_explicit_support(monkeypatch):
    monkeypatch.setattr(
        cv_mod,
        "structured",
        lambda schema, tier="light": RunnableLambda(
            lambda _: CVFacts(current_location="Jakarta, Indonesia")
        ),
    )
    facts = extract_facts("Current location: Jakarta, Indonesia")
    assert facts.current_location == "Jakarta, Indonesia"


def test_current_location_extraction_clears_unsupported_model_value(monkeypatch):
    monkeypatch.setattr(
        cv_mod,
        "structured",
        lambda schema, tier="light": RunnableLambda(
            lambda _: CVFacts(current_location="London, United Kingdom")
        ),
    )
    cv_text = """Education: University of London
Employer: London Labs
Project: London mobility study
Citizenship: United Kingdom
Nationality: British
Address: London, United Kingdom"""
    assert extract_facts(cv_text).current_location is None


def test_location_evidence_is_grounded_and_remote_is_rejected(monkeypatch):
    kept = _evidence("employment", "Acme — Engineer", "Berlin, Germany", "2022 - Present")
    remote = _evidence(
        "employment", "Remote Co — Engineer", "London, United Kingdom", "2023 - Present"
    )
    unsupported = _evidence("education", "Example U — MSc", "Paris, France", "2020 - 2021")
    monkeypatch.setattr(
        cv_mod,
        "structured",
        lambda schema, tier="light": RunnableLambda(
            lambda _: CVFacts(location_evidence=[kept, remote, unsupported])
        ),
    )
    text = "Acme — Engineer | Berlin, Germany | 2022 - Present\nRemote Co — Engineer | London, United Kingdom | 2023 - Present | remote"
    facts = extract_facts(text)
    assert facts.location_evidence == [kept]


def test_location_hint_uses_newest_date_across_kinds_and_explicit_location_wins(monkeypatch):
    monkeypatch.setattr(cv_mod, "_today", lambda: date(2026, 10, 10))
    employment = _evidence("employment", "Acme — Engineer", "Berlin, Germany", "2022 - 2024")
    education = _evidence("education", "Oxford — DPhil", "Oxford, United Kingdom", "2024 - Present")
    facts = CVFacts(location_evidence=[employment, education])
    assert select_location_hint(facts) == education
    assert select_location_hint(facts.model_copy(update={"current_location": "Jakarta, Indonesia"})) is None


def test_location_hint_rejects_conflicting_ties_complete_ties_and_unparseable_dates():
    berlin = _evidence("employment", "A", "Berlin, Germany", "2022 - 2024")
    paris = _evidence("education", "B", "Paris, France", "2023 - 2024")
    duplicate = _evidence("education", "C", "Berlin, Germany", "2022 - 2024")
    unknown = _evidence("employment", "D", "Rome, Italy", "since graduation")
    assert select_location_hint(CVFacts(location_evidence=[berlin, paris])) is None
    assert select_location_hint(CVFacts(location_evidence=[berlin, duplicate])) is None
    assert select_location_hint(CVFacts(location_evidence=[unknown])) is None


def test_location_hint_same_location_tie_uses_latest_start():
    older = _evidence("employment", "A", "Berlin, Germany", "2020 - 2024")
    newer = _evidence("education", "B", "Berlin, Germany", "2022 - 2024")
    assert select_location_hint(CVFacts(location_evidence=[older, newer])) == newer


def test_confirmed_correction_must_be_re_echoed_before_later_confirmation(monkeypatch):
    replies = iter([
        CVConfirmation(
            confirmed=True,
            corrections=CVFacts(current_location="Bandung, Indonesia"),
        ),
        CVConfirmation(
            confirmed=True,
            corrections=CVFacts(current_location=None),
        ),
        CVConfirmation(confirmed=True),
    ])
    monkeypatch.setattr(
        cv_mod,
        "structured",
        lambda schema, tier="light": RunnableLambda(lambda _: next(replies)),
    )
    facts = CVFacts(current_location="Jakarta, Indonesia")
    confirmed, facts = review_facts(facts, "yes, but I now live in Bandung")
    assert confirmed is False and facts.current_location == "Bandung, Indonesia"
    confirmed, facts = review_facts(facts, "yes, but clear my location")
    assert confirmed is False and facts.current_location is None
    confirmed, facts = review_facts(facts, "yes, that is now correct")
    assert confirmed is True and facts.current_location is None


def test_work_history_correction_replaces_whole_list_and_rederives(monkeypatch):
    original = derive_experience(CVFacts(experience=[_position("5 yrs", "Old")]))
    corrected = CVFacts(experience=[_position("1 yr", "New").model_copy(update={"months": 999})])
    monkeypatch.setattr(
        cv_mod,
        "structured",
        lambda schema, tier="light": RunnableLambda(
            lambda _: CVConfirmation(confirmed=False, corrections=corrected)
        ),
    )
    confirmed, facts = review_facts(original, "replace it")
    assert confirmed is False
    assert [p.title for p in facts.experience] == ["New"]
    assert facts.experience[0].months == 12 and facts.experience_months == 12

    empty = CVFacts(experience=[])
    monkeypatch.setattr(
        cv_mod,
        "structured",
        lambda schema, tier="light": RunnableLambda(
            lambda _: CVConfirmation(confirmed=False, corrections=empty)
        ),
    )
    _, facts = review_facts(original, "clear it")
    assert facts.experience == [] and facts.experience_months is None
    assert "none found" in render_facts(facts)


def test_fit_projects_only_allowed_work_fields(monkeypatch):
    facts = derive_experience(CVFacts(
        degree="BSc Mathematics",
        gpa="3.8 / 4.0",
        publications=["Paper A"],
        english_test="IELTS 8.0",
        current_location="Jakarta, Indonesia",
        experience=[_position("2 yrs", employer="Secret Corp", summary="private summary")],
    ))
    state = InterviewState(
        degree_intent="phd",
        field="mathematics",
        target_regions=["UK"],
        cv_facts=facts,
        cv_confirmed=True,
        shortlist=[ShortlistEntry(
            university="Test U", country="UK", rank_context="rank 1",
            degree_type="phd", why_it_fits="fit",
        )],
    )
    seen = []
    monkeypatch.setattr(
        fit_mod, "search_requirements",
        lambda q: SearchResult(query=q, text="Requires relevant experience", sources=["https://u.test"]),
    )
    monkeypatch.setattr(
        fit_mod, "structured",
        lambda schema, tier="heavy": RunnableLambda(
            lambda prompt: seen.append(prompt) or FitVerdict(
                label="strong", reasons=["experience matches"], sources=["https://u.test"],
                sufficient=True,
            )
        ),
    )
    fit_node(state)
    prompt = seen[0][-1][1]
    assert "'title': 'Scientist'" in prompt and "'tenure': '2 yrs'" in prompt
    assert "experience_months': 24" in prompt
    assert all(value in prompt for value in ("BSc Mathematics", "3.8 / 4.0", "Paper A", "IELTS 8.0"))
    assert "Secret Corp" not in prompt and "private summary" not in prompt and "'months'" not in prompt
    assert "Jakarta" not in prompt and "current_location" not in prompt
