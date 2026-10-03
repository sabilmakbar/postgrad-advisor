from datetime import date

from langchain_core.runnables import RunnableLambda

import app.cv as cv_mod
import app.fit as fit_mod
from app.cv import CVConfirmation, derive_experience, render_facts, render_interview_facts, review_facts
from app.fit import FitVerdict, fit_node
from app.requirements_search import SearchResult
from app.state import CVFacts, InterviewState, ShortlistEntry, WorkExperience


def _position(tenure, title="Scientist", employer="Acme", summary="Built NLP systems"):
    return WorkExperience(employer=employer, title=title, tenure=tenure, summary=summary)


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
                label="strong", reasons=["experience matches"], sources=["https://u.test"]
            )
        ),
    )
    fit_node(state)
    prompt = seen[0][-1][1]
    assert "'title': 'Scientist'" in prompt and "'tenure': '2 yrs'" in prompt
    assert "experience_months': 24" in prompt
    assert all(value in prompt for value in ("BSc Mathematics", "3.8 / 4.0", "Paper A", "IELTS 8.0"))
    assert "Secret Corp" not in prompt and "private summary" not in prompt and "'months'" not in prompt
