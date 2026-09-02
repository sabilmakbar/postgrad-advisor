"""Model-layer resilience: retry-then-fallback on transient errors, fail-fast on
the rest (AC 3), typed exhaustion (AC 4), shared per-(provider, model) limiters.
Story 007 adds tier tests: separation, light retry/fallback, degrade paths.

Providers are mocked; no live API calls.
"""

import httpx
import pytest
from langchain_core.runnables import RunnableLambda

import app.model_layer as model_layer
from app.model_layer import (
    _GOOGLE_TRANSIENT,
    AllProvidersExhausted,
    _build_models,
    _compose,
    chat,
    structured,
)
from app.config import settings


@pytest.fixture(autouse=True)
def _no_live_probes(monkeypatch):
    """Health probe answers 200 offline (fresh cache per test); dead-id tests
    override _probe_status themselves."""
    monkeypatch.setattr(model_layer, "_probe_cache", {})
    monkeypatch.setattr(model_layer, "_probe_status", lambda provider, model_id: 200)

FAST_BACKOFF = {"initial": 0.001, "max": 0.005}


class FakeRateLimit(Exception):
    """Stands in for a provider's transient error (429/5xx)."""


class FakeAuthError(Exception):
    """Stands in for a provider's non-transient error (401)."""


def _failing(counter: dict, exc: Exception) -> RunnableLambda:
    def _f(_):
        counter["calls"] += 1
        raise exc

    return RunnableLambda(_f)


def test_429_retries_then_falls_back():
    primary_calls = {"calls": 0}
    chain = _compose(
        [
            (_failing(primary_calls, FakeRateLimit("429")), (FakeRateLimit,)),
            (RunnableLambda(lambda _: "ok from fallback"), (FakeRateLimit,)),
        ],
        jitter_params=FAST_BACKOFF,
    )
    # no exception reaches the caller
    assert chain.invoke("hi") == "ok from fallback"
    # primary was retried (3 attempts) before falling back
    assert primary_calls["calls"] == 3


def test_401_skips_retries_and_falls_back():
    primary_calls = {"calls": 0}
    chain = _compose(
        [
            (_failing(primary_calls, FakeAuthError("401")), (FakeRateLimit,)),
            (RunnableLambda(lambda _: "ok from fallback"), (FakeRateLimit,)),
        ],
        jitter_params=FAST_BACKOFF,
    )
    assert chain.invoke("hi") == "ok from fallback"
    # non-transient error: exactly 1 attempt, no retries
    assert primary_calls["calls"] == 1


def test_all_providers_fail_raises_typed_error():
    chain = _compose(
        [
            (_failing({"calls": 0}, FakeRateLimit("429")), (FakeRateLimit,)),
            (_failing({"calls": 0}, FakeRateLimit("429")), (FakeRateLimit,)),
        ],
        jitter_params=FAST_BACKOFF,
    )
    with pytest.raises(AllProvidersExhausted) as exc_info:
        chain.invoke("hi")
    # plain-language message, original error kept as cause
    assert "failed" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None


def test_google_transient_predicate_matches_429_only():
    # google.genai's ClientError covers all 4xx; the layer's isinstance predicate
    # must retry 429 but not 401. ServerError (5xx) is transient too.
    from google.genai.errors import ClientError, ServerError

    assert isinstance(ClientError(429, {}), _GOOGLE_TRANSIENT)
    assert not isinstance(ClientError(401, {}), _GOOGLE_TRANSIENT)
    assert isinstance(ServerError(500, {}), _GOOGLE_TRANSIENT)


def test_rate_limiters_shared_across_chain_builds(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", "dummy")
    monkeypatch.setattr(settings, "groq_api_key", "dummy")
    first = [m for m, _ in _build_models()]
    second = [m for m, _ in _build_models()]
    # same limiter object per (provider, model id) across builds
    assert first[0].rate_limiter is second[0].rate_limiter  # gemini
    assert first[1].rate_limiter is second[1].rate_limiter  # groq
    # keyed per model, not one global limiter
    assert first[0].rate_limiter is not first[1].rate_limiter


# --- Story 007: tiers ---------------------------------------------------------

HEAVY_IDS = (settings.gemini_model, settings.groq_model)
LIGHT_IDS = (settings.gemini_light_model, settings.groq_light_model)


def _fake_providers(monkeypatch, fail: dict | None = None):
    """Replace the three provider classes with recording fakes and configure
    google+groq keys. Returns (constructed_ids, invoked_ids). `fail` maps a
    model id to the exception its fake raises on every invoke."""
    fail = fail or {}
    constructed: list[str] = []
    invoked: list[str] = []

    class FakeModel(RunnableLambda):
        def __init__(self, model, **kwargs):
            constructed.append(model)

            def _f(_):
                invoked.append(model)
                if model in fail:
                    raise fail[model]
                return f"ok:{model}"

            super().__init__(_f)

        def with_structured_output(self, schema):
            return self

    for cls in ("ChatGoogleGenerativeAI", "ChatGroq", "ChatOpenAI"):
        monkeypatch.setattr(model_layer, cls, FakeModel)
    monkeypatch.setattr(settings, "google_api_key", "dummy")
    monkeypatch.setattr(settings, "groq_api_key", "dummy")
    monkeypatch.setattr(settings, "cloudflare_api_token", None)
    return constructed, invoked


def _blank_light_ids(monkeypatch):
    monkeypatch.setattr(settings, "gemini_light_model", "")
    monkeypatch.setattr(settings, "groq_light_model", "")
    monkeypatch.setattr(settings, "workers_ai_light_model", "")


def test_tier_separation(monkeypatch):
    # AC 4: a light call constructs/invokes only light models; heavy only heavy.
    constructed, invoked = _fake_providers(monkeypatch)
    assert chat(tier="light").invoke("hi") == f"ok:{LIGHT_IDS[0]}"
    assert constructed == list(LIGHT_IDS)
    assert invoked == [LIGHT_IDS[0]]

    # forced-failure counterpart: the recorder DOES catch heavy construction —
    # the same "no heavy ids" assertions fail against the heavy chain.
    constructed.clear(), invoked.clear()
    assert chat().invoke("hi") == f"ok:{HEAVY_IDS[0]}"  # default tier is heavy
    assert constructed == list(HEAVY_IDS)
    assert invoked == [HEAVY_IDS[0]]
    assert not set(constructed).isdisjoint(HEAVY_IDS)


def test_structured_light_binds_and_uses_light_ids(monkeypatch):
    constructed, invoked = _fake_providers(monkeypatch)
    assert structured(dict, tier="light").invoke("hi") == f"ok:{LIGHT_IDS[0]}"
    assert constructed == list(LIGHT_IDS)


def test_structured_chains_are_tagged_nostream(monkeypatch):
    """Story 006 fix round: structured runs must never enter langgraph's
    messages-mode stream — raw schema JSON on the chat page otherwise.
    Paired opposite: the plain chat chain must NOT carry the tag."""
    from langgraph.constants import TAG_NOSTREAM

    _fake_providers(monkeypatch)
    for tier in ("light", "heavy"):
        assert TAG_NOSTREAM in structured(dict, tier=tier).config.get("tags", [])
    assert TAG_NOSTREAM not in (getattr(chat(), "config", None) or {}).get("tags", [])


def test_light_retry_then_fallback_same_seam(monkeypatch):
    # AC 3: light rides the same _compose — transient error retried 3x on the
    # light primary, then falls back to the next LIGHT model, never to heavy.
    constructed, invoked = _fake_providers(
        monkeypatch, fail={LIGHT_IDS[0]: httpx.ConnectError("boom")}
    )
    orig = model_layer._compose
    monkeypatch.setattr(
        model_layer, "_compose", lambda models, jitter_params=None: orig(models, FAST_BACKOFF)
    )
    assert chat(tier="light").invoke("hi") == f"ok:{LIGHT_IDS[1]}"
    assert invoked == [LIGHT_IDS[0]] * 3 + [LIGHT_IDS[1]]
    assert constructed == list(LIGHT_IDS)  # heavy never built


def test_light_unconfigured_degrades_to_heavy(monkeypatch, caplog):
    # AC 5: all light ids blanked = tier disabled -> heavy chain + logged notice.
    constructed, invoked = _fake_providers(monkeypatch)
    _blank_light_ids(monkeypatch)
    with caplog.at_level("WARNING"):
        assert chat(tier="light").invoke("hi") == f"ok:{HEAVY_IDS[0]}"
    assert constructed == list(HEAVY_IDS)
    assert "degrading to the heavy chain" in caplog.text


def test_light_exhaustion_mid_call_degrades_to_heavy(monkeypatch, caplog):
    # AC 5: every light provider fails -> heavy answers; heavy models are only
    # constructed AFTER the light chain exhausted (lazy degrade).
    constructed, invoked = _fake_providers(
        monkeypatch, fail={i: FakeAuthError(i) for i in LIGHT_IDS}
    )
    with caplog.at_level("WARNING"):
        assert chat(tier="light").invoke("hi") == f"ok:{HEAVY_IDS[0]}"
    assert constructed == list(LIGHT_IDS) + list(HEAVY_IDS)
    assert invoked == list(LIGHT_IDS) + [HEAVY_IDS[0]]
    assert "light tier exhausted" in caplog.text


def test_both_tiers_exhausted_raises(monkeypatch):
    # AC 5: AllProvidersExhausted only when heavy is exhausted too.
    _fake_providers(
        monkeypatch, fail={i: FakeAuthError(i) for i in (*LIGHT_IDS, *HEAVY_IDS)}
    )
    with pytest.raises(AllProvidersExhausted):
        chat(tier="light").invoke("hi")


def test_dead_id_warned_and_skipped_at_build(monkeypatch, caplog):
    # AC 2 / health check: a 404-on-generation id is skipped with a warning;
    # counterpart in the same run: probe 200 keeps it.
    constructed, _ = _fake_providers(monkeypatch)
    monkeypatch.setattr(
        model_layer,
        "_probe_status",
        lambda provider, model_id: 404 if model_id == settings.gemini_model else 200,
    )
    with caplog.at_level("WARNING"):
        models = _build_models()
    assert constructed == [settings.groq_model]  # gemini skipped, groq kept
    assert len(models) == 1
    assert "404s on generation" in caplog.text


def test_every_provider_gets_a_finite_timeout_and_owns_no_retries(monkeypatch):
    """A provider that is merely SLOW must not block forever.

    Without a per-request timeout no exception is raised, so the transient
    predicates never fire and the fallback chain is unreachable — measured
    2026-09-02, one light call returned 200 OK after 926s and never failed
    over. The timeout is what makes _compose's fallbacks reachable at all.

    max_retries=0 keeps retrying in ONE place: a client-side retry count
    multiplies inside a single invoke, before our predicates see anything.
    """
    seen: list[dict] = []

    class Recorder(RunnableLambda):
        def __init__(self, model, **kwargs):
            seen.append({"model": model, **kwargs})
            super().__init__(lambda _: f"ok:{model}")

        def with_structured_output(self, schema):
            return self

    for name in ("ChatGoogleGenerativeAI", "ChatGroq", "ChatOpenAI"):
        monkeypatch.setattr(model_layer, name, Recorder)
    monkeypatch.setattr(model_layer, "_probe_status", lambda provider, model_id: 200)
    monkeypatch.setattr(settings, "google_api_key", "k")
    monkeypatch.setattr(settings, "groq_api_key", "k")
    monkeypatch.setattr(settings, "cloudflare_api_token", "k")
    monkeypatch.setattr(settings, "cloudflare_account_id", "acct")
    monkeypatch.setattr(settings, "workers_ai_model", "wa-model")
    monkeypatch.setattr(settings, "model_timeout", 12.5)  # distinctive, not the default

    model_layer._build_models("heavy")

    assert len(seen) == 3, f"expected all three providers, got {[s['model'] for s in seen]}"
    for kwargs in seen:
        # Providers spell it differently; exactly one spelling must be present.
        timeout = kwargs.get("timeout", kwargs.get("request_timeout"))
        assert timeout == 12.5, f"{kwargs['model']} did not receive the configured timeout"
        assert kwargs.get("max_retries") == 0, f"{kwargs['model']} kept client-side retries"
