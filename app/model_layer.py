"""Free-tier model layer (AD-2): Gemini -> Groq -> Workers AI, one resilient chain.

All model calls in this project go through `chat()` or `structured(schema)`.
No provider SDK imports belong anywhere else.

Two tiers (AD-7 / FR-011): `heavy` (default) is the original chain; `light` is a
cheap-model chain for intent detection and slot-parsing. Light exhaustion (or an
unconfigured light tier) degrades to heavy with a logged notice.
"""

import logging

import google.genai.errors as google_errors
import groq
import httpx
import openai
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.constants import TAG_NOSTREAM
from langgraph.config import get_stream_writer

from app.config import settings

logger = logging.getLogger(__name__)


class AllProvidersExhausted(Exception):
    """Every provider in the chain failed after retries."""

    def __init__(self, message: str, kind: str = "unknown"):
        super().__init__(message)
        self.kind = kind


class _GoogleRateLimitMeta(type):
    """Status-code predicate for `retry_if_exception_type`: the retry check is a
    plain `isinstance`, and google.genai's ClientError covers ALL 4xx — this
    class matches only code 429."""

    def __instancecheck__(cls, exc: object) -> bool:
        return isinstance(exc, google_errors.ClientError) and exc.code == 429


class _GoogleRateLimitError(Exception, metaclass=_GoogleRateLimitMeta):
    """Never raised; exists only for the isinstance predicate above."""


# Transient = worth retrying: rate limit (429), 5xx, timeout, connection errors.
# Anything else (401, 404, ...) fails fast to the next provider.
_GOOGLE_TRANSIENT = (
    _GoogleRateLimitError,
    google_errors.ServerError,
    httpx.TransportError,  # google.genai uses httpx; covers timeouts + connection
)
_GROQ_TRANSIENT = (groq.RateLimitError, groq.InternalServerError, groq.APIConnectionError)
_OPENAI_TRANSIENT = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
)

_TIMEOUT_ERRORS = (httpx.TimeoutException, groq.APITimeoutError, openai.APITimeoutError)
_CONNECTION_ERRORS = (httpx.TransportError, groq.APIConnectionError, openai.APIConnectionError)
_RATE_LIMIT_ERRORS = (_GoogleRateLimitError, groq.RateLimitError, openai.RateLimitError)


def _error_kind(exc: BaseException) -> str:
    if isinstance(exc, _TIMEOUT_ERRORS):
        return "slow"
    if isinstance(exc, _CONNECTION_ERRORS):
        return "offline"
    if isinstance(exc, _RATE_LIMIT_ERRORS) or getattr(exc, "status_code", None) == 429:
        return "quota"
    return "unknown"


def _emit_status(text: str) -> None:
    try:
        get_stream_writer()(text)
    except Exception:
        pass


class _AttemptStatus(BaseCallbackHandler):
    """Report provider transitions without retaining state between runs."""

    def __init__(self, provider_position: int):
        self.provider_position = provider_position

    def on_chat_model_start(self, serialized, messages, *, tags=None, **kwargs):
        attempt = next(
            (
                int(tag.rsplit(":", 1)[1])
                for tag in tags or ()
                if tag.startswith("retry:attempt:")
            ),
            1,
        )
        if attempt > 1:
            _emit_status("model busy, retrying")
        elif self.provider_position > 0:
            _emit_status("model unavailable, trying the backup")

# Free-tier RPM quotas are PER MODEL: one shared limiter per (provider, model id),
# so every chain built in this process draws from the same budget.
_rate_limiters: dict[tuple[str, str], InMemoryRateLimiter] = {}

# (model, transient exception types); retries apply only to the transient types.
_ProviderModel = tuple[Runnable, tuple[type[BaseException], ...]]


def _limiter(provider: str, model: str, rpm: float) -> InMemoryRateLimiter:
    key = (provider, model)
    if key not in _rate_limiters:
        _rate_limiters[key] = InMemoryRateLimiter(requests_per_second=rpm / 60)
    return _rate_limiters[key]


# Health-probe verdict per (provider, model id), for the process lifetime.
_probe_cache: dict[tuple[str, str], bool] = {}


def _probe_status(provider: str, model_id: str) -> int:
    """HTTP status of a 1-token generation with the configured key; 0 on network
    error. Generation access is the real check — list presence lies (a retired
    id can stay on the models list while 404ing on generation)."""
    openai_style = {
        "model": model_id,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    try:
        if provider == "google":
            resp = httpx.post(
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_id}:generateContent",
                params={"key": settings.google_api_key},
                json={
                    "contents": [{"parts": [{"text": "hi"}]}],
                    "generationConfig": {"maxOutputTokens": 1},
                },
                timeout=30,
            )
        elif provider == "groq":
            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                json=openai_style,
                timeout=30,
            )
        else:  # workers_ai
            resp = httpx.post(
                "https://api.cloudflare.com/client/v4/accounts/"
                f"{settings.cloudflare_account_id}/ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.cloudflare_api_token}"},
                json=openai_style,
                timeout=30,
            )
        return resp.status_code
    except httpx.HTTPError:
        return 0


def _generation_ok(provider: str, model_id: str) -> bool:
    """Dead iff the generation probe 404s. Anything else (200/401/429/network
    trouble) keeps the id — fail open, so a flaky probe never disables a
    provider. One probe per id per process (cached)."""
    # ponytail: fail-open + process-lifetime cache; add re-probe/TTL only if a
    # provider ever starts 404ing intermittently.
    key = (provider, model_id)
    if key not in _probe_cache:
        alive = _probe_status(provider, model_id) != 404
        if not alive:
            logger.warning(
                "model id %r (%s) 404s on generation; skipping it in the chain",
                model_id,
                provider,
            )
        _probe_cache[key] = alive
    return _probe_cache[key]


def _build_models(tier: str = "heavy") -> list[_ProviderModel]:
    """(model, transient types) for each configured provider, in fallback order.

    Light tier: a blank light id disables that provider for the tier; an empty
    result means the tier is unconfigured and the caller degrades to heavy
    (AC 5) instead of raising. Ids that fail the generation probe are warned
    about and skipped (AC 2)."""
    light = tier == "light"
    models: list[_ProviderModel] = []
    configured = False
    if settings.google_api_key:
        model_id = settings.gemini_light_model if light else settings.gemini_model
        rpm = (settings.gemini_light_rpm if light else None) or settings.gemini_rpm
        configured |= bool(model_id)
        if model_id and _generation_ok("google", model_id):
            models.append(
                (
                    ChatGoogleGenerativeAI(
                        model=model_id,
                        api_key=settings.google_api_key,
                        rate_limiter=_limiter("google", model_id, rpm),
                        timeout=settings.model_timeout,
                        # This layer owns retrying (see _compose). A client-side
                        # retry count multiplies INSIDE one invoke, before our
                        # predicates see anything, so leaving it on would make
                        # the worst case provider_retries x our_attempts x
                        # providers instead of our_attempts x providers.
                        max_retries=0,
                    ),
                    _GOOGLE_TRANSIENT,
                )
            )
    if settings.groq_api_key:
        model_id = settings.groq_light_model if light else settings.groq_model
        rpm = (settings.groq_light_rpm if light else None) or settings.groq_rpm
        configured |= bool(model_id)
        if model_id and _generation_ok("groq", model_id):
            models.append(
                (
                    ChatGroq(
                        model=model_id,
                        api_key=settings.groq_api_key,
                        rate_limiter=_limiter("groq", model_id, rpm),
                        request_timeout=settings.model_timeout,  # ChatGroq's spelling
                        max_retries=0,  # this layer owns retrying
                    ),
                    _GROQ_TRANSIENT,
                )
            )
    if settings.cloudflare_api_token and settings.cloudflare_account_id:
        model_id = (
            settings.workers_ai_light_model if light else settings.workers_ai_model
        )
        configured |= bool(model_id)
        if model_id and _generation_ok("workers_ai", model_id):
            models.append(
                (
                    ChatOpenAI(
                        model=model_id,
                        api_key=settings.cloudflare_api_token,
                        base_url=(
                            "https://api.cloudflare.com/client/v4/accounts/"
                            f"{settings.cloudflare_account_id}/ai/v1"
                        ),
                        request_timeout=settings.model_timeout,  # ChatOpenAI's spelling
                        max_retries=0,  # this layer owns retrying
                    ),
                    _OPENAI_TRANSIENT,
                )
            )
    if not models:
        if light:
            return []
        if configured:
            raise AllProvidersExhausted(
                "The configured model IDs are unavailable. Run the smoke check "
                "and review its model-access results.",
                "model_unavailable",
            )
        raise AllProvidersExhausted(
            "No model provider is configured. Set an API key and enabled model "
            "ID in .env (see .env.example).",
            "unconfigured",
        )
    return models


def _compose(
    models: list[_ProviderModel], jitter_params: dict | None = None
) -> Runnable:
    """Wrap each model with retry+backoff on its transient exception types only,
    chain fallbacks, type the final error.

    `jitter_params` exists so tests can shrink the backoff waits.
    """
    retried = [
        m.with_config(callbacks=[_AttemptStatus(position)]).with_retry(
            retry_if_exception_type=transient,
            stop_after_attempt=settings.model_attempts,
            exponential_jitter_params=jitter_params,
        )
        for position, (m, transient) in enumerate(models)
    ]
    chain = retried[0].with_fallbacks(retried[1:])

    # ponytail: sync invoke only; add stream/async passthrough when the chat UI story needs it
    def _invoke(value, config: RunnableConfig | None = None):
        try:
            return chain.invoke(value, config)
        except Exception as e:
            raise AllProvidersExhausted(
                "All model providers failed after retries. Check the API keys in "
                ".env and your network connection, then try again.",
                _error_kind(e),
            ) from e

    return RunnableLambda(_invoke)


def _tier_chain(tier: str, prepare) -> Runnable:
    """Route a tier through the single `_compose` seam.

    `prepare` maps the raw model list to the composed one (identity for chat,
    per-model schema binding for structured). Heavy is the plain chain. Light
    composes only light models; heavy models are built LAZILY, only if light
    degrades — a successful light call never constructs a heavy model (AC 4).
    """
    if tier not in ("heavy", "light"):
        raise ValueError(f"unknown tier {tier!r}; use 'light' or 'heavy'")

    def heavy() -> Runnable:
        return _compose(prepare(_build_models("heavy")))

    if tier == "heavy":
        return heavy()

    light_models = _build_models("light")
    if not light_models:
        logger.warning(
            "light tier has no usable model ids; degrading to the heavy chain"
        )
        _emit_status("light model unavailable, trying the full model")
        return heavy()
    light_chain = _compose(prepare(light_models))

    def _invoke(value, config: RunnableConfig | None = None):
        try:
            return light_chain.invoke(value, config)
        except AllProvidersExhausted:
            logger.warning("light tier exhausted; degrading to the heavy chain")
            _emit_status("light model unavailable, trying the full model")
            return heavy().invoke(value, config)

    return RunnableLambda(_invoke)


def chat(tier: str = "heavy") -> Runnable:
    """Plain chat chain across the tier's configured providers."""
    return _tier_chain(tier, lambda models: models)


def structured(schema, tier: str = "heavy") -> Runnable:
    """Structured-output chain. Binds the schema per provider model BEFORE
    composing fallbacks (a composed chain has no `with_structured_output`).

    Tagged ``nostream``: langgraph's messages-mode streaming skips tagged runs,
    so raw schema JSON never enters a chat token stream (story 006 fix round —
    the providers' ``ls_structured_output_format`` marker lands in callback
    *options*, never in the run metadata that stream filters can see, so
    downstream filtering alone cannot identify these runs)."""
    return _tier_chain(
        tier, lambda models: [(m.with_structured_output(schema), t) for m, t in models]
    ).with_config(tags=[TAG_NOSTREAM])
