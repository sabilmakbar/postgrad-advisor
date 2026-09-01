"""Settings from .env — keys, model ids, RPMs are config, never literals (AD-2).

Also the one home for the upload directory constant (story 009): the server
writes uploads there and the graph decides "is this message an upload?" from it.
It lives here because this module imports nothing app-internal — importing it
from app/server.py instead would close the circle server -> graph -> cv -> server.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
CV_DIR = ROOT / "data" / "cv"  # gitignored; /upload returns ABSOLUTE paths here


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str | None = None
    groq_api_key: str | None = None
    cloudflare_api_token: str | None = None
    cloudflare_account_id: str | None = None
    tavily_api_key: str | None = None  # story 005: requirements search (AD-3)

    gemini_model: str = "gemini-3.6-flash"
    groq_model: str = "openai/gpt-oss-120b"
    workers_ai_model: str = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"

    # Light tier (FR-011 / AD-7). Blank id in .env = provider disabled for the
    # light tier; all light ids blank = tier disabled (light calls degrade to
    # heavy). Defaults verified LIVE with a 1-token generation probe, 200 for
    # both (set 2026-08-28, re-verified 2026-08-29; list presence alone lies —
    # gemini-2.5-flash is still listed yet 404s on generation).
    gemini_light_model: str = "gemini-3.5-flash-lite"
    groq_light_model: str = "openai/gpt-oss-20b"
    # No Cloudflare token in .env on 2026-08-28, so no live-verified default can
    # ship (AC 2); set WORKERS_AI_LIGHT_MODEL in .env to enable.
    workers_ai_light_model: str = ""

    gemini_rpm: float = 10
    groq_rpm: float = 30

    # Optional light-tier RPM overrides; unset = reuse the provider's heavy RPM.
    gemini_light_rpm: float | None = None
    groq_light_rpm: float | None = None


settings = Settings()
