"""Live smoke test: `uv run python -m app.smoke [--tier light|heavy] [--check]`.

Default: one prompt through the chosen tier's chain, printing tier + provider.
`--check`: probe generation access for every configured model id, then exit.

Deliberately outside tests/ so pytest never collects it (needs real API keys).
"""

import argparse

from app.config import settings
from app.model_layer import _probe_status, chat


def _provider(model_name: str) -> str:
    for name, ids in (
        ("Gemini", (settings.gemini_model, settings.gemini_light_model)),
        ("Groq", (settings.groq_model, settings.groq_light_model)),
        ("Workers AI", (settings.workers_ai_model, settings.workers_ai_light_model)),
    ):
        if any(i and i in model_name for i in ids):
            return name
    return f"unknown ({model_name or 'no model_name in metadata'})"


def _check() -> None:
    """Generation-access probe per configured id — list presence is not checked
    because it lies (a retired id can stay listed while 404ing on generation)."""
    checked = False
    for provider, key_ok, heavy_id, light_id in (
        ("google", settings.google_api_key, settings.gemini_model, settings.gemini_light_model),
        ("groq", settings.groq_api_key, settings.groq_model, settings.groq_light_model),
        (
            "workers_ai",
            settings.cloudflare_api_token and settings.cloudflare_account_id,
            settings.workers_ai_model,
            settings.workers_ai_light_model,
        ),
    ):
        if not key_ok:
            continue
        for tier, model_id in (("heavy", heavy_id), ("light", light_id)):
            if not model_id:
                continue
            checked = True
            status = _probe_status(provider, model_id)
            if status == 404:
                verdict = "DEAD (404 on generation)"
            elif status == 0:
                verdict = "unreachable (network error)"
            else:
                verdict = f"ok (HTTP {status})"
            print(f"{provider} {tier} {model_id}: {verdict}")
    if not checked:
        print("no provider configured (no API keys in .env)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("heavy", "light"), default="heavy")
    parser.add_argument(
        "--check",
        action="store_true",
        help="probe generation access for every configured model id, then exit",
    )
    args = parser.parse_args()
    if args.check:
        _check()
        return
    response = chat(tier=args.tier).invoke("Reply with exactly one word: ready")
    model_name = response.response_metadata.get("model_name", "")
    print(f"tier: {args.tier}")
    print(f"provider answered: {_provider(model_name)}")
    print(f"response: {response.content}")


if __name__ == "__main__":
    main()
