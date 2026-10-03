"""Content-free local operational tracing (AD-9)."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from uuid import uuid4

from app.config import settings

logger = logging.getLogger(__name__)
_salt = secrets.token_bytes(32)


@dataclass
class TraceContext:
    conversation_id: str
    request_id: str
    warning_emitted: bool = False


@dataclass
class ModelOperation:
    operation: str
    tier: str
    logical_operation_id: str = field(default_factory=lambda: uuid4().hex)
    tier_operation_id: str = field(default_factory=lambda: uuid4().hex)
    cause: str | None = None


_context: ContextVar[TraceContext | None] = ContextVar("trace_context", default=None)
_model_operation: ContextVar[ModelOperation | None] = ContextVar(
    "trace_model_operation", default=None
)
_attempts: dict[str, list[dict]] = {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def operation_id() -> str:
    return uuid4().hex


def opaque_conversation_id(cookie: str) -> str:
    return hashlib.sha256(_salt + cookie.encode()).hexdigest()


@contextmanager
def request_context(cookie: str):
    token = _context.set(TraceContext(opaque_conversation_id(cookie), uuid4().hex))
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def model_operation(operation: ModelOperation):
    token = _model_operation.set(operation)
    try:
        yield
    finally:
        _model_operation.reset(token)


def current_context() -> TraceContext | None:
    return _context.get()


def current_model_operation() -> ModelOperation | None:
    return _model_operation.get()


def write(event: dict, context: TraceContext | None = None) -> bool:
    context = context or current_context()
    if not settings.trace_enabled or context is None:
        return False
    record = {
        "timestamp": now(),
        "conversation_id": context.conversation_id,
        "request_id": context.request_id,
        **event,
    }
    try:
        path = Path(settings.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(record, separators=(",", ":")) + "\n")
    except (OSError, TypeError, ValueError):
        if not context.warning_emitted:
            logger.warning("operational trace write failed")
            context.warning_emitted = True
        return False
    if record.get("event") == "provider_attempt":
        _attempts.setdefault(context.conversation_id, []).append(record)
    return True


def conversation_summary() -> None:
    context = current_context()
    if context is None:
        return
    attempts = _attempts.pop(context.conversation_id, [])
    write(
        {
            "event": "conversation_summary",
            "logical_operations": len(
                {event["logical_operation_id"] for event in attempts}
            ),
            "provider_attempts": len(attempts),
            "retries": sum(event.get("cause") == "retry" for event in attempts),
            "fallbacks": sum(event.get("cause") == "fallback" for event in attempts),
            "degradations": sum(
                event.get("cause") == "degradation" for event in attempts
            ),
        }
    )


def elapsed(started: float) -> float:
    return round((monotonic() - started) * 1000, 3)
