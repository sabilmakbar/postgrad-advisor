"""FastAPI web chat over the full graph (story 006, FR-009).

One page, one local user (AD-1): ``GET /`` serves the static chat page,
``POST /chat`` runs one graph turn and streams it as SSE, ``POST /upload``
saves a CV under data/cv/ (gitignored) and returns the path the page holds as
an attachment chip until the user's next Send (story 009).

Run: ``uv run uvicorn app.server:app``.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from time import monotonic
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessageChunk
from langgraph.types import Command

from app import tracing
from app.config import CV_DIR, ROOT
from app.graph import build_graph
from app.model_layer import AllProvidersExhausted
from app.shortlist import CANONICAL_DEGREE_TOKENS, humanize_degree_text

logger = logging.getLogger(__name__)

# SSE contract: token/message/status carry plain text; error carries JSON for
# clients that request json-v1 and mapped plain text otherwise; done carries
# "over" or an empty string.
ERROR_MESSAGES = {
    "slow": "The model provider is not responding. Please try again in a moment.",
    "offline": "I could not reach the model provider. Check your connection and try again.",
    "quota": "I'm out of free model quota for now — try again tomorrow, when the free tiers reset.",
    "unconfigured": "No model provider is configured. Add its credentials and model ID to .env.",
    "model_unavailable": "The configured model IDs are unavailable. Run the smoke check, then try again.",
    "unknown": "Something went wrong on the server — check its logs.",
}
QUOTA_MESSAGE = ERROR_MESSAGES["quota"]
TURN_ACTIVE_MESSAGE = "The previous turn is still finishing. Please try again shortly."

NODE_STATUS = {
    "interviewer": "working through your answer…",
    "shortlister": "filtering universities…",
    "cv_ask": "getting ready for your CV…",
    "cv_intake": "reading your CV…",
    "cv_confirm": "checking your CV details…",
    "fit_batch": "comparing your profile with each program…",
}

app = FastAPI()
GRAPH = build_graph()
# ponytail: in-memory sessions, move to persistent store when multi-user (AD-1)
_sessions: set[str] = set()  # started graph thread ids; the cookie is the key
_active_turns: dict[str, asyncio.Task] = {}


def _sse(event: str, data: str) -> str:
    lines = "".join(f"data: {line}\n" for line in data.split("\n"))
    return f"event: {event}\n{lines}\n"


def _internal_json(text: str) -> bool:
    """True for structured-output shapes: a bare JSON object. That covers every
    internal schema (SlotUpdate, CVFacts, picks, verdicts) — the agent's real
    messages are prose and never a bare JSON object. Belt-and-braces with the
    model layer's ``nostream`` tag: parser output must never reach the page
    (story 006 fix round)."""
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


class _DegreeTokenStream:
    """Humanize canonical degree tokens without exposing split prefixes."""

    def __init__(self):
        self.pending = ""
        self.left_word = False

    def _clean(self, text: str) -> str:
        prefix = "x" if self.left_word else ""
        cleaned = humanize_degree_text(prefix + text)[len(prefix) :]
        if text:
            self.left_word = bool(re.match(r"\w", text[-1]))
        return cleaned

    def feed(self, text: str) -> str:
        self.pending += text
        lower = self.pending.lower()
        hold = len(self.pending)
        start_at = max(0, len(self.pending) - len(CANONICAL_DEGREE_TOKENS[0]))
        for start in range(start_at, len(self.pending)):
            left_word = (
                self.left_word
                if start == 0
                else bool(re.match(r"\w", self.pending[start - 1]))
            )
            if left_word:
                continue
            suffix = lower[start:]
            if any(token.startswith(suffix) for token in CANONICAL_DEGREE_TOKENS):
                hold = start
                break
        ready, self.pending = self.pending[:hold], self.pending[hold:]
        return self._clean(ready)

    def finish(self) -> str:
        ready, self.pending = self.pending, ""
        return self._clean(ready)


def _error_payload(kind: str, json_errors: bool) -> str:
    message = ERROR_MESSAGES[kind]
    return json.dumps({"kind": kind, "message": message}) if json_errors else message


async def _turn(payload, config: dict, seen: int, json_errors: bool):
    """SSE events for one graph turn: live tokens while the heavy chat
    generates (AC 2), then the authoritative new messages — code-rendered
    texts and the decline prefix never stream — then done.

    Structured-output calls (slot parser, shortlist picks, fit verdicts) are
    kept off the page three ways (story 006 fix round): the model layer tags
    them ``nostream`` so messages mode skips them at the source; any chunk that
    still carries the ``ls_structured_output_format`` metadata marker is
    filtered here; and the authoritative replay drops anything that parses as
    internal-schema JSON (``_internal_json``).
    """
    interrupts: list[str] = []
    degree_streams: dict[str, _DegreeTokenStream] = {}
    active_message: str | None = None
    nodes: dict[str, tuple[float, str, str]] = {}

    def flush(message_id: str | None = None) -> list[str]:
        ids = [message_id] if message_id is not None else list(degree_streams)
        tails = []
        for stream_id in ids:
            if stream := degree_streams.pop(stream_id, None):
                if tail := stream.finish():
                    tails.append(tail)
        return tails

    def fail_open_nodes() -> None:
        for task_id, (began, node_id, name) in list(nodes.items()):
            tracing.write(
                {
                    "event": "graph_node",
                    "node_operation_id": node_id,
                    "node": name,
                    "phase": "finish",
                    "duration_ms": tracing.elapsed(began),
                    "outcome": "failure",
                }
            )
            nodes.pop(task_id)

    try:
        async for mode, data in GRAPH.astream(
            payload, config, stream_mode=["messages", "updates", "tasks", "custom"]
        ):
            if mode == "messages":
                chunk, meta = data
                if not isinstance(chunk, AIMessageChunk):
                    continue
                message_id = chunk.id
                if active_message is not None and message_id != active_message:
                    for tail in flush(active_message):
                        yield _sse("token", tail)
                active_message = message_id
                if chunk.text and "ls_structured_output_format" not in meta:
                    # LangGraph's v1 messages stream preserves the model-run
                    # chunk id. Missing ids are isolated rather than shared.
                    stream = degree_streams.setdefault(
                        message_id or f"chunk-{id(chunk)}", _DegreeTokenStream()
                    )
                    text = stream.feed(chunk.text)
                    if text:
                        yield _sse("token", text)
                    if chunk.chunk_position == "last":
                        for tail in flush(message_id or f"chunk-{id(chunk)}"):
                            yield _sse("token", tail)
                        active_message = None
                elif chunk.chunk_position == "last":
                    active_message = None
            elif mode == "updates":
                for tail in flush():
                    yield _sse("token", tail)
                active_message = None
                if "__interrupt__" in data:
                    interrupts += [i.value for i in data["__interrupt__"]]
            elif mode == "tasks":
                if isinstance(data, dict) and (name := data.get("name")):
                    task_id = str(data.get("id") or name)
                    if "result" not in data and "error" not in data:
                        node_id = tracing.operation_id()
                        nodes[task_id] = (monotonic(), node_id, name)
                        tracing.write(
                            {
                                "event": "graph_node",
                                "node_operation_id": node_id,
                                "node": name,
                                "phase": "start",
                                "duration_ms": None,
                                "outcome": None,
                            }
                        )
                    elif started := nodes.pop(task_id, None):
                        began, node_id, _ = started
                        tracing.write(
                            {
                                "event": "graph_node",
                                "node_operation_id": node_id,
                                "node": name,
                                "phase": "finish",
                                "duration_ms": tracing.elapsed(began),
                                "outcome": (
                                    "failure" if "error" in data else "success"
                                ),
                            }
                        )
                if (
                    isinstance(data, dict)
                    and "result" not in data
                    and "error" not in data
                    and (status := NODE_STATUS.get(data.get("name")))
                ):
                    yield _sse("status", status)
            elif mode == "custom" and isinstance(data, str):
                yield _sse("status", data)
        for tail in flush():
            yield _sse("token", tail)
    except AllProvidersExhausted as exc:
        fail_open_nodes()
        for tail in flush():
            yield _sse("token", tail)
        kind = exc.kind if exc.kind in ERROR_MESSAGES else "unknown"
        yield _sse("error", _error_payload(kind, json_errors))
        return
    except Exception:  # AC 4: neither page nor logs retain candidate content
        fail_open_nodes()
        for tail in flush():
            yield _sse("token", tail)
        logger.warning("graph turn failed")
        yield _sse("error", _error_payload("unknown", json_errors))
        return
    snapshot = GRAPH.get_state(config)
    new = [t for role, t in snapshot.values.get("messages", [])[seen:] if role == "ai"]
    for text in new:
        if _internal_json(text):  # parser output never replays to the user
            logger.warning("dropped internal JSON from the chat replay")
            continue
        yield _sse("message", text)
    for value in interrupts:  # a pause whose prompt isn't in messages yet (CV ask)
        if value not in new and not _internal_json(value):
            yield _sse("message", value)
    if not snapshot.next:
        tracing.conversation_summary()
    yield _sse("done", "" if snapshot.next else "over")


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


def _queued_turn(sid: str, payload, config: dict, seen: int, json_errors: bool):
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def produce():
        with tracing.request_context(sid):
            try:
                async for event in _turn(payload, config, seen, json_errors):
                    await queue.put(event)
            finally:
                _active_turns.pop(sid, None)
                await queue.put(None)

    task = asyncio.create_task(produce())
    _active_turns[sid] = task

    async def consume():
        while (event := await queue.get()) is not None:
            yield event

    return consume()


@app.post("/chat")
async def chat(request: Request):
    message = (await request.json()).get("message", "")
    sid = request.cookies.get("session", "")
    if sid and sid in _active_turns:
        raise HTTPException(409, TURN_ACTIVE_MESSAGE)
    if message and sid in _sessions:
        payload: dict | Command = Command(resume=message)
    elif message:
        raise HTTPException(400, "no active session — reload the page")
    else:  # empty message = (re)start: fresh thread, opening question (AC 1)
        sid = uuid4().hex
        _sessions.add(sid)
        payload = {}
    config = {"configurable": {"thread_id": sid}}
    seen = len(GRAPH.get_state(config).values.get("messages", []))
    json_errors = request.headers.get("X-Postgrad-Error-Format") == "json-v1"
    response = StreamingResponse(
        _queued_turn(sid, payload, config, seen, json_errors), media_type="text/event-stream"
    )
    response.set_cookie("session", sid)
    return response


@app.post("/upload")
async def upload(request: Request, name: str = ""):
    """Raw-body upload (no multipart dependency): the file's bytes are the
    body, its name a query param. Saved under data/cv/ with a fresh name —
    the browser filename is untrusted and never used as a path."""
    suffix = Path(name).suffix.lower()
    if suffix not in (".pdf", ".txt"):  # FR-005 formats, checked at the boundary
        raise HTTPException(400, "only .pdf or .txt CVs are supported")
    CV_DIR.mkdir(parents=True, exist_ok=True)
    dest = CV_DIR / f"{uuid4().hex}{suffix}"
    dest.write_bytes(await request.body())
    return {"path": str(dest)}
