"""FastAPI web chat over the full graph (story 006, FR-009).

One page, one local user (AD-1): ``GET /`` serves the static chat page,
``POST /chat`` runs one graph turn and streams it as SSE, ``POST /upload``
saves a CV under data/cv/ (gitignored) and returns the path the page then
feeds into story 005's intake as a normal chat message.

Run: ``uv run uvicorn app.server:app``.
"""

import logging
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessageChunk
from langgraph.types import Command

from app.graph import build_graph
from app.model_layer import AllProvidersExhausted

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CV_DIR = ROOT / "data" / "cv"

# FR-008 acceptance wording: plain language, never a stack trace (AC 4).
QUOTA_MESSAGE = (
    "I'm out of free model quota for now — try again tomorrow, when the "
    "free tiers reset."
)

app = FastAPI()
GRAPH = build_graph()
# ponytail: in-memory sessions, move to persistent store when multi-user (AD-1)
_sessions: set[str] = set()  # started graph thread ids; the cookie is the key


def _sse(event: str, data: str) -> str:
    lines = "".join(f"data: {line}\n" for line in data.split("\n"))
    return f"event: {event}\n{lines}\n"


async def _turn(payload, config: dict, seen: int):
    """SSE events for one graph turn: live tokens while the heavy chat
    generates (AC 2), then the authoritative new messages — code-rendered
    texts and the decline prefix never stream — then done.

    Structured-output calls (slot parser, shortlist picks, fit verdicts) also
    emit chunks under messages mode; their runs carry the
    ``ls_structured_output_format`` metadata marker, so their raw JSON/tool
    tokens are filtered out of the page stream.
    """
    interrupts: list[str] = []
    try:
        async for mode, data in GRAPH.astream(
            payload, config, stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                chunk, meta = data
                if (
                    isinstance(chunk, AIMessageChunk)
                    and chunk.text
                    and "ls_structured_output_format" not in meta
                ):
                    yield _sse("token", chunk.text)
            elif "__interrupt__" in data:
                interrupts += [i.value for i in data["__interrupt__"]]
    except AllProvidersExhausted:
        yield _sse("error", QUOTA_MESSAGE)
        return
    except Exception:  # AC 4: the page never sees a stack trace
        logger.exception("graph turn failed")
        yield _sse("error", "Something went wrong on the server — check its logs.")
        return
    snapshot = GRAPH.get_state(config)
    new = [t for role, t in snapshot.values.get("messages", [])[seen:] if role == "ai"]
    for text in new:
        yield _sse("message", text)
    for value in interrupts:  # a pause whose prompt isn't in messages yet (CV ask)
        if value not in new:
            yield _sse("message", value)
    yield _sse("done", "" if snapshot.next else "over")


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/chat")
async def chat(request: Request):
    message = (await request.json()).get("message", "")
    sid = request.cookies.get("session", "")
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
    response = StreamingResponse(
        _turn(payload, config, seen), media_type="text/event-stream"
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
