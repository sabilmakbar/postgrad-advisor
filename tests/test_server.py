"""Web chat server (story 006): SSE assembly and error boundary over a mocked
graph, no live API. The full journey against the live graph is scripts/
e2e_scripted.py (SC-001/SC-004), which burns quota and stays out of pytest.
"""

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableLambda

import app.fit as fit
import app.server as server
from app.fit import FitVerdict
from app.model_layer import AllProvidersExhausted
from app.requirements_search import SearchResult
from app.shortlist import humanize_degree_text


class FakeGraph:
    """Replays scripted astream events; get_state returns the pre-turn state
    first (the server counts already-seen messages), then the post-turn one."""

    def __init__(self, events=(), states=(), exc=None):
        self.events = events
        self.states = list(states) or [SimpleNamespace(values={}, next=())]
        self.exc = exc

    async def astream(self, payload, config, stream_mode):
        for event in self.events:
            yield event
        if self.exc:
            raise self.exc

    def get_state(self, config):
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "_sessions", set())
    monkeypatch.setattr(server, "_active_turns", {})

    def _make(graph):
        monkeypatch.setattr(server, "GRAPH", graph)
        return TestClient(server.app)

    return _make


def _start(client):
    return client.post(
        "/chat",
        json={"message": ""},
        headers={"X-Postgrad-Error-Format": "json-v1"},
    )


def _token_text(body):
    pieces = []
    for block in body.split("\n\n"):
        lines = block.splitlines()
        if lines[:1] == ["event: token"]:
            pieces.append("\n".join(line[6:] for line in lines[1:] if line.startswith("data: ")))
    return "".join(pieces)


def _token_pieces(body):
    pieces = []
    for block in body.split("\n\n"):
        lines = block.splitlines()
        if lines[:1] == ["event: token"]:
            pieces.append("\n".join(line[6:] for line in lines[1:] if line.startswith("data: ")))
    return pieces


def test_stream_tokens_then_authoritative_messages(client):
    """AC 2: prose chunks stream as token events, structured-output chunks are
    filtered, and the turn ends with the real messages + pending interrupt."""
    events = [
        (
            "messages",
            (AIMessageChunk(content="Wel", id="welcome"), {"langgraph_node": "interviewer"}),
        ),
        (
            "messages",
            (
                AIMessageChunk(content="come", id="welcome", chunk_position="last"),
                {"langgraph_node": "interviewer"},
            ),
        ),
        (  # slot-parser run: marked structured, its JSON must not reach the page
            "messages",
            (
                AIMessageChunk(
                    content='{"degree_intent":', id="parser", chunk_position="last"
                ),
                {"langgraph_node": "interviewer", "ls_structured_output_format": {}},
            ),
        ),
        ("updates", {"__interrupt__": (SimpleNamespace(value="Send your CV"),)}),
    ]
    states = [
        SimpleNamespace(values={"messages": []}, next=("interviewer",)),
        SimpleNamespace(
            values={"messages": [("human", "hi"), ("ai", "Welcome")]},
            next=("cv_intake",),
        ),
    ]
    body = _start(client(FakeGraph(events, states))).text

    assert "event: token\ndata: Wel\n" in body
    assert "event: token\ndata: come\n" in body
    assert _token_text(body) == "Welcome"
    assert '{"degree_intent":' not in body  # opposite case: plain chunks got through
    assert "event: message\ndata: Welcome\n" in body
    assert "event: message\ndata: Send your CV\n" in body  # interrupt not in messages
    assert body.rstrip().endswith("event: done\ndata:")  # next pending: not "over"


def test_stream_humanizes_whole_and_every_split_degree_token(client):
    tokens = ("masters_coursework", "masters_course", "masters_research", "phd")
    events = []
    raw = ""
    for token in tokens:
        phrase = f"whole {token}. "
        raw += phrase
        events.append(("messages", (AIMessageChunk(content=phrase, id="degree-prose"), {})))
        for split in range(1, len(token)):
            left, right = f"split {token[:split]}", f"{token[split:]}. "
            raw += left + right
            events.extend(
                [
                    ("messages", (AIMessageChunk(content=left, id="degree-prose"), {})),
                    ("messages", (AIMessageChunk(content=right, id="degree-prose"), {})),
                ]
            )
    events[-1][1][0].chunk_position = "last"
    states = [
        SimpleNamespace(values={"messages": []}, next=("interviewer",)),
        SimpleNamespace(values={"messages": []}, next=("interviewer",)),
    ]
    body = _start(client(FakeGraph(events, states))).text
    streamed = _token_text(body)

    assert streamed == humanize_degree_text(raw)
    assert all(token not in streamed.lower() for token in tokens if token != "phd")
    assert "phd" not in streamed


def test_fit_continuation_prompt_streams_once_and_keeps_session_open(client):
    states = [
        SimpleNamespace(values={"messages": []}, next=("fit_batch",)),
        SimpleNamespace(
            values={"messages": [("ai", "Assess the next batch?")]},
            next=("fit_more_decide",),
        ),
    ]

    body = _start(client(FakeGraph((), states))).text

    assert body.count("event: message\ndata: Assess the next batch?\n") == 1
    assert body.rstrip().endswith("event: done\ndata:")


def test_stream_humanizes_split_token_within_one_model_message(client):
    events = [
        ("messages", (AIMessageChunk(content="masters_", id="one-message"), {})),
        (
            "messages",
            (
                AIMessageChunk(
                    content="research", id="one-message", chunk_position="last"
                ),
                {},
            ),
        ),
    ]
    states = [
        SimpleNamespace(values={"messages": []}, next=("interviewer",)),
        SimpleNamespace(values={"messages": []}, next=("interviewer",)),
    ]

    body = _start(client(FakeGraph(events, states))).text

    assert _token_text(body) == "research master's"


def test_stream_does_not_combine_degree_token_across_model_messages(client):
    events = [
        (
            "messages",
            (AIMessageChunk(content="masters_", id="message-a", chunk_position="last"), {}),
        ),
        (
            "messages",
            (AIMessageChunk(content="research", id="message-b", chunk_position="last"), {}),
        ),
    ]
    states = [
        SimpleNamespace(values={"messages": []}, next=("interviewer",)),
        SimpleNamespace(values={"messages": []}, next=("interviewer",)),
    ]
    body = _start(client(FakeGraph(events, states))).text

    assert _token_pieces(body) == ["masters_", "research"]
    assert "research master's" not in body


def test_parser_json_in_state_never_replays_to_the_page(client):
    """Fix round: a message whose content is bare internal-schema JSON must be
    dropped from the authoritative replay; the real question (paired positive)
    survives."""
    states = [
        SimpleNamespace(values={"messages": []}, next=("interviewer",)),
        SimpleNamespace(
            values={
                "messages": [
                    ("ai", '{"degree_intent": "masters_coursework"}'),
                    ("ai", "Which field would you like to study?"),
                ]
            },
            next=("interviewer",),
        ),
    ]
    body = _start(client(FakeGraph((), states))).text

    assert '{"degree_intent"' not in body  # seeded parser message dropped
    assert "event: message\ndata: Which field would you like to study?\n" in body


def test_exhausted_providers_become_friendly_sse_error(client):
    """AC 4: AllProvidersExhausted -> typed JSON, no trace."""
    graph = FakeGraph(exc=AllProvidersExhausted("boom", "quota"))
    response = _start(client(graph))

    assert response.status_code == 200
    assert "event: error" in response.text
    payload = response.text.split("data: ", 1)[1].strip()
    assert json.loads(payload) == {"kind": "quota", "message": server.QUOTA_MESSAGE}
    assert "\\u2014" in payload  # json.dumps escaping is preserved on the wire
    assert "Traceback" not in response.text and "boom" not in response.text


@pytest.mark.parametrize("kind", server.ERROR_MESSAGES)
def test_every_error_kind_keeps_its_wire_identity(client, kind):
    response = _start(client(FakeGraph(exc=AllProvidersExhausted("private", kind))))
    payload = json.loads(response.text.split("data: ", 1)[1])
    assert payload == {"kind": kind, "message": server.ERROR_MESSAGES[kind]}


def test_unknown_provider_kind_normalizes_to_unknown_json(client):
    response = _start(client(FakeGraph(exc=AllProvidersExhausted("private", "new-kind"))))
    payload = json.loads(response.text.split("data: ", 1)[1])
    assert payload == {"kind": "unknown", "message": server.ERROR_MESSAGES["unknown"]}


@pytest.mark.parametrize("header", [None, "", "json-v2"])
def test_stale_pages_receive_plain_error_text(client, header):
    headers = {} if header is None else {"X-Postgrad-Error-Format": header}
    response = client(FakeGraph(exc=AllProvidersExhausted("private", "quota"))).post(
        "/chat", json={"message": ""}, headers=headers
    )
    payload = response.text.split("data: ", 1)[1].strip()
    assert payload == server.ERROR_MESSAGES["quota"]
    assert not payload.startswith("{")


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"X-Postgrad-Error-Format": "json-v1"}, "json"),
        ({}, "plain"),
    ],
)
def test_generic_exception_uses_negotiated_error_format(client, headers, expected):
    response = client(FakeGraph(exc=RuntimeError("private"))).post(
        "/chat", json={"message": ""}, headers=headers
    )
    payload = response.text.split("data: ", 1)[1].strip()
    if expected == "json":
        assert json.loads(payload) == {
            "kind": "unknown",
            "message": server.ERROR_MESSAGES["unknown"],
        }
    else:
        assert payload == server.ERROR_MESSAGES["unknown"]
    assert "private" not in response.text


def test_task_and_custom_statuses_are_ordered_and_not_message_text(client):
    events = [
        ("tasks", {"name": "shortlister"}),
        ("custom", "searching requirements: Test U (1/2)"),
        ("tasks", {"name": "fit_batch", "result": {}}),  # completion is not a start
        ("messages", (AIMessageChunk(content="answer", id="a", chunk_position="last"), {})),
    ]
    states = [
        SimpleNamespace(values={"messages": []}, next=("shortlister",)),
        SimpleNamespace(values={"messages": [],}, next=("fit_batch",)),
    ]
    body = _start(client(FakeGraph(events, states))).text

    first = "event: status\ndata: filtering universities…"
    second = "event: status\ndata: searching requirements: Test U (1/2)"
    assert body.index(first) < body.index(second) < body.index("event: token")
    assert "comparing your profile" not in body
    assert _token_text(body) == "answer"
    assert "filtering universities" not in _token_text(body)


def test_fit_emits_one_progress_status_per_program(monkeypatch):
    statuses = []
    monkeypatch.setattr(fit, "get_stream_writer", lambda: statuses.append)
    monkeypatch.setattr(
        fit,
        "search_requirements",
        lambda query: SearchResult(query=query, text="requirements", sources=["https://example.test"]),
    )
    monkeypatch.setattr(
        fit,
        "structured",
        lambda schema, tier: RunnableLambda(
            lambda _: FitVerdict(
                label="strong",
                reasons=["matches"],
                sources=["https://example.test"],
                sufficient=True,
            )
        ),
    )
    facts = SimpleNamespace(
        degree="BSc", gpa="3.8", publications=[], english_test=None,
        experience=[], experience_months=None,
    )
    entries = [
        SimpleNamespace(university="One U", country="X", rank_context="rank 1", degree_type="phd"),
        SimpleNamespace(university="Two U", country="Y", rank_context="rank 2", degree_type="phd"),
    ]
    state = SimpleNamespace(
        cv_facts=facts, searches_used=0, additional_searches=0, shortlist=entries, field="CS", messages=[]
    )

    fit.fit_node(state)

    assert statuses == [
        "searching requirements: One U (1/2)",
        "searching requirements: Two U (2/2)",
    ]


def test_active_session_stays_owned_until_producer_finishes(client):
    web = client(FakeGraph())
    assert _start(web).status_code == 200  # establish session cookie

    started = threading.Event()
    release = threading.Event()

    class BlockingGraph(FakeGraph):
        def __init__(self):
            super().__init__()
            self.invocations = 0

        async def astream(self, payload, config, stream_mode):
            self.invocations += 1
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            if False:
                yield None

    graph = BlockingGraph()
    server.GRAPH = graph
    first = threading.Thread(target=lambda: web.post("/chat", json={"message": "first"}))
    first.start()
    assert started.wait(1)

    rejected = web.post("/chat", json={"message": "retry"})
    assert rejected.status_code == 409
    assert rejected.json()["detail"] == server.TURN_ACTIVE_MESSAGE
    assert graph.invocations == 1

    release.set()
    first.join(2)
    assert not first.is_alive()
    assert web.post("/chat", json={"message": "retry"}).status_code == 200
    assert graph.invocations == 2


def test_closing_response_consumer_does_not_release_producer_ownership(monkeypatch):
    async def scenario():
        release = asyncio.Event()

        class DisconnectGraph(FakeGraph):
            async def astream(self, payload, config, stream_mode):
                yield "custom", "started"
                await release.wait()

        monkeypatch.setattr(server, "GRAPH", DisconnectGraph())
        monkeypatch.setattr(server, "_active_turns", {})
        sid = "held-session"
        config = {"configurable": {"thread_id": sid}}
        consumer = server._queued_turn(sid, {}, config, 0, True)
        assert await anext(consumer) == "event: status\ndata: started\n\n"
        producer = server._active_turns[sid]

        await consumer.aclose()  # equivalent to the response consumer disconnecting
        assert server._active_turns[sid] is producer
        assert not producer.done()

        release.set()
        await producer
        assert sid not in server._active_turns

    asyncio.run(scenario())


def test_page_has_exact_silence_lifecycle_and_legacy_error_fallback():
    page = (Path(__file__).parents[1] / "static" / "index.html").read_text()

    assert "}, 75000);" in page
    assert 'signal: controller.signal' in page and "controller.abort();" in page
    assert '["token", "message", "status"].includes(event)' in page
    assert 'event === "error"' in page and 'event === "done"' in page
    assert 'state.timedOut && e.name === "AbortError"' in page
    assert "generation !== streamGeneration" in page
    assert 'let error = { kind: "unknown", message: text };' in page
    assert '"X-Postgrad-Error-Format": "json-v1"' in page
    assert "This turn is taking unusually long." in page


def test_page_keeps_turn_status_transient():
    page = (Path(__file__).parents[1] / "static" / "index.html").read_text()
    painter = page.split("function startPainter(state) {", 1)[1].split("function finish(state) {", 1)[0]
    finish = page.split("function finish(state) {", 1)[1].split("const OFFLINE", 1)[0]

    assert """      if (state.dropPh) {
        state.dropPh();
        state.dropPh = null;
        state.traceEl.remove();
      }""" in painter
    assert finish.splitlines().count("  state.traceEl.remove();") == 1
    assert 'add("trace",' not in page
    assert 'event === "status"' in page
    assert 'state.traceEl.textContent = state.trace.join(" › ");' in page
    assert 'state.finals.some(([role]) => role === "error")' in finish
    assert "state.timedOut = true;" in page and "controller.abort();" in page


def test_user_text_selection_contrasts_with_the_bubble_in_both_themes():
    page = (Path(__file__).parents[1] / "static" / "index.html").read_text()

    assert "--user-selection-bg: #fef3c7;" in page
    assert "--user-selection-fg: #18181b;" in page
    assert "--user-selection-bg: #164e63;" in page
    assert "--user-selection-fg: #f0f9ff;" in page
    assert ".user .bubble::selection" in page


def test_message_without_session_is_rejected(client):
    response = client(FakeGraph()).post("/chat", json={"message": "hello"})
    assert response.status_code == 400
