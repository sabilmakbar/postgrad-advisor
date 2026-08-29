"""Web chat server (story 006): SSE assembly and error boundary over a mocked
graph, no live API. The full journey against the live graph is scripts/
e2e_scripted.py (SC-001/SC-004), which burns quota and stays out of pytest.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

import app.server as server
from app.model_layer import AllProvidersExhausted


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

    def _make(graph):
        monkeypatch.setattr(server, "GRAPH", graph)
        return TestClient(server.app)

    return _make


def _start(client):
    return client.post("/chat", json={"message": ""})


def test_stream_tokens_then_authoritative_messages(client):
    """AC 2: prose chunks stream as token events, structured-output chunks are
    filtered, and the turn ends with the real messages + pending interrupt."""
    events = [
        ("messages", (AIMessageChunk(content="Wel"), {"langgraph_node": "interviewer"})),
        ("messages", (AIMessageChunk(content="come"), {"langgraph_node": "interviewer"})),
        (  # slot-parser run: marked structured, its JSON must not reach the page
            "messages",
            (
                AIMessageChunk(content='{"degree_intent":'),
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
    assert '{"degree_intent":' not in body  # opposite case: plain chunks got through
    assert "event: message\ndata: Welcome\n" in body
    assert "event: message\ndata: Send your CV\n" in body  # interrupt not in messages
    assert body.rstrip().endswith("event: done\ndata:")  # next pending: not "over"


def test_exhausted_providers_become_friendly_sse_error(client):
    """AC 4: AllProvidersExhausted -> plain-language quota message, no trace."""
    graph = FakeGraph(exc=AllProvidersExhausted("boom"))
    response = _start(client(graph))

    assert response.status_code == 200
    assert "event: error" in response.text
    assert server.QUOTA_MESSAGE in response.text
    assert "Traceback" not in response.text and "boom" not in response.text


def test_message_without_session_is_rejected(client):
    response = client(FakeGraph()).post("/chat", json={"message": "hello"})
    assert response.status_code == 400
