import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import SimpleChatModel

import app.model_layer as model_layer
import app.requirements_search as search
import app.server as server
from app import tracing
from app.config import settings


class ScriptedChat(SimpleChatModel):
    responses: list[object]
    calls: int = 0

    @property
    def _llm_type(self):
        return "scripted"

    def _call(self, messages, **kwargs):
        response = self.responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_provider_attempt_order_and_summary_are_content_free(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "trace_enabled", True)
    monkeypatch.setattr(settings, "trace_path", path)
    monkeypatch.setattr(settings, "model_attempts", 2)
    primary = ScriptedChat(
        responses=[httpx.ConnectError("CANARY_ERROR"), httpx.ConnectError("CANARY_ERROR")]
    )
    backup = ScriptedChat(responses=["CANARY_ANSWER"])

    with tracing.request_context("CANARY_COOKIE"):
        operation = tracing.ModelOperation("chat", "heavy")
        with tracing.model_operation(operation):
            chain = model_layer._compose(
                [
                    (primary, (httpx.ConnectError,)),
                    (backup, (httpx.ConnectError,)),
                ],
                {"initial": 0.001, "max": 0.005},
            )
            assert chain.invoke("CANARY_PROMPT").content == "CANARY_ANSWER"
        tracing.conversation_summary()

    events = _events(path)
    attempts = [event for event in events if event["event"] == "provider_attempt"]
    assert [(event["provider"], event["attempt_index"], event["cause"], event["outcome"]) for event in attempts] == [
        ("provider_0", 1, None, "error"),
        ("provider_0", 2, "retry", "error"),
        ("provider_1", 1, "fallback", "success"),
    ]
    assert len({event["logical_operation_id"] for event in attempts}) == 1
    assert len({event["tier_operation_id"] for event in attempts}) == 1
    assert len({event["conversation_id"] for event in events}) == 1
    assert len({event["request_id"] for event in events}) == 1
    summary = events[-1]
    assert summary | {} == summary
    assert (summary["logical_operations"], summary["provider_attempts"], summary["retries"], summary["fallbacks"]) == (1, 3, 1, 1)
    trace = path.read_text()
    assert not any(canary in trace for canary in ("CANARY_COOKIE", "CANARY_PROMPT", "CANARY_ANSWER", "CANARY_ERROR"))


def test_attempt_metadata_includes_observed_usage_status_and_limits(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "trace_enabled", True)
    monkeypatch.setattr(settings, "trace_path", path)
    operation = tracing.ModelOperation("structured", "heavy", cause="degradation")
    callback = model_layer._AttemptStatus(0, "groq", "model-id")
    run_id = uuid4()
    message = SimpleNamespace(
        usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        response_metadata={
            "status_code": 200,
            "headers": {
                "x-ratelimit-remaining-requests": "9",
                "x-ratelimit-remaining-tokens": "99",
                "x-ratelimit-reset-requests": "1s",
                "x-ratelimit-reset-tokens": "2s",
            },
        },
    )
    result = SimpleNamespace(generations=[[SimpleNamespace(message=message)]])
    with tracing.request_context("cookie"), tracing.model_operation(operation):
        callback.on_chat_model_start({}, [], run_id=run_id)
        callback.on_llm_end(result, run_id=run_id)
        tracing.conversation_summary()

    event, summary = _events(path)
    assert event["token_usage"]["total_tokens"] == 5
    assert (event["http_status"], event["remaining_requests"], event["remaining_tokens"]) == (200, "9", "99")
    assert (event["request_reset"], event["token_reset"]) == ("1s", "2s")
    assert event["cause"] == "degradation" and summary["degradations"] == 1


def test_concurrent_invocations_keep_request_and_operation_ids_isolated(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "trace_enabled", True)
    monkeypatch.setattr(settings, "trace_path", path)
    chain = model_layer._compose(
        [(ScriptedChat(responses=["A", "B"]), (httpx.ConnectError,))]
    )
    barrier = Barrier(2)
    original_start = model_layer._AttemptStatus.on_chat_model_start

    def synchronized_start(self, *args, **kwargs):
        barrier.wait()
        return original_start(self, *args, **kwargs)

    monkeypatch.setattr(
        model_layer._AttemptStatus, "on_chat_model_start", synchronized_start
    )

    def invoke(name):
        with tracing.request_context(f"cookie-{name}"):
            context = tracing.current_context()
            chain.invoke(
                name,
                {
                    "metadata": {
                        "trace_operation": name,
                        "trace_tier": "heavy",
                        "trace_logical_operation_id": f"logical-{name}",
                        "trace_tier_operation_id": f"tier-{name}",
                    }
                },
            )
            return name, context.conversation_id, context.request_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        expected = dict(
            (name, (conversation_id, request_id))
            for name, conversation_id, request_id in executor.map(invoke, ("A", "B"))
        )

    events = _events(path)
    assert len(events) == 2
    for event in events:
        conversation_id, request_id = expected[event["operation"]]
        assert event["conversation_id"] == conversation_id
        assert event["request_id"] == request_id
        assert event["logical_operation_id"] == f"logical-{event['operation']}"
        assert event["tier_operation_id"] == f"tier-{event['operation']}"


def test_total_exhaustion_traces_every_attempt(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "trace_enabled", True)
    monkeypatch.setattr(settings, "trace_path", path)
    monkeypatch.setattr(settings, "model_attempts", 2)
    chains = [ScriptedChat(responses=[httpx.ConnectError("secret")] * 2) for _ in range(2)]
    with tracing.request_context("cookie"):
        chain = model_layer._compose(
            [(model, (httpx.ConnectError,)) for model in chains],
            {"initial": 0.001, "max": 0.005},
        )
        try:
            chain.invoke("private")
        except model_layer.AllProvidersExhausted:
            pass
    attempts = _events(path)
    assert [(event["provider"], event["attempt_index"], event["cause"]) for event in attempts] == [
        ("provider_0", 1, None), ("provider_0", 2, "retry"),
        ("provider_1", 1, "fallback"), ("provider_1", 2, "retry"),
    ]


def test_search_events_and_logs_never_include_query_or_snippets(tmp_path, monkeypatch, caplog):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "trace_enabled", True)
    monkeypatch.setattr(settings, "trace_path", path)
    monkeypatch.setattr(settings, "tavily_api_key", "key")
    query = "CANARY_QUERY"

    class BrokenTavily:
        def __init__(self, **kwargs):
            pass

        def invoke(self, value):
            raise RuntimeError("search failed")

    class FakeDDGS:
        def text(self, value, max_results):
            return [{"href": "https://example.test", "body": "CANARY_SNIPPET"}]

    monkeypatch.setattr(search, "TavilySearch", BrokenTavily)
    monkeypatch.setattr(search, "DDGS", FakeDDGS)
    with tracing.request_context("cookie"), caplog.at_level(logging.WARNING):
        result = search.search_requirements(query)

    assert result.text.endswith("CANARY_SNIPPET")
    events = _events(path)
    assert [(event["backend"], event["outcome"], event["cause"]) for event in events] == [
        ("tavily", "error", None),
        ("duckduckgo", "success", "fallback"),
    ]
    captured = path.read_text() + caplog.text
    assert query not in captured and "CANARY_SNIPPET" not in captured


def test_disabled_and_unwritable_tracing(tmp_path, monkeypatch, caplog):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "trace_path", path)
    monkeypatch.setattr(settings, "trace_enabled", False)
    with tracing.request_context("cookie"):
        tracing.write({"event": "graph_node"})
    assert not path.exists()

    monkeypatch.setattr(settings, "trace_enabled", True)
    monkeypatch.setattr(settings, "trace_path", tmp_path)  # opening a directory fails
    with caplog.at_level(logging.WARNING), tracing.request_context("cookie"):
        tracing.write({"event": "graph_node"})
        tracing.write({"event": "graph_node"})
    assert caplog.messages.count("operational trace write failed") == 1


def test_graph_nodes_include_story_020_nodes_and_terminal_summary(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "trace_enabled", True)
    monkeypatch.setattr(settings, "trace_path", path)
    monkeypatch.setattr(server, "_sessions", set())
    monkeypatch.setattr(server, "_active_turns", {})

    class Graph:
        async def astream(self, payload, config, stream_mode):
            for name in ("fit_batch", "fit_more_ask", "fit_more_decide"):
                yield "tasks", {"id": name, "name": name}
                yield "tasks", {"id": name, "name": name, "result": {"CANARY_CV": "hidden"}}

        def get_state(self, config):
            return SimpleNamespace(values={"messages": []}, next=())

    monkeypatch.setattr(server, "GRAPH", Graph())
    response = TestClient(server.app).post("/chat", json={"message": ""})

    assert response.status_code == 200 and "event: done\ndata: over" in response.text
    events = _events(path)
    nodes = [event for event in events if event["event"] == "graph_node"]
    assert [(event["node"], event["phase"], event["outcome"]) for event in nodes] == [
        ("fit_batch", "start", None), ("fit_batch", "finish", "success"),
        ("fit_more_ask", "start", None), ("fit_more_ask", "finish", "success"),
        ("fit_more_decide", "start", None), ("fit_more_decide", "finish", "success"),
    ]
    for name in ("fit_batch", "fit_more_ask", "fit_more_decide"):
        pair = [event for event in nodes if event["node"] == name]
        assert pair[0]["node_operation_id"] == pair[1]["node_operation_id"]
    assert events[-1]["event"] == "conversation_summary"
    assert "CANARY_CV" not in path.read_text()


def test_interrupted_turn_has_no_summary(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "trace_enabled", True)
    monkeypatch.setattr(settings, "trace_path", path)
    monkeypatch.setattr(server, "_sessions", set())
    monkeypatch.setattr(server, "_active_turns", {})

    class Graph:
        async def astream(self, payload, config, stream_mode):
            yield "tasks", {"id": "fit_more_decide", "name": "fit_more_decide"}
            yield "tasks", {"id": "fit_more_decide", "name": "fit_more_decide", "result": {}}

        def get_state(self, config):
            return SimpleNamespace(values={"messages": []}, next=("fit_more_decide",))

    monkeypatch.setattr(server, "GRAPH", Graph())
    TestClient(server.app).post("/chat", json={"message": ""})
    assert all(event["event"] != "conversation_summary" for event in _events(path))


def test_graph_stream_failure_closes_started_node(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(settings, "trace_enabled", True)
    monkeypatch.setattr(settings, "trace_path", path)
    monkeypatch.setattr(server, "_sessions", set())
    monkeypatch.setattr(server, "_active_turns", {})

    class Graph:
        async def astream(self, payload, config, stream_mode):
            yield "tasks", {"id": "broken", "name": "interviewer"}
            raise RuntimeError("CANARY_FAILURE")

        def get_state(self, config):
            return SimpleNamespace(values={"messages": []}, next=())

    monkeypatch.setattr(server, "GRAPH", Graph())
    response = TestClient(server.app).post("/chat", json={"message": ""})

    assert response.status_code == 200 and '"kind": "unknown"' in response.text
    nodes = [event for event in _events(path) if event["event"] == "graph_node"]
    assert [(event["phase"], event["outcome"]) for event in nodes] == [
        ("start", None),
        ("finish", "failure"),
    ]
    assert nodes[0]["node_operation_id"] == nodes[1]["node_operation_id"]
    assert nodes[1]["duration_ms"] is not None
    assert "CANARY_FAILURE" not in path.read_text()
