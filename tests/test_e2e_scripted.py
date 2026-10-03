"""Offline checks for Story 021's focused live-drive trace boundaries."""

import json

import pytest

import scripts.e2e_scripted as scripted


def _write(path, *, nodes=(), searches=0):
    events = [
        {"event": "graph_node", "phase": "start", "node": node}
        for node in nodes
    ] + [
        {"event": "requirements_search"} for _ in range(searches)
    ]
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


@pytest.mark.parametrize(
    ("stop", "nodes", "searches"),
    [
        ("pre-final", ("interviewer",), 0),
        ("shortlist", ("shortlister", "cv_ask", "cv_intake"), 0),
        ("cv-echo", ("cv_intake", "cv_confirm"), 0),
        ("first-fit-batch", ("fit_batch", "fit_more_ask", "fit_more_decide"), 5),
    ],
)
def test_focused_boundary_accepts_only_selected_work(tmp_path, stop, nodes, searches):
    trace = tmp_path / "trace.jsonl"
    _write(trace, nodes=nodes, searches=searches)
    scripted.assert_boundary(trace, stop)


@pytest.mark.parametrize(
    ("stop", "nodes", "searches"),
    [
        ("pre-final", ("interviewer", "shortlister"), 0),
        ("shortlist", ("shortlister", "cv_confirm"), 0),
        ("cv-echo", ("cv_confirm", "fit_batch"), 1),
        ("first-fit-batch", ("fit_batch", "fit_batch"), 10),
    ],
)
def test_focused_boundary_rejects_later_work(tmp_path, stop, nodes, searches):
    trace = tmp_path / "trace.jsonl"
    _write(trace, nodes=nodes, searches=searches)
    with pytest.raises(AssertionError):
        scripted.assert_boundary(trace, stop)


@pytest.mark.parametrize(
    ("stop", "replies", "nodes", "searches", "expected_calls"),
    [
        ("pre-final", [["degree?"], ["field?"], ["regions?"]], ("interviewer",), 0, 3),
        (
            "shortlist",
            [["degree?"], ["field?"], ["regions?"], []],
            ("shortlister", "cv_ask", "cv_intake"),
            0,
            4,
        ),
        (
            "cv-echo",
            [["degree?"], ["field?"], ["regions?"], [], ["I read from your CV"]],
            ("shortlister", "cv_ask", "cv_intake", "cv_confirm"),
            0,
            5,
        ),
        (
            "first-fit-batch",
            [["degree?"], ["field?"], ["regions?"], [], ["I read from your CV"], []],
            ("shortlister", "cv_ask", "cv_intake", "cv_confirm", "fit_batch"),
            5,
            6,
        ),
    ],
)
def test_focused_drive_exits_at_named_boundary(
    tmp_path, monkeypatch, stop, replies, nodes, searches, expected_calls
):
    shortlist = "Here is your shortlist:\n" + "\n".join(f"- University {i}" for i in range(8))
    first_batch = "Here is how your confirmed CV facts stack up:\n" + "\n".join(
        f"- University {i}: strong" for i in range(5)
    )
    replies = list(replies)
    replies[3:4] = [[shortlist, "Please upload your CV"]]
    if stop == "first-fit-batch":
        replies[5:6] = [[first_batch]]
    calls = []

    def fake_turn(client, message):
        calls.append(message)
        return replies[len(calls) - 1]

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"path": "/tmp/cv.txt"}

    class Client:
        def post(self, *args, **kwargs):
            return Response()

    trace = tmp_path / "trace.jsonl"
    _write(trace, nodes=nodes, searches=searches)
    monkeypatch.setattr(scripted, "turn", fake_turn)
    monkeypatch.setattr(scripted, "TOKEN_COUNTS", [2])

    scripted.run_conversation(Client(), stop, trace)

    assert len(calls) == expected_calls
