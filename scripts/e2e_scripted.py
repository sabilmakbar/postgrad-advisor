"""Scripted end-to-end conversation over the live web chat (story 006, AC 5).

Burns real free-tier quota, so it lives here and not in pytest. Two runs make
the cycle-1 exit check:

    uv run python scripts/e2e_scripted.py               # SC-001 (all keys)
    uv run python scripts/e2e_scripted.py --no-gemini   # SC-004 (Gemini unset)

Each run spawns its own server with a fresh Story 015 trace. Use ``--stop`` to
exercise only the selected boundary; every focused stop asserts from that
isolated trace that later graph/search work did not run.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # plain `python scripts/...` lacks the repo root

from app.fit import _PROB_RE  # noqa: E402  SC-005: the one probability pattern, reused

PORT = 8016
CV_FIXTURE = ROOT / "tests" / "fixtures" / "cv.txt"

# Each answer may fill several slots; the last is a catch-all restatement in
# case earlier parses missed a slot. More answers than this = FAIL (SC-001
# allows at most 10 agent turns anyway).
ANSWERS = [
    "I'm after a research master's in computer science; my focus is natural "
    "language processing.",
    "I'd like to study in Europe — ideally the United Kingdom or the Netherlands.",
    "No hard constraints: budget is flexible and I can start anytime.",
    "To restate: degree intent masters_research, field computer science, "
    "target regions United Kingdom and Netherlands.",
]
CONFIRM = "Yes, that's all correct."

# Fix round: a vague answer must get a probing question back — never silence,
# never the slot parser's raw JSON (the user-reported defect).
VAGUE = "Hmm, honestly I'm not sure yet — something after my bachelor's, I guess?"

TOKEN_COUNTS: list[int] = []  # token events per turn, for the AC-2 streaming check
STOPS = ("pre-final", "shortlist", "cv-echo", "first-fit-batch")


def trace_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def assert_boundary(path: Path, stop: str) -> None:
    events = trace_events(path)
    starts = [e["node"] for e in events if e.get("event") == "graph_node" and e.get("phase") == "start"]
    searches = sum(e.get("event") == "requirements_search" for e in events)
    forbidden = {
        "pre-final": {"shortlister", "cv_ask", "cv_intake", "cv_confirm", "fit_batch"},
        "shortlist": {"cv_confirm", "fit_batch"},
        "cv-echo": {"fit_batch"},
    }
    if stop in forbidden:
        assert forbidden[stop].isdisjoint(starts), f"later nodes ran: {forbidden[stop] & set(starts)}"
        assert searches == 0, f"later requirements searches ran: {searches}"
    else:
        assert starts.count("fit_batch") == 1, "more than the selected first fit batch ran"
        assert searches > 0, "first fit batch recorded no requirements search"


def bare_json(text: str) -> bool:
    """Structured-output leak signature: the text is a bare JSON object."""
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


def sse_events(response) -> list[tuple[str, str]]:
    """Parse an SSE body into (event, data) pairs; multiline data rejoined."""
    events = []
    for block in response.read().decode().split("\n\n"):
        event, data = "message", []
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
        if data:
            events.append((event, "\n".join(data)))
    return events


def turn(client: httpx.Client, message: str) -> list[str]:
    """One chat turn; returns the agent's authoritative message texts."""
    with client.stream("POST", "/chat", json={"message": message}) as response:
        response.raise_for_status()
        events = sse_events(response)
    errors = [d for e, d in events if e == "error"]
    assert not errors, f"agent turn failed: {errors}"
    TOKEN_COUNTS.append(sum(1 for e, _ in events if e == "token"))
    # Fix round: no structured-output JSON may reach the page — neither in the
    # assembled token draft nor in the authoritative messages.
    draft = "".join(d for e, d in events if e == "token")
    assert not draft.lstrip().startswith("{"), f"JSON leaked into the token stream: {draft[:100]}"
    messages = [d for e, d in events if e == "message"]
    leaks = [m for m in messages if bare_json(m)]
    assert not leaks, f"structured-output JSON replayed as a message: {leaks}"
    return messages


def run_conversation(client: httpx.Client, stop: str | None, trace_path: Path) -> None:
    said: list[str] = []  # everything the agent said, for the SC-005 scan

    print("interview:")
    said += turn(client, "")  # page-load start: opening question

    # Fix round: a vague answer must end with a rendered question — non-JSON
    # prose with a question mark — never silence or parser output.
    probe = turn(client, VAGUE)
    said += probe
    assert probe, "vague answer got silence (no agent message)"
    assert probe[-1].rstrip().endswith("?"), (
        f"vague-answer turn did not end in a question: {probe[-1]!r}"
    )
    print(f"  vague answer -> probing question: {probe[-1][:70]}...")

    shortlist = None
    agent_turns = 2  # opening question + the vague-answer follow-up
    for index, answer in enumerate(ANSWERS):
        if stop == "pre-final" and index == 1:
            assert_boundary(trace_path, stop)
            print("boundary: stopped before the final interview answer")
            return
        messages = turn(client, answer)
        said += messages
        print(f"  turn {agent_turns}: {messages[0][:70]}...")
        shortlist = next((m for m in messages if m.startswith("Here is your shortlist:")), None)
        if shortlist:
            break
        agent_turns += 1
    assert shortlist, f"interview did not complete within {len(ANSWERS)} answers"
    assert agent_turns <= 10, f"interview took {agent_turns} agent turns (SC-001: <=10)"

    assert max(TOKEN_COUNTS) > 1, "agent prose arrived as one blob, never as tokens (AC 2)"
    print(f"streaming: up to {max(TOKEN_COUNTS)} token events in a turn")

    programs = [ln for ln in shortlist.splitlines() if ln.startswith("- ")]
    print(f"shortlist: {len(programs)} programs after {agent_turns} agent turns")
    assert 8 <= len(programs) <= 12, f"shortlist size {len(programs)} (SC-001: 8-12)"
    assert any("CV" in m for m in said), "no CV request after the shortlist"

    if stop == "shortlist":
        assert_boundary(trace_path, stop)
        print("boundary: stopped at the CV-input pause after shortlist")
        return

    upload = client.post(f"/upload?name={CV_FIXTURE.name}", content=CV_FIXTURE.read_bytes())
    upload.raise_for_status()
    said += (echo := turn(client, upload.json()["path"]))
    assert any("read from your CV" in m for m in echo), f"no CV facts echo in {echo}"
    print("cv: uploaded, facts echoed")

    if stop == "cv-echo":
        assert_boundary(trace_path, stop)
        print("boundary: stopped after CV echo")
        return

    said += (final := turn(client, CONFIRM))
    verdicts = next((m for m in final if m.startswith("Here is how your confirmed")), None)
    assert verdicts, f"no fit verdicts after confirmation, got {final}"
    lines = [ln for ln in verdicts.splitlines() if ln.startswith("- ")]
    assert len(lines) == min(5, len(programs)), f"first fit batch assessed {len(lines)} programs"
    if stop == "first-fit-batch":
        assert_boundary(trace_path, stop)
        print("boundary: stopped after the first fit batch")
        return

    while len(lines) < len(programs):
        said += (final := turn(client, "yes"))
        verdicts = next((m for m in final if m.startswith("Here is how your confirmed")), None)
        assert verdicts, f"no continued fit verdicts, got {final}"
        lines = [ln for ln in verdicts.splitlines() if ln.startswith("- ")]
    assert len(lines) == len(programs), f"{len(lines)} verdicts for {len(programs)} programs"
    labeled = [ln for ln in lines if any(f": {label}" in ln for label in
               ("strong", "borderline", "reach", "unknown"))]
    assert len(labeled) == len(lines), f"unlabeled verdict lines: {set(lines) - set(labeled)}"
    print(f"verdicts: one per program ({len(lines)}), all labeled")

    hits = [m for m in said if _PROB_RE.search(m)]
    assert not hits, f"numeric admission odds in agent output (SC-005): {hits}"
    print("sc-005: zero probability patterns in agent output")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gemini", action="store_true",
                        help="run with GOOGLE_API_KEY unset (SC-004 fallback proof)")
    parser.add_argument("--stop", choices=STOPS, help="stop at one focused acceptance boundary")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="postgrad-advisor-trace-") as trace_dir:
        trace_path = Path(trace_dir) / "trace.jsonl"
        env = dict(os.environ)
        env.update(TRACE_ENABLED="true", TRACE_PATH=str(trace_path))
        if args.no_gemini:
            env["GOOGLE_API_KEY"] = ""  # masks the .env value for the child server

        server = subprocess.Popen(
            ["uv", "run", "uvicorn", "app.server:app", "--port", str(PORT)],
            cwd=ROOT, env=env, start_new_session=True,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{PORT}",
                              timeout=httpx.Timeout(600, connect=5)) as client:
                for _ in range(60):  # readiness: up to 30s
                    if server.poll() is not None:
                        sys.exit(f"server exited early (is port {PORT} taken?)")
                    try:  # must be OUR page — a stranger on the port also 200s
                        if "Postgrad Advisor" in client.get("/").raise_for_status().text:
                            break
                    except httpx.HTTPError:
                        time.sleep(0.5)
                else:
                    sys.exit("server never became ready")
                run_conversation(client, args.stop, trace_path)
            label = args.stop or ("SC-004 (no Gemini)" if args.no_gemini else "SC-001")
            print(f"PASS: {label} scripted conversation completed")
        finally:
            os.killpg(server.pid, signal.SIGTERM)


if __name__ == "__main__":
    main()
