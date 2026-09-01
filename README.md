# postgrad-advisor

A local chat agent that interviews you about postgraduate study goals, shortlists real
programs from a verified QS-2026 snapshot, and judges your CV against each one — honestly,
with sources, and without inventing anything.

Runs entirely on free API tiers. Single local user, no auth, no deployment.

## Run it

```bash
uv sync
cp .env.example .env      # fill in the keys you have
uv run uvicorn app.server:app
```

Open <http://localhost:8000>. A page load starts a fresh interview.

Keys (all free tiers, all optional except at least one model provider):

| Key | Used for | Without it |
|---|---|---|
| `GOOGLE_API_KEY` | primary model (Gemini) | falls back to Groq |
| `GROQ_API_KEY` | fallback model | falls back to Cloudflare, then fails honestly |
| `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` | last-resort model | chain is shorter |
| `TAVILY_API_KEY` | entry-requirement search | falls back to DuckDuckGo |

Tests: `uv run pytest` (68, all offline — no key needed).
End-to-end against live providers: `uv run python scripts/e2e_scripted.py` (add `--no-gemini`
to prove the fallback path).

## What to expect, step by step

This is the manual-test checklist: what each step should do, and what would be a defect.

### 1. Interview

The agent asks **one question per turn**, starting with degree intent (taught master's,
research master's, or PhD), then field, then target regions. Vague answers get a narrower
follow-up rather than a guess. The interview ends after at most **10 agent turns**.

- Tone should be realistic, not encouraging: it names difficulties and offers alternatives.
  "Realistically this will be demanding, consider X" is right; "you'll do great!" is a defect.
- Asking about a **postdoc** gets a plain decline (postdocs are supervisor- and funding-driven,
  out of scope here) and the interview continues.
- Prose answers **stream in visibly**, word by word. Expect **6–13 seconds of silence first** —
  that is the free-tier rate limiter pacing requests, not a hang. A placeholder line and typing
  dots sit in the bubble during the wait.
- **Defect if:** more than one question in a turn, raw JSON in a reply, an empty bubble, or the
  same question asked twice in a row.

### 2. Shortlist

When intent + field + at least one region are known, you get **8–12 real universities** from
`data/programs.json` (120 entries, 20 per continent, ranks web-verified 2026-08-29).

- Every entry shows university, country, QS rank context, degree type, and one line on why it
  fits **your stated answers** — plus at least one honest caveat (competition, tuition, language,
  weak coverage of your sub-field).
- If too few match, the agent **says it widened** and why. If fewer than 8 could be grounded, it
  says that too, rather than padding.
- KAIST shows "excluded from QS 2026 (survey-canvassing penalty)" instead of a rank — that is
  correct, not a bug.
- **Defect if:** a university not in the dataset appears, generic praise with no caveat anywhere,
  or a silent short list.

### 3. CV

Attach with the **CV** button. The file uploads immediately but the path is **held as a chip in
the composer** — it is sent only when you press Send, with whatever you typed.

| When you attach | What should happen |
|---|---|
| Mid-interview, with typed text | "Got your CV — I'll read it right after we finish the interview", **and** your typed answer is counted; the interview advances |
| Mid-interview, nothing typed | Same acknowledgment; the current question stands (it is not repeated) |
| At the CV step | Parsed straight away — no repeated ask |
| At the confirm step | Treated as "use this one instead": re-parsed, and any correction you typed is applied too |
| On the final (10th) interview turn | The run ends honestly, saying the CV is held but the interview lacked detail — start a fresh chat |
| After the run is over | The page says the verdicts already used the earlier CV; a fresh chat is needed |

Parsing is **local** (pypdf / plain text). Extracted facts are echoed back — degree, GPA, publications,
English test — for you to confirm or correct before anything is judged.

Known gap (story 011): a CV handed over **during** the interview is held unread until the interview
ends, so the questions that follow it are not yet grounded in your background. Intent questions
(degree type, regions, timing) are asked regardless — a CV states where you have been, not what you
want next.

- **Defect if:** the agent claims it "can't open files", the CV path shows up as a chat answer,
  the CV step asks twice, or an attached CV is silently forgotten.

### 4. Fit verdicts

For each shortlisted program the agent searches current entry requirements and returns one of
**strong / borderline / reach / unknown**, with reasons referencing both your CV and the
requirements found, plus **at least one source URL**.

- `unknown` means the requirements could not be found — it shows the query it tried instead of
  guessing.
- There are **never numeric admission odds**. "You have a 70% chance" is prohibited by design and
  scrubbed in code; seeing one is a defect.
- Searches are capped at 12 per conversation to protect the free tier.

### 5. Errors

- All model providers exhausted → a plain-language message, never a stack trace.
- Connection lost mid-turn → the partial answer stays on screen with a distinct error line and a
  **Try again** button.
- Known gap (fixed in story 010): being **offline** currently reports the quota message. Cause-specific
  copy is the next story's work.

## How it is built

`app/graph.py` is a LangGraph state machine: interview → shortlist → CV intake → confirm → fit →
summary. Each node calls models through `app/model_layer.py`, never a provider SDK directly — that
layer owns the fallback chain (Gemini → Groq → Cloudflare), retries on transient errors only, shares
rate limiters per model, and skips models whose ids fail a generation probe.

Two anti-hallucination guarantees are **structural, not prompt-based**:

- The shortlister filters the dataset in code and lets the model rank only by index, so a made-up
  university cannot reach the output.
- Fit verdicts may cite only URLs that were actually fetched; invented sources are replaced.

Engagement docs (spec, decision record, stories) live in `docs/` and `dev/`, kept out of git by
local exclude.
