# Voice-Enabled Virtual Assistant — Multi-Tool AI Agent

A voice assistant for a small college. You ask a question out loud, it picks
one of **five skills**, runs it, and reads a short answer back to you. Sign in
as a student and it answers about *you* — your attendance, your marks, your
timetable, your reminders. The web UI shows its working: which skill fired and
why, the SQL it generated, the passages it retrieved, and how long each stage
took.

| Skill | Handles | Example |
|---|---|---|
| `query_database` | student records, marks, attendance, rankings | *"Who are the top three students in semester 6?"* |
| `search_documents` | policies, rules, procedures | *"How much does re-evaluation cost?"* |
| `timetable` | class schedule | *"What's my next class?"* |
| `reminders` | personal to-dos (the only tool that writes) | *"Remind me to submit the AI assignment on Friday"* |
| `smalltalk` | greetings, capabilities, off-topic | *"What can you do?"* |

Everything runs locally except one LLM call chain, which uses NVIDIA NIM's free
endpoint. **No paid API, no credit card, one free key.**

---

## Contents

- [What it does](#what-it-does)
- [Requirements](#requirements)
- [Setup (Windows)](#setup-windows)
- [Running it](#running-it)
- [Example questions](#example-questions)
- [How it works](#how-it-works)
- [Accounts and data isolation](#accounts-and-data-isolation)
- [SQL safety guards](#sql-safety-guards)
- [Run with Docker](#run-with-docker)
- [Public demo (Hugging Face Space)](#public-demo-hugging-face-space)
- [Swapping to Piper TTS](#swapping-to-piper-tts)
- [Changing the model](#changing-the-model)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

---

## What it does

```
microphone ─► faster-whisper (local STT, auto language detect)
                    │
              wake phrase gate (hands-free mode only)
                    │
                    ▼
              router (1 LLM call, strict JSON, 5-way)
                    │
   ┌────────────┬───┴────┬────────────┬───────────┐
   ▼            ▼        ▼            ▼           ▼
query_       search_   timetable   reminders   smalltalk
database     documents                            │
text►SQL►    embed►     fixed       bound         │
SQLite (ro)  Chroma     queries     params (rw)   │
   └────────────┴────────┴────────────┴───────────┘
                    ▼
       answer composition (1 LLM call, spoken style)
                    │
                    ▼
            pyttsx3 (local TTS) ─► browser plays WAV
```

Conversation memory keeps the last 6 turns per session, so *"and what about
semester 4?"* resolves against the previous question. Signing in binds the
session to a student id, which is injected into tool prompts so "my" resolves
to that student.

### Accounts and data isolation

Sign in with a roll number and password, or create an account by claiming your
seeded student record. Identity comes from a **verified JWT** — never from
anything the browser or the model can assert.

**Demo logins** (seeded by `data/seed_db.py`):

| Login | Password | Role | Sees |
|---|---|---|---|
| `CS06020` | `demo1234` | student | only their own records |
| `staff` | `staff1234` | staff | the whole college |

**To test signing up**, you need a roll number and name that match a seeded
student — you cannot register as someone who does not exist. Use
`CS04001` with the name `Karan Verma` (the seed is deterministic, so that pair
is always valid), or read any other pair out of the database:

```powershell
python -c "import sqlite3; print(sqlite3.connect('data/app.db').execute('SELECT roll_no, name FROM students LIMIT 5').fetchall())"
```

Names are matched ignoring case and extra spaces. Note there is deliberately no
endpoint that lists students — that was removed because it leaked the whole
cohort, which is also why the roll number has to come from outside the app.

There are two roles, and the difference is enforced by the database rather than
by prompting. Ask the same question as each:

| Question | As the student | As staff |
|---|---|---|
| *How many students are in semester 6?* | "I can only see your own record, so I can't tell you the class size." | "There are 15 students in semester 6." |
| *Who has the lowest attendance?* | "I can only show you your own records. Your attendance is 100 percent." | "Pooja Joshi has the lowest attendance at 25 percent." |
| *Who are the top three students by marks?* | refuses, then gives their own 74.4% | "Nikhil Reddy is highest, then Kabir Nair, then Ananya Verma." |

**The generated SQL is identical in both columns.** See
[SQL safety guards](#sql-safety-guards) for how that works.

---

## Requirements

- **Windows** with **Python 3.11 (64-bit / x64)**
- About 1.5 GB free disk (mostly PyTorch, pulled in by sentence-transformers)
- A free NVIDIA NIM API key — see below

> **Important — architecture matters.** Use an **x64** build of Python. Three
> dependencies (`ctranslate2` for faster-whisper, `torch` for
> sentence-transformers, and `chromadb`'s native deps) publish **no
> Windows-on-ARM wheels**, so `pip install` fails outright on an ARM64 Python.
> If your machine is Windows on ARM, install the regular **x64** Python 3.11
> from python.org — it runs fine under Windows' x64 emulation and every wheel
> then installs normally. Check what you have with:
>
> ```powershell
> python -c "import platform; print(platform.machine(), platform.python_version())"
> ```
>
> You want `AMD64 3.11.x`. If it prints `ARM64`, install x64 Python 3.11 first.

---

## Setup (Windows)

> **Already done on this machine.** x64 Python 3.11.9 is installed at
> `C:\Users\manas\AppData\Local\Programs\Python\Python311-x64`, `.venv` is
> created with every dependency installed, `data\app.db` is seeded, the docs
> are ingested (17 chunks), the whisper and MiniLM models are downloaded, and
> `.env` holds your key. **You can skip straight to
> [Running it](#running-it)** and just run `.\run.bat`. The steps below are for
> setting this up somewhere else.

### 1. Get the free NVIDIA NIM key

1. Go to **<https://build.nvidia.com>**
2. Sign in with an email address. **No credit card is required.**
3. Open any model page (e.g. Llama 3.3 70B Instruct) and click **Get API Key**.
4. Copy the key — it starts with `nvapi-`.

While you are there, **confirm the exact model identifier** shown on the page.
Model strings on NIM occasionally change, and a string being *listed* by the
API does not guarantee it is actually serving traffic. This project ships with
`meta/llama-3.1-70b-instruct`, set in [app/config.py](app/config.py) as
`LLM_MODEL`.

> **Why not Llama 3.3?** `meta/llama-3.3-70b-instruct` appears in NIM's
> `/v1/models` list but, when tested on this account, every inference request
> to it hung until timeout while `meta/llama-3.1-70b-instruct` answered the
> identical request in 0.6 s. If you want to try 3.3 later, change the one
> constant and check `/health` — it will report the failure immediately rather
> than leaving you guessing.

### 2. Create the virtual environment and install

Open PowerShell in the `voice-assistant` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the activate script, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

### 3. Add your key

```powershell
copy .env.example .env
notepad .env
```

Paste the key so the file reads:

```
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. Seed the database

```powershell
python data\seed_db.py
```

This creates `data\app.db` with 30 students, 6 subjects, 150 marks rows and 200
attendance rows. A fixed random seed (42) means the data is identical on every
machine, so the example answers below always match.

---

## Running it

```powershell
.\run.bat
```

or manually:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open **<http://127.0.0.1:8000>**.

`run.bat` seeds the database if it is missing; document ingestion happens
automatically on server startup. The status line at the top right of the page
shows whether the LLM, database and document index are all healthy.

### First run is slow

The first launch downloads two models, one time only, roughly **150 MB total**:

- faster-whisper `base` (~75 MB) — downloads on the first voice question
- `all-MiniLM-L6-v2` embeddings (~90 MB) — downloads during startup ingestion

They are cached in your user profile, so subsequent starts take a few seconds.
The first voice question after a fresh install will feel slow while the whisper
model loads; typed questions work immediately.

### Using it

- **Hold** the *Push to talk* button — **or just hold the Space bar** — speak,
  and release. A live level meter shows the mic is picking you up. The browser
  records via MediaRecorder and uploads to `/ask-audio`.
- **👂 Hands-free** listens continuously instead — no button. It spends the
  first ~1.5s learning your room's noise floor (the hint line says
  "calibrating"), then cuts a clip whenever you stop speaking. Because it
  uploads whatever it hears, a clip is only acted on when the transcript
  **starts with "Hey Campus…"**. Say *"Hey Campus, what's my next class?"*
  Anything not addressed to it appears greyed out for a few seconds and then
  disappears, so you can always see that the microphone is working.
  Untick **"Hands-free needs Hey Campus"** in voice settings to have it answer
  everything it hears — simpler for a demo, but it will react to any nearby
  speech.
- **Language** in voice settings: leave it on auto-detect and Whisper picks the
  language per clip, so you can ask one question in English and the next in
  Hindi. Force `en` or `hi` if detection struggles with your microphone.
- Or **type** a question and press Enter — no microphone needed. Typed
  questions are spoken too.
- **✎ misheard? edit & resend** appears under every voice question. Whisper
  mishears occasionally, and a garbled question can never produce a good
  answer, so one click drops the transcript into the text box to fix.
- **⏹ stops** the assistant mid-sentence; **🔁 replays** the last answer.
  Starting a new recording also cuts the audio, so it never talks over you.
- **🔊 Voice settings** picks the speaking voice, sets the speed (80–320 wpm)
  and can turn auto-play off. Choices persist in `localStorage`.
- Click the **🔍 show working** tag under any answer to see the tool, the SQL,
  the rows, the retrieved passages, and **how long each stage took**
  (speech-to-text, router, tool, compose, speech).
- **Clear conversation** wipes that session's memory server-side.

Your browser will ask for microphone permission on first use. `127.0.0.1`
counts as a secure origin, so this works without HTTPS.

---

## Example questions

**Database tool** (`query_database`):

1. *How many students are in semester 6?* → 15
2. *What's the average score in Compiler Design?* → about 69.5%
3. *Which student has the lowest attendance?* → Pooja Joshi, around 25%
4. *How many students scored above 80 in AI?* → 7
5. *How many students are there in each branch?* → 10 in each of three branches
6. *What's the highest midterm score in Database Systems?* → 50

**Document tool** (`search_documents`):

7. *What is the minimum attendance requirement?* → 75% per subject
8. *How much does re-evaluation cost and when can I apply?* → Rs. 800, within
   15 days
9. *How many books can I borrow from the library?* → four, for fourteen days
10. *What happens if I lose my ID card?*

**Personal — pick a name in "Signed in as" first:**

11. *What is my attendance?*
12. *What are my marks in Compiler Design?*
13. *What is my next class?* / *What are my classes on Friday?*
14. *Remind me to submit the AI assignment on Friday* → then *What are my reminders?*

**Assistant itself:**

15. *What can you do?* · *Hello* · *Who won the cricket world cup?* (declined politely)

**Follow-up, to show conversation memory:**

> **You:** How many students are in semester 6?
> **Assistant:** There are 15 students in semester six.
> **You:** *And what about semester 4?*
> **Assistant:** There are 15 students in semester four as well.

The second question has no subject of its own — it works because the last 6
turns are prepended to both the router prompt and the SQL prompt.

---

## How it works

A question arrives as text (typed) or as audio, which **faster-whisper**
transcribes locally on CPU. A single **router** LLM call classifies it and
returns strict JSON naming one of two tools; the response is parsed
defensively — fences stripped, embedded JSON extracted, and any failure falls
back to document search rather than erroring.

If the router picks **query_database**, the text-to-SQL tool builds a system
prompt containing the live `CREATE TABLE` statements plus three sample rows per
table, so the model sees real column names and value formats. The generated SQL
is cleaned, statically validated, and executed against a **read-only** SQLite
connection with results capped at 100 rows. If SQLite raises, the error is fed
back to the model once for a repair attempt.

If the router picks **search_documents**, the question is embedded with
**all-MiniLM-L6-v2** and the top 4 chunks are pulled from a local **Chroma**
collection built from `./docs` at startup (~500-character chunks, 50-character
overlap; re-ingestion is skipped if the collection is already populated).

Either way, raw rows and raw chunks are **never** spoken. A final composition
call turns them into one or two plain-text sentences suitable for
text-to-speech, which **pyttsx3** renders offline to a WAV that the browser
plays automatically. The question and answer are then appended to that
session's memory.

Every external call — NVIDIA NIM, SQLite, JSON parsing, embedding, synthesis —
is wrapped so a failure degrades to a sensible message instead of a crash.
Rate-limit (429) responses get an exponential backoff (2s, 4s, 8s), since the
free NIM tier is capped at about 40 requests per minute.

---

## SQL safety guards

Each guard is one line in [app/sql_tool.py](app/sql_tool.py):

| Guard | What it does |
|---|---|
| **Prompt constraint** | The system prompt demands exactly one `SELECT`, no prose, no fences. |
| **Defensive cleaning** | Strips markdown fences, `SQL:` prefixes and leading prose even when the model ignores the instruction. |
| **Single-statement check** | Rejects any query containing a semicolon other than one trailing one, blocking stacked-statement injection. |
| **SELECT-only check** | Rejects anything that does not begin with `SELECT`. |
| **Keyword blocklist** | Rejects `INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, TRUNCATE, ATTACH, DETACH, PRAGMA, VACUUM, REINDEX, GRANT, REVOKE`. |
| **Comment blocklist** | Rejects `--`, `/*` and `*/`, a common way to smuggle payloads past a parser. |
| **Read-only connection** | The database is opened as `file:app.db?mode=ro`, so SQLite itself refuses writes even if a validator bug let one through. |
| **Row cap** | A `LIMIT 100` is appended when the query has no limit, and only 100 rows are fetched. |
| **Single repair retry** | One SQLite error is fed back to the model for a fix; a second failure returns *"I couldn't answer that from the data"* instead of raising. |

The read-only guard is real, not advisory — attempting a write on that
connection raises `sqlite3.OperationalError: attempt to write a readonly
database`.

### Row-level security for LLM-generated SQL

The model chooses the SQL, so "don't look at other students" cannot be a prompt
instruction — it has to be enforced where the data lives. When a student is
signed in, the connection creates **temp views that shadow the base tables**:

```sql
CREATE TEMP VIEW marks AS SELECT * FROM main.marks WHERE student_id = 20;
```

SQLite resolves unqualified names against `temp` before `main`, so every
ordinary `SELECT ... FROM marks` now sees one student. Measured on the real
database with the *same* query text:

| Query | As staff | As student 20 |
|---|---|---|
| `SELECT COUNT(*) FROM students` | 30 | **1** |
| `SELECT COUNT(*) FROM marks` | 150 | **7** |
| `SELECT COUNT(*) FROM attendance` | 200 | **9** |
| "rank everyone by average" | all 30 | **only themselves** |

Three details make it actually hold, and each has a test in
[tests/test_rls.py](tests/test_rls.py):

1. **Schema qualification is rejected.** A temp view is bypassed by
   `SELECT * FROM main.marks`, which returns all 150 rows. `validate_sql`
   refuses `main.`, `temp.` and `sqlite_*`, with word-boundary matching so
   ordinary `s.name` aliases still work. **This is the single most important
   line in the security model** — without it the views are decorative.
2. **The credentials table is blanked on every connection**, staff included,
   and excluded from the schema dump. Otherwise `schema_context()` would paste
   password hashes into the prompt and into the "show the working" panel.
3. **The sample rows in the prompt are read through the same scoped
   connection.** An unscoped dump would leak other students' names to the model
   even when the query itself was properly restricted.

The composer is also told when its results are restricted; without that it read
its single row as the whole class and answered *"you have the highest marks"*,
which is misleading rather than private.

### Authentication

| Concern | How it's handled |
|---|---|
| Password storage | bcrypt with a per-user salt; passwords over 72 bytes rejected rather than silently truncated |
| User enumeration | unknown username and wrong password return the **identical** message |
| Brute force | 5 failed attempts locks the account for 15 minutes, correct password included |
| Token integrity | HS256 JWT signed with `JWT_SECRET`; tampered, forged and expired tokens all rejected |
| Logout | `users.token_version` is incremented, invalidating every token already issued — real revocation without a denylist |
| Token storage | `sessionStorage` (per-tab, cleared on close), 60-minute expiry, strict CSP, and the page loads zero third-party scripts |
| Endpoints | `/ask-text`, `/ask-audio`, `/clear` return 401 without a valid token |

**An honest note on the token:** a JWT in `sessionStorage` is readable by any
successful XSS, which an `HttpOnly` cookie would not be. That was a deliberate
choice here; the mitigations above (short expiry, real revocation, strict CSP,
no third-party JavaScript) reduce but do not eliminate the exposure. For a
production deployment, cookies are the safer default.

`/students` and `/sign-in` were **removed**: the first listed every student in
the college, and the second let any caller claim to be any student.

### How reminders write without weakening any of that

Reminders are the one feature that must write to the database, and they are a
separate module for exactly that reason. **The model never writes SQL on this
path.** It only extracts structured fields as JSON:

```json
{"action": "add", "text": "submit the AI assignment", "due": "friday"}
```

Python then parses the date itself and performs the write with a fixed
statement and bound parameters. There is no route from model output to a write
statement, so the text-to-SQL tool keeps its read-only connection and every
guard above stays exactly as strong. `test_skills.py` asserts this: the SQL
validator still rejects `INSERT INTO reminders`, and the text-to-SQL connection
still raises on a write to that table.

---

## Swapping to Piper TTS

TTS sits behind a small interface in [app/tts.py](app/tts.py), so nothing else
in the codebase needs to change:

1. `pip install piper-tts` and download a voice `.onnx` model.
2. Add a class next to `Pyttsx3Engine` implementing one method:
   `synthesize(self, text: str) -> tuple[bytes, str]`, returning
   `(wav_bytes, error)` and never raising.
3. Register it in the `_ENGINES` dict at the bottom of the file:
   `"piper": PiperEngine`.
4. Set `TTS_ENGINE=piper` in `.env`.

`speak()` and every caller stay exactly as they are.

---

## Changing the model

`LLM_MODEL` in [app/config.py](app/config.py) is the single place to change it.
The default is `meta/llama-3.1-70b-instruct`, which answered all 11 example
questions correctly. Alternatives that suit this workload (SQL generation,
strict JSON, and short prose):

| Model | Status |
|---|---|
| `meta/llama-3.1-70b-instruct` | **default** — verified working, ~0.6 s to first token |
| `meta/llama-3.1-8b-instruct` | verified working, fastest, slightly weaker SQL |
| `qwen2.5-coder-32b-instruct` | good at SQL; confirm the exact string on build.nvidia.com |
| `meta/llama-3.3-70b-instruct` | listed by NIM but timed out on every request when tested |

**Do not use a "thinking", "reasoning" or R1-style model.** They emit
chain-of-thought before the answer, which breaks both the strict-JSON router
parsing and the bare-SQL requirement.

### Performance

A voice turn originally took ~12.4s, all of it before anything appeared on
screen. Four changes, each measured:

| Change | Effect |
|---|---|
| **Stream the transcript** — `/ask-audio` returns newline-delimited JSON, emitting the transcript as soon as whisper finishes | transcript appears at **~1.5s** instead of ~12.4s |
| **Speculative tools** — the router, the SQL generator and document retrieval all start at once; the loser is abandoned (`SPECULATIVE_TOOLS`) | removes the router from the critical path entirely |
| **Split models by task** — routing and composition moved to 8B | those two steps went from 1.4–5.6s each to a consistent ~0.5s |
| **Preload whisper at startup** | first voice question no longer pays the ~6s model load |
| **Emit answer text before synthesizing speech** | text is readable before the WAV is built |

Measured end to end afterwards: transcript at **1.40–1.84s** (very consistent,
it is local), answer text typically **2.3–5.5s**.

**The remaining variance is NVIDIA's free tier, not this code.** Eight identical
calls, same prompt, same token count:

| Model | Latency across 8 calls |
|---|---|
| `llama-3.1-8b` | 0.41, 0.47, 0.41, 0.47, 0.50, 0.44, 0.46, 0.48 |
| `llama-3.1-70b` | 2.40, **25.19**, 3.59, **18.27**, 2.72, 1.28, 0.94, **24.15** |

The 70B endpoint is queue-contended, so an occasional turn still takes 30s+.
Nothing in the client can fix that. If you want predictable speed more than you
want accuracy, set `LLM_MODEL = ROUTER_MODEL` in
[app/config.py](app/config.py) to run 8B everywhere — it is ~60x faster but
drops from 11/11 to 8–10/11 on the example questions, flaking on "which student
has the lowest attendance" and on normalising "average score".

### Prompt rules that matter

Four rules in the text-to-SQL system prompt exist because the model got these
wrong in testing, and they are worth keeping if you adapt the schema:

- **Normalise marks before averaging.** `quiz` is out of 20, `midterm` out of
  50, `final` out of 100. Without the rule the model wrote `AVG(score)` and
  reported the Compiler Design average as *38* instead of *69.5%*.
- **Count distinct students.** "How many students scored above 80" over the
  `marks` table returned 8 with `COUNT(*)` (one student had two qualifying
  rows) instead of 7 with `COUNT(DISTINCT student_id)`.
- **Expand subject abbreviations.** The model wrote
  `LOWER(s.name) LIKE '%AI%'` — an uppercase pattern against a lowercased
  column, which can never match, and "AI" is not a substring of "artificial
  intelligence" anyway. It returned 0 rows.
- **Dump low-cardinality values.** `schema_context()` lists every value of
  `subjects.name`, `students.branch` and `marks.exam_type` so the model matches
  exact spellings instead of guessing at `LIKE` patterns.
- **Ranking questions need five explicit steps.** "Who are the top three
  students in semester 6?" produced
  `SELECT name FROM students WHERE semester = 6 ORDER BY id LIMIT 3` — ranked
  by *nothing*, and the answer then stated three arbitrary names as the top
  three. Silently wrong and confidently phrased, which is the worst kind of
  failure. The rule now forces the model to identify the measure, join to it,
  **select it as a second column**, order by it, and limit.
- **Always select the measure, not just the name.** When the query returned
  only names, the composer could not see any numbers and refused with "their
  marks are not provided" — even when the join was right. Returning
  `name, avg_percent` fixed both the refusals and made the ranking checkable.
- **Never LEFT JOIN when ranking.** Asked for the lowest attendance, the model
  wrote a LEFT JOIN; students with no attendance rows get a NULL average,
  SQLite sorts NULL first, and it confidently reported a student with no
  records at all as having "0.0 percent" attendance.
- **Keep the history short.** `SQL_HISTORY_TURNS = 2` — a long conversation
  history stapled onto a text-to-SQL prompt measurably degrades accuracy on
  smaller models, which start answering the *earlier* question. Follow-ups only
  need the turn immediately before.
- **Derive grades, never guess them.** Asked *"what is my CGPA?"*, the model
  found nothing resembling one, averaged the mark percentages and answered
  *"your CGPA is 74.4"* — confident and wrong, since a CGPA is out of 10.
  `subject_grades` and `cgpa` are now temp views computed from the weights and
  grade bands in [docs/exam_grading_policy.md](docs/exam_grading_policy.md)
  (quiz 20, midterm 30, final 50; O=10 down to F=0, credit-weighted), so the
  model just selects the right column. Only the components a student actually
  has are counted, rescaled to 100, so a missing quiz does not drag the total
  down. `tests/test_skills.py` re-implements the policy independently and
  asserts the views agree.
- **Clarifications are not lookups.** *"Is that a percentage or a CGPA?"* used
  to hit the database and generate `SELECT 'percentage'`. Questions answerable
  by re-reading the previous turn now route to `smalltalk`, which sees the
  conversation and repeats the figure it already gave rather than
  recalculating a different one.

---

## Project layout

```
voice-assistant/
  app/
    main.py           FastAPI app, endpoints, startup ingestion
    llm.py            NVIDIA NIM client (OpenAI-compatible), retry/backoff, JSON helper
    auth.py           bcrypt passwords, JWT issue/verify, signup, login, lockout
    router.py         5-way tool routing
    sql_tool.py       text-to-SQL: generate, validate, execute read-only, retry once
    rag_tool.py       chunk, embed, ingest, retrieve, confidence threshold
    timetable_tool.py class schedule; fixed queries, no generated SQL
    reminder_tool.py  the only write path; bound parameters, never model SQL
    smalltalk.py      greetings, capabilities, out-of-scope questions
    identity.py       verified token -> principal (user + student + role)
    wake.py           wake-phrase gating for hands-free mode
    stt.py            faster-whisper wrapper, auto language detection
    tts.py            pyttsx3 wrapper behind a swappable interface
    memory.py         per-session history (last 6 turns), TTL eviction
    compose.py        final spoken-answer composition
    config.py         env vars, paths, constants
  data/
    seed_db.py    creates and populates app.db (fixed seed 42)
    app.db        generated
    chroma/       generated vector store
  docs/
    attendance_policy.md
    exam_grading_policy.md
    student_faq.md
  static/
    index.html    single-file vanilla JS frontend
  tests/
    conftest.py       seeds the db if missing, puts app/ on the path
    test_sql_safety.py    the SQL guards
    test_pipeline.py      router, retry loop, RAG threshold, memory
    test_auth.py          passwords, tokens, lockout, revocation
    test_rls.py           row-level isolation (security critical)
    hands_free_harness.js headless VAD test (needs node)
  requirements.txt
  .env.example
  run.bat / run.sh
```

### Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/ask-text` | `{question, speak?, voice?, rate?}` + `Authorization: Bearer` | `{answer, tool, reason, timings, sql?, columns?, rows?, chunks?, audio_base64?, error}` |
| `POST` | `/ask-audio` | multipart: `audio`, `voice?`, `rate?`, `language?`, `require_wake?` + Bearer | **streams NDJSON**, see below |
| `POST` | `/clear` | Bearer | `{ok}` |
| `POST` | `/auth/signup` | `{roll_no, name, password}` | `{token, user}` |
| `POST` | `/auth/login` | `{username, password}` | `{token, user}` |
| `POST` | `/auth/logout` | Bearer | `{ok}` — revokes every issued token |
| `GET` | `/auth/me` | Bearer | `{user}` |
| `GET` | `/voices` | — | `{voices: [{id, name}], default_rate}` |
| `GET` | `/health` | — | LLM / database / document-index status |
| `GET` | `/` | — | the web UI |

`/ask-audio` returns `application/x-ndjson`, one JSON object per line, so the
UI can render each stage as it completes rather than waiting for the whole
turn:

```
{"event":"transcript","transcript":"how many students are in semester 6","stt_ms":1402}
{"event":"final","answer":"There are 15 students in semester 6.","tool":"query_database",...}
{"event":"audio","audio_base64":"UklGR...","audio_mime":"audio/wav","tts_ms":267}
```

## Run with Docker

One command for anyone cloning the repo:

```bash
cp .env.example .env      # paste your NVIDIA key and a JWT_SECRET
docker compose up --build
```

Then open <http://127.0.0.1:8000>. The port is bound to `127.0.0.1` on purpose:
browsers only grant microphone access on `localhost` or HTTPS, and this app
carries no TLS of its own.

The build bakes in the whisper and MiniLM models and seeds the database, so the
container works immediately. Two things worth knowing: **the image is large
(~3 GB)**, almost entirely PyTorch, and `espeak-ng` is installed because
`pyttsx3` has no voice at all on Linux without it. Accounts, reminders and the
vector store live in a named volume, so they survive a rebuild — but set
`JWT_SECRET` in `.env`, or a fresh random one is generated per container and
every login is invalidated on restart.

## Public demo (Hugging Face Space)

The app runs as a Docker Space on Hugging Face's free CPU tier. Every push to
`main` redeploys it through `.github/workflows/deploy-space.yml`, which swaps
in `hf/README.md` (the Space needs its own metadata header).

A public demo needs a few safeguards a local install does not, so
`app/demo.py` switches them on automatically inside a Space (or with
`DEMO_MODE=1`):

| Safeguard | Why |
| --- | --- |
| 15 questions per visitor every 10 minutes | one person or bot cannot use up the shared free LLM key |
| 12 questions per minute site-wide | each question is about three LLM calls, so this stays under NIM's ~40 requests/minute |
| Shared demo logins cannot be locked out | five wrong passwords from a stranger would otherwise lock `CS06020` for everyone |
| Logging out of a shared demo login only drops your own token | revoking every token would log out every other visitor |

The limits are configurable with `DEMO_QUESTIONS_PER_10_MIN` and
`DEMO_QUESTIONS_PER_MINUTE`. When a limit is hit, the assistant replies with a
friendly "please wait" message instead of an error.

**Setup:** create a Docker Space on Hugging Face, add `NVIDIA_API_KEY` and
`JWT_SECRET` as Space secrets, then add a write token as the GitHub secret
`HF_TOKEN` and the Space id (`username/space-name`) as the GitHub variable
`HF_SPACE`.

## Tests

```powershell
pytest tests -q          # 115 tests, ~14s, no network or API key needed
```

The LLM is stubbed, so the suite runs offline. It covers the SQL guards
(injection, DDL, stacked statements, real read-only enforcement, the row cap),
defensive router parsing, the SQL repair-retry loop, the RAG confidence
threshold, memory capping/isolation/eviction, sign-in and anonymous refusal,
timetable ordering and semester filtering, the reminder add/list/complete cycle
with per-student privacy, date-phrase parsing, and wake-phrase gating including
the false triggers it exists to prevent.

`test_auth.py` and `test_rls.py` are the security-critical ones: password
hashing and salting, the no-enumeration property, lockout, token
tamper/forge/expiry, logout revocation, and — most importantly — the adversarial
queries a signed-in student must not be able to answer. A regression in row-level
isolation fails *silently*: the assistant keeps working, it just answers with
somebody else's data. Every bug found while building this project has a test here.

---

## Troubleshooting

**`NVIDIA_API_KEY is not set`** — you skipped step 3, or the `.env` file is not
in the `voice-assistant` folder next to `run.bat`.

**Answers say "I couldn't answer that from the data"** — usually a wrong or
non-serving model string. Check `/health` in a browser: it reports whether the
LLM call actually succeeds.

**Every request hangs for ~60 s then fails** — the model string is listed but
not serving (this is exactly what `meta/llama-3.3-70b-instruct` did in
testing). Confirm the key itself is fine by checking that a deliberately wrong
key returns `403` quickly, then switch `LLM_MODEL` to
`meta/llama-3.1-70b-instruct`.

**`ERROR: Could not find a version that satisfies the requirement torch` (or
`ctranslate2`)** — you are on an ARM64 Python. See the note in
[Requirements](#requirements): install x64 Python 3.11.

**`Failed building wheel for httptools` / `grpcio`** — same cause as above, an
ARM64 Python trying to compile packages that ship no wheels for it.

**Microphone button does nothing** — the browser blocked the permission, or the
page is not on `127.0.0.1`/`localhost`. The typed input always works as a
fallback.

**Hands-free never responds** — open the browser console. On start it logs the
measured noise floor and threshold:
`[hands-free] noise floor 0.0078 threshold 0.0300`. Then watch the level meter
while you speak: if the bar barely moves, your microphone gain is too low for
the threshold. The three constants to adjust are at the top of the hands-free
block in [static/index.html](static/index.html):

| Constant | Meaning | Symptom if wrong |
|---|---|---|
| `hfThreshold()` floor (`0.030`) | how loud counts as speech | too high: never triggers · too low: triggers on room tone |
| `HF_STOP_FRAMES` (17 ≈ 1s) | silence that ends an utterance | too low: cuts you off mid-sentence |
| `HF_MIN_SPEECH_MS` (350) | shortest clip worth transcribing | too high: short questions dropped |

**Hands-free hears me but always ignores me** — the wake phrase has to *start*
the transcript. Check the greyed-out text to see what it actually heard; if it
says "Campus, what's…" the opening word was clipped. Either say "Hey Campus"
slightly more deliberately, or untick the wake-phrase requirement.

**Testing hands-free without a microphone**

```powershell
node tests\hands_free_harness.js static\index.html
```

Runs the real frontend script against fake Web Audio, MediaRecorder and
`fetch`, driving the endpointing state machine with simulated microphone
levels and a virtual clock. It checks that recording starts *before* speech is
detected (so the wake phrase is never clipped), that an utterance produces
exactly one clip, that coughs and room tone are ignored, and that the assistant
does not trigger on its own voice. Needs node; the Python suite does not.

**No audio plays but text appears** — browsers block autoplay until you have
interacted with the page; click anywhere and ask again. The answer text is
always shown regardless.

**Documents index is empty** — delete `data\chroma` and restart the server to
force a re-ingest.

**Rate limited** — the free NIM tier allows roughly 40 requests per minute, and
each question costs two calls (router + composition). The client backs off and
retries automatically; just ask more slowly.
