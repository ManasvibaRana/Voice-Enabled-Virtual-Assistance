"""FastAPI application: endpoints, startup ingestion, static frontend."""

import base64
import json
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Allow `python app/main.py` as well as `uvicorn app.main:app`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import (
    auth,
    compose,
    config,
    demo,
    identity,
    llm,
    memory,
    rag_tool,
    reminder_tool,
    router,
    smalltalk,
    sql_tool,
    stt,
    timetable_tool,
    tts,
)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def _ensure_database() -> None:
    """Seed the SQLite database if it is missing, so first run just works."""
    if config.DB_PATH.exists():
        return
    print("[startup] app.db not found, seeding it now ...")
    sys.path.insert(0, str(config.DATA_DIR))
    try:
        import seed_db

        seed_db.main()
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] automatic seeding failed: {exc!r}")
        print("[startup] run:  python data/seed_db.py")


def _preload_models() -> None:
    """Load whisper in the background so the first voice question is not slow.

    Loading the model costs about 6 seconds. Doing it at startup rather than on
    the first request keeps that cost off the user's first question.
    """
    try:
        stt.get_model()
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] whisper preload failed (will retry on demand): {exc!r}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("[startup] voice assistant starting")
    _ensure_database()
    try:
        rag_tool.ingest_docs()
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] document ingestion failed: {exc!r}")
    threading.Thread(target=_preload_models, daemon=True).start()
    if not config.NVIDIA_API_KEY:
        print(
            "[startup] WARNING: NVIDIA_API_KEY is not set. Copy .env.example to "
            ".env and paste your free key from https://build.nvidia.com"
        )
    print(f"[startup] ready -> http://127.0.0.1:8000  (model: {config.LLM_MODEL})")
    yield
    print("[shutdown] bye")


app = FastAPI(title="Voice Assistant", version="1.0.0", lifespan=lifespan)

# Long-lived pool for speculative tool execution. Deliberately NOT a context
# manager: when the router picks one tool we abandon the other future rather
# than waiting for it, which is the whole point of running them concurrently.
_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class AskTextRequest(BaseModel):
    question: str = Field(default="", max_length=2000)
    voice: str | None = Field(default=None, max_length=400)
    rate: int | None = Field(default=None, ge=80, le=320)
    speak: bool = Field(default=False)


class LoginRequest(BaseModel):
    username: str = Field(default="", max_length=128)
    password: str = Field(default="", max_length=200)


class SignupRequest(BaseModel):
    roll_no: str = Field(default="", max_length=64)
    name: str = Field(default="", max_length=128)
    password: str = Field(default="", max_length=200)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def _principal(authorization: str | None) -> identity.Principal | None:
    """Resolve `Authorization: Bearer <jwt>` to an authenticated principal."""
    return identity.principal_for(auth.user_from_header(authorization))


def _memory_key(principal: identity.Principal | None) -> str:
    """Conversation history is keyed by account, never by a client string."""
    return f"user:{principal['user']['id']}" if principal else ""


UNAUTHORIZED = JSONResponse(
    {"error": "Sign in to use the assistant."}, status_code=401
)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def _settle(future, fallback_fn, *args):
    """Take the speculative future's result, or run the tool now if there is none."""
    if future is None:
        return fallback_fn(*args)
    try:
        return future.result()
    except Exception as exc:  # noqa: BLE001 - a crashed future must not 500
        print(f"[pipeline] speculative {fallback_fn.__name__} failed: {exc!r}")
        return fallback_fn(*args)


def turn_events(
    principal: identity.Principal | None, question: str
) -> Iterator[dict[str, Any]]:
    """Run one turn, emitting progress as it happens.

    Yields `stage` events as each part of the pipeline finishes, `delta`
    events carrying the answer as the model writes it, and finally a `final`
    event with the complete payload. The UI uses the stages to light up its
    pipeline strip and the deltas to type the answer out live; a scripted
    client can ignore everything except `final`.
    """
    memory_key = _memory_key(principal)
    question = (question or "").strip()
    if not question:
        yield {
            "event": "final",
            "answer": "I didn't catch a question. Could you say that again?",
            "tool": "none",
            "reason": "Empty question.",
            "error": "empty question",
        }
        return

    timings: dict[str, int] = {}

    f_sql = f_rag = None
    started = time.perf_counter()
    if config.SPECULATIVE_TOOLS:
        f_sql = _pool.submit(sql_tool.run, question, principal, memory_key)
        f_rag = _pool.submit(rag_tool.retrieve, question)

    decision = router.route(question, principal, memory_key)
    tool = decision["tool"]
    timings["router_ms"] = int((time.perf_counter() - started) * 1000)
    yield {
        "event": "stage",
        "stage": "routed",
        "ms": timings["router_ms"],
        "tool": tool,
        "reason": decision["reason"],
    }

    payload: dict[str, Any] = {
        "tool": tool,
        "reason": decision["reason"],
        "sql": None,
        "columns": None,
        "rows": None,
        "row_count": None,
        "chunks": None,
        "attempts": None,
        "classes": None,
        "reminders": None,
        "library": None,
        "detail": None,
        "error": "",
    }

    tool_started = time.perf_counter()
    personal_only = identity.student_scope(principal) is not None
    stream = None
    fixed_answer = None

    if tool == "query_database":
        result = _settle(f_sql, sql_tool.run, question, principal, memory_key)
        payload.update(
            sql=result["sql"], columns=result["columns"], rows=result["rows"],
            row_count=result["row_count"], attempts=result["attempts"],
            error=result["error"],
        )
        stream = compose.compose_from_sql_stream(
            question, result, memory_key, personal_only
        )

    elif tool == "timetable":
        result = timetable_tool.run(question, principal)
        rendered = timetable_tool.as_text(result)
        payload.update(classes=result["classes"], detail=rendered, error=result["error"])
        if result["ok"]:
            stream = compose.compose_from_tool_stream(question, rendered, memory_key)
        else:
            fixed_answer = result["error"]

    elif tool == "reminders":
        result = reminder_tool.run(question, principal, memory_key=memory_key)
        rendered = reminder_tool.as_text(result)
        payload.update(
            reminders=result["reminders"], library=result.get("library"),
            detail=rendered,
            error="" if result["ok"] else result["error"],
        )
        if result["ok"]:
            stream = compose.compose_from_tool_stream(question, rendered, memory_key)
        else:
            fixed_answer = result["error"]

    elif tool == "smalltalk":
        fixed_answer = smalltalk.answer(question, principal, memory_key)

    else:
        result = _settle(f_rag, rag_tool.retrieve, question)
        payload.update(chunks=result["chunks"], error=result["error"])
        stream = compose.compose_from_rag_stream(question, result, memory_key)

    timings["tool_ms"] = int((time.perf_counter() - tool_started) * 1000)
    yield {"event": "stage", "stage": "queried", "ms": timings["tool_ms"]}

    composing = time.perf_counter()
    if stream is not None:
        try:
            while True:
                yield {"event": "delta", "text": next(stream)}
        except StopIteration as done:
            answer = done.value
        except Exception as exc:  # noqa: BLE001 - a broken stream must not 500
            print(f"[pipeline] answer stream failed: {exc!r}")
            answer = compose.FALLBACK_DB
    else:
        answer = fixed_answer or compose.FALLBACK_TOOL

    timings["compose_ms"] = int((time.perf_counter() - composing) * 1000)
    yield {"event": "stage", "stage": "composed", "ms": timings["compose_ms"]}

    payload["answer"] = answer
    payload["timings"] = timings
    payload["event"] = "final"
    memory.add_turn(memory_key, question, answer)
    yield payload


def handle_question(
    principal: identity.Principal | None, question: str
) -> dict[str, Any]:
    """Buffered form of `turn_events` - drains the stream and returns the
    final payload. Kept so callers that do not care about progress (and the
    tests) have a simple, synchronous entry point."""
    payload: dict[str, Any] = {}
    for event in turn_events(principal, question):
        if event.get("event") == "final":
            payload = {k: v for k, v in event.items() if k != "event"}
    return payload


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/auth/signup")
def auth_signup(req: SignupRequest) -> JSONResponse:
    user, error = auth.signup(req.roll_no, req.name, req.password)
    if user is None:
        return JSONResponse({"error": error}, status_code=400)
    token = auth.create_token(user)
    return JSONResponse(
        {"token": token, "user": identity.describe(identity.principal_for(user))}
    )


@app.post("/auth/login")
def auth_login(req: LoginRequest) -> JSONResponse:
    token, user, error = auth.login(req.username, req.password)
    if user is None:
        return JSONResponse({"error": error}, status_code=401)
    return JSONResponse(
        {"token": token, "user": identity.describe(identity.principal_for(user))}
    )


@app.post("/auth/logout")
def auth_logout(authorization: str | None = Header(default=None)) -> JSONResponse:
    principal = _principal(authorization)
    if principal is None:
        return JSONResponse({"ok": True})
    # Bumping token_version invalidates every token already issued to them.
    auth.logout(principal["user"]["id"])
    memory.clear(_memory_key(principal))
    return JSONResponse({"ok": True})


@app.get("/auth/me")
def auth_me(authorization: str | None = Header(default=None)) -> JSONResponse:
    principal = _principal(authorization)
    if principal is None:
        return UNAUTHORIZED
    return JSONResponse({"user": identity.describe(principal)})


def _ndjson(obj: dict[str, Any]) -> str:
    return json.dumps(obj, default=str) + "\n"


NDJSON_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _demo_limited(request: Request, *, audio: bool) -> StreamingResponse | None:
    """In public-demo mode, answer with a friendly "please wait" instead of
    running the pipeline once a visitor (or the whole site) is over its limit.
    Sent as a normal `final` event so the UI shows it like any other reply."""
    message = demo.check_question(
        demo.client_ip(request.headers, request.client.host if request.client else None)
    )
    if message is None:
        return None

    def events() -> Iterator[str]:
        if audio:
            yield _ndjson({"event": "transcript", "transcript": ""})
        yield _ndjson({"event": "final", "answer": message, "tool": "none", "reason": "Demo rate limit."})

    return StreamingResponse(events(), media_type="application/x-ndjson", headers=NDJSON_HEADERS)


@app.post("/ask-text", response_model=None)
def ask_text(
    req: AskTextRequest, request: Request, authorization: str | None = Header(default=None)
) -> StreamingResponse | JSONResponse:
    """Streams newline-delimited JSON, same event shape as /ask-audio.

    A client that only wants the answer can read the last line: the `final`
    event carries the complete payload.
    """
    principal = _principal(authorization)
    if principal is None:
        return UNAUTHORIZED
    limited = _demo_limited(request, audio=False)
    if limited is not None:
        return limited

    def events() -> Iterator[str]:
        answer = ""
        try:
            for event in turn_events(principal, req.question):
                if event.get("event") == "final":
                    answer = event.get("answer", "")
                yield _ndjson(event)
        except Exception as exc:  # noqa: BLE001 - last-resort guard
            print(f"[/ask-text] unexpected failure: {exc!r}")
            yield _ndjson({
                "event": "final",
                "answer": "Something went wrong on my side. Please try again.",
                "tool": "none", "reason": "Internal error.", "error": str(exc),
            })
            return

        if req.speak and answer:
            spoken = time.perf_counter()
            audio = _speak_b64(answer, req.voice, req.rate)
            ms = int((time.perf_counter() - spoken) * 1000)
            yield _ndjson({"event": "stage", "stage": "spoke", "ms": ms})
            yield _ndjson({
                "event": "audio", "audio_base64": audio,
                "audio_mime": "audio/wav", "tts_ms": ms,
            })

    return StreamingResponse(
        events(), media_type="application/x-ndjson", headers=NDJSON_HEADERS
    )


@app.get("/voices")
def voices() -> JSONResponse:
    return JSONResponse(
        {"voices": tts.list_voices(), "default_rate": config.TTS_DEFAULT_RATE}
    )


# /students and /sign-in are deliberately gone. /students listed every student
# in the college, which is itself a data leak, and /sign-in let any caller
# claim to be any student. Identity now comes only from a verified token.


@app.post("/ask-audio", response_model=None)
async def ask_audio(
    request: Request,
    voice: str = Form(""),
    rate: int = Form(0),
    language: str = Form(""),
    audio: UploadFile = File(...),
    authorization: str | None = Header(default=None),
) -> StreamingResponse | JSONResponse:
    principal = _principal(authorization)
    if principal is None:
        return UNAUTHORIZED
    limited = _demo_limited(request, audio=True)
    if limited is not None:
        return limited
    """Stream newline-delimited JSON: the transcript first, the answer second.

    Transcription takes about a second but the rest of the pipeline takes
    several more. Emitting the transcript as soon as it exists lets the UI show
    what it heard immediately instead of leaving the page blank until the whole
    turn finishes.
    """
    try:
        raw = await audio.read()
    except Exception as exc:  # noqa: BLE001
        raw = b""
        print(f"[/ask-audio] could not read upload: {exc!r}")

    suffix = Path(audio.filename or "clip.webm").suffix or ".webm"
    chosen_voice = voice or None
    chosen_rate = rate or None

    def events() -> Iterator[str]:
        def line(obj: dict[str, Any]) -> str:
            return json.dumps(obj, default=str) + "\n"

        def say(text: str) -> str:
            return _speak_b64(text, chosen_voice, chosen_rate)

        if not raw:
            answer = "I couldn't read that recording. Please try again."
            yield line({"event": "transcript", "transcript": "", "error": "empty upload"})
            yield line(
                {
                    "event": "final",
                    "answer": answer,
                    "transcript": "",
                    "tool": "none",
                    "reason": "Upload failed.",
                    "audio_base64": say(answer),
                    "audio_mime": "audio/wav",
                    "error": "empty upload",
                }
            )
            return

        heard = time.perf_counter()
        transcript, detected, stt_error = stt.transcribe(
            raw, suffix=suffix, language=language or None
        )
        stt_ms = int((time.perf_counter() - heard) * 1000)

        question = transcript

        yield line(
            {
                "event": "transcript",
                "transcript": transcript,
                "language": detected,
                "stt_ms": stt_ms,
                "error": stt_error,
            }
        )
        yield line({"event": "stage", "stage": "heard", "ms": stt_ms})

        if not transcript:
            answer = "I couldn't hear anything in that recording. Please try again."
            yield line(
                {
                    "event": "final",
                    "answer": answer,
                    "transcript": "",
                    "tool": "none",
                    "reason": "Speech to text returned no text.",
                    "audio_base64": say(answer),
                    "audio_mime": "audio/wav",
                    "error": stt_error,
                }
            )
            return

        answer = ""
        try:
            for event in turn_events(principal, question):
                if event.get("event") == "final":
                    answer = event.get("answer", "")
                    event["transcript"] = transcript
                    event["language"] = detected
                    event.setdefault("timings", {})["stt_ms"] = stt_ms
                yield line(event)
        except Exception as exc:  # noqa: BLE001
            print(f"[/ask-audio] unexpected failure: {exc!r}")
            answer = "Something went wrong on my side. Please try again."
            yield line({
                "event": "final", "answer": answer, "transcript": transcript,
                "tool": "none", "reason": "Internal error.", "error": str(exc),
            })

        # The answer text is already on screen by now; speech is synthesised
        # afterwards so nobody waits for audio they can read.
        spoken = time.perf_counter()
        audio_b64 = say(answer)
        ms = int((time.perf_counter() - spoken) * 1000)
        yield line({"event": "stage", "stage": "spoke", "ms": ms})
        yield line(
            {
                "event": "audio",
                "audio_base64": audio_b64,
                "audio_mime": "audio/wav",
                "tts_ms": ms,
            }
        )

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _speak_b64(
    text: str, voice: str | None = None, rate: int | None = None
) -> str:
    wav, error = tts.speak(text, voice=voice, rate=rate)
    if error:
        print(f"[tts] {error}")
    return base64.b64encode(wav).decode("ascii") if wav else ""


@app.post("/clear")
def clear(authorization: str | None = Header(default=None)) -> JSONResponse:
    principal = _principal(authorization)
    if principal is None:
        return UNAUTHORIZED
    memory.clear(_memory_key(principal))
    return JSONResponse({"ok": True})


@app.get("/health")
def health() -> JSONResponse:
    ok, message = llm.health()

    db_ok, db_message = False, "app.db is missing - run python data/seed_db.py"
    if config.DB_PATH.exists():
        try:
            with sqlite3.connect(
                f"file:{config.DB_PATH.as_posix()}?mode=ro", uri=True
            ) as conn:
                count = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
            db_ok, db_message = True, f"{count} students loaded"
        except sqlite3.Error as exc:
            db_message = str(exc)

    try:
        chunks = rag_tool.get_collection().count()
        rag_message = f"{chunks} document chunks indexed"
        rag_ok = chunks > 0
    except Exception as exc:  # noqa: BLE001
        rag_ok, rag_message = False, str(exc)

    return JSONResponse(
        {
            "llm": {"ok": ok, "message": message, "model": config.LLM_MODEL},
            "database": {"ok": db_ok, "message": db_message},
            "documents": {"ok": rag_ok, "message": rag_message},
        }
    )


@app.get("/")
def index() -> FileResponse:
    # The page loads no third-party scripts at all, so a strict CSP costs
    # nothing and closes the main route by which a token in sessionStorage
    # could be stolen. 'unsafe-inline' is needed because the app is one file.
    return FileResponse(
        config.STATIC_DIR / "index.html",
        headers={
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                # Inline SVG grain/noise textures, no external requests.
                "img-src 'self' data:; "
                "media-src 'self' data: blob:; "
                "connect-src 'self'; "
                "base-uri 'none'; frame-ancestors 'none'"
            ),
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            # The whole app is one file, so a cached copy means edits appear to
            # do nothing until the user thinks to hard-refresh. Never cache it.
            "Cache-Control": "no-store, must-revalidate",
            "Pragma": "no-cache",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
