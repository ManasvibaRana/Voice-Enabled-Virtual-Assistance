"""Thin wrapper around the NVIDIA NIM chat endpoint.

NIM speaks the OpenAI wire protocol, so we use the `openai` Python package
purely as a client and point it at https://integrate.api.nvidia.com/v1.
No OpenAI account or key is involved.

Every public function here is total: it never raises. On failure it returns a
sentinel (empty string / fallback dict) so the caller can degrade gracefully.
"""

import json
import re
import time
from typing import Any

from openai import OpenAI

from app import config

_client: OpenAI | None = None


class LLMError(RuntimeError):
    """Raised internally; never escapes the module boundary."""


def get_client() -> OpenAI:
    """Lazily build the shared client so import never fails on a missing key."""
    global _client
    if _client is None:
        if not config.NVIDIA_API_KEY:
            raise LLMError(
                "NVIDIA_API_KEY is not set. Copy .env.example to .env and paste "
                "your free key from https://build.nvidia.com"
            )
        _client = OpenAI(
            base_url=config.NVIDIA_BASE_URL,
            api_key=config.NVIDIA_API_KEY,
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=0,  # we do our own backoff so we can log it
        )
    return _client


def _is_rate_limit(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    return "429" in str(exc) or "rate limit" in str(exc).lower()


def chat(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 512,
    model: str | None = None,
) -> str:
    """Run a chat completion and return the assistant text.

    Returns "" if the call ultimately fails. The free NIM tier is capped at
    ~40 requests/minute, so 429s get an exponential backoff.
    """
    last_error: Exception | None = None
    made = 0

    for attempt in range(config.LLM_MAX_RETRIES):
        made += 1
        try:
            client = get_client()
            resp = client.chat.completions.create(
                model=model or config.LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = (resp.choices[0].message.content or "").strip()
            return content
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all
            last_error = exc
            if isinstance(exc, LLMError):
                break  # missing key: retrying will not help
            wait = config.LLM_BACKOFF_SECONDS * (2**attempt)
            if _is_rate_limit(exc):
                print(f"[llm] rate limited (429), retrying in {wait:.0f}s")
            else:
                print(f"[llm] call failed ({exc!r}), retrying in {wait:.0f}s")
            if attempt < config.LLM_MAX_RETRIES - 1:
                time.sleep(wait)

    print(f"[llm] giving up after {made} attempt(s): {last_error!r}")
    return ""


def chat_stream(
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 512,
    model: str | None = None,
):
    """Yield the reply in pieces as the model produces it.

    Same failure contract as `chat`: on an unrecoverable error it simply stops
    yielding, so a caller that accumulates the pieces ends up with "" and can
    fall back exactly as before. Retries only happen before the first token -
    once text has been handed to the caller it cannot be taken back, so a
    mid-stream failure ends the stream rather than restarting it.
    """
    last_error: Exception | None = None

    for attempt in range(config.LLM_MAX_RETRIES):
        produced_any = False
        try:
            client = get_client()
            stream = client.chat.completions.create(
                model=model or config.LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content or ""
                if piece:
                    produced_any = True
                    yield piece
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if produced_any:
                print(f"[llm] stream broke after partial output: {exc!r}")
                return
            if isinstance(exc, LLMError):
                break
            wait = config.LLM_BACKOFF_SECONDS * (2**attempt)
            if _is_rate_limit(exc):
                print(f"[llm] rate limited (429), retrying stream in {wait:.0f}s")
            else:
                print(f"[llm] stream failed ({exc!r}), retrying in {wait:.0f}s")
            if attempt < config.LLM_MAX_RETRIES - 1:
                time.sleep(wait)

    print(f"[llm] stream gave up: {last_error!r}")


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*|\s*```$")


def strip_fences(text: str) -> str:
    """Remove markdown code fences that models add despite being told not to."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        # Drop the opening fence line and any trailing fence.
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines.pop()
        cleaned = "\n".join(lines)
    return _FENCE_RE.sub("", cleaned).strip()


def chat_json(
    messages: list[dict[str, str]],
    fallback: dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int = 256,
    model: str | None = None,
) -> dict[str, Any]:
    """Chat and parse the reply as a JSON object, falling back on any failure."""
    raw = chat(messages, temperature=temperature, max_tokens=max_tokens, model=model)
    if not raw:
        return dict(fallback)

    candidate = strip_fences(raw)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Second chance: grab the first {...} block in the text.
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    print(f"[llm] could not parse JSON from: {raw[:200]!r}")
    return dict(fallback)


def health() -> tuple[bool, str]:
    """Cheap liveness probe used by the UI banner."""
    if not config.NVIDIA_API_KEY:
        return False, "NVIDIA_API_KEY is not set (see .env.example)"
    reply = chat(
        [{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=8,
    )
    if reply:
        return True, f"connected to {config.LLM_MODEL}"
    return False, f"could not reach NVIDIA NIM with model {config.LLM_MODEL}"
