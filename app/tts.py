"""Text to speech, offline.

Ships with pyttsx3 (SAPI5 on Windows, NSSpeechSynthesizer on macOS, espeak on
Linux). The engine sits behind a tiny `TTSEngine` interface so it can be
swapped for Piper later without touching any caller: implement `synthesize`
and register the class in `_ENGINES`, then set TTS_ENGINE in .env.

Synthesis runs on a dedicated short-lived thread. pyttsx3's SAPI5 driver talks
to COM, which is apartment-threaded, so a fresh engine on a fresh thread is far
more reliable inside a web server than a long-lived shared one.
"""

import os
import queue
import tempfile
import threading
from typing import Protocol

ENGINE_NAME = os.getenv("TTS_ENGINE", "pyttsx3").strip().lower()

# Guards against two SAPI voices talking over each other.
_synthesis_lock = threading.Lock()


DEFAULT_RATE = 175
MIN_RATE, MAX_RATE = 80, 320


class TTSEngine(Protocol):
    def synthesize(
        self, text: str, voice: str | None = None, rate: int | None = None
    ) -> tuple[bytes, str]:
        """Return (wav_bytes, error). Must never raise."""

    def voices(self) -> list[dict[str, str]]:
        """Return [{'id': ..., 'name': ...}] of installed voices."""


def _run_on_com_thread(fn, timeout: int = 60):
    """Run `fn` on a fresh COM-initialised thread and return (result, error).

    SAPI5 is apartment-threaded, so a short-lived thread with its own COM
    apartment is far more reliable inside a web server than a shared engine.
    """
    out: queue.Queue = queue.Queue(maxsize=1)

    def worker() -> None:
        com_initialised = False
        try:
            try:
                import comtypes

                comtypes.CoInitialize()
                com_initialised = True
            except Exception:  # noqa: BLE001 - non-Windows or no comtypes
                pass
            out.put((fn(), ""))
        except Exception as exc:  # noqa: BLE001
            out.put((None, str(exc)))
        finally:
            if com_initialised:
                try:
                    import comtypes

                    comtypes.CoUninitialize()
                except Exception:  # noqa: BLE001
                    pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        return None, "speech engine timed out"
    try:
        return out.get_nowait()
    except queue.Empty:
        return None, "speech engine produced no result"


class Pyttsx3Engine:
    """Offline synthesis via the OS speech engine."""

    name = "pyttsx3"

    def voices(self) -> list[dict[str, str]]:
        def work() -> list[dict[str, str]]:
            import pyttsx3

            engine = pyttsx3.init()
            found = []
            for v in engine.getProperty("voices") or []:
                name = getattr(v, "name", "") or getattr(v, "id", "")
                found.append({"id": getattr(v, "id", ""), "name": name})
            try:
                engine.stop()
            except Exception:  # noqa: BLE001
                pass
            return found

        with _synthesis_lock:
            result, error = _run_on_com_thread(work, timeout=30)
        if error:
            print(f"[tts] could not list voices: {error}")
        return result or []

    def synthesize(
        self, text: str, voice: str | None = None, rate: int | None = None
    ) -> tuple[bytes, str]:
        if not text or not text.strip():
            return b"", "no text to speak"

        speed = DEFAULT_RATE if rate is None else max(MIN_RATE, min(MAX_RATE, rate))

        def work() -> bytes:
            import pyttsx3

            path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                    path = fh.name

                engine = pyttsx3.init()
                engine.setProperty("rate", speed)
                if voice:
                    try:
                        engine.setProperty("voice", voice)
                    except Exception:  # noqa: BLE001 - unknown id, keep default
                        print(f"[tts] voice {voice!r} not available, using default")
                engine.save_to_file(text, path)
                engine.runAndWait()
                try:
                    engine.stop()
                except Exception:  # noqa: BLE001
                    pass

                with open(path, "rb") as fh:
                    return fh.read()
            finally:
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        with _synthesis_lock:
            audio, error = _run_on_com_thread(work, timeout=60)

        if error:
            print(f"[tts] synthesis failed: {error}")
            return b"", error
        if not audio:
            return b"", "the speech engine produced an empty file"
        return audio, ""


_ENGINES: dict[str, type] = {
    "pyttsx3": Pyttsx3Engine,
    # To add Piper: implement PiperEngine.synthesize(text) -> (wav_bytes, error)
    # and register it here as "piper", then set TTS_ENGINE=piper in .env.
}

_engine: TTSEngine | None = None


def get_engine() -> TTSEngine:
    global _engine
    if _engine is None:
        cls = _ENGINES.get(ENGINE_NAME, Pyttsx3Engine)
        _engine = cls()
    return _engine


def speak(
    text: str, voice: str | None = None, rate: int | None = None
) -> tuple[bytes, str]:
    """Synthesize `text` to WAV bytes. Returns (wav_bytes, error)."""
    try:
        return get_engine().synthesize(text, voice=voice, rate=rate)
    except Exception as exc:  # noqa: BLE001
        print(f"[tts] engine error: {exc!r}")
        return b"", str(exc)


def list_voices() -> list[dict[str, str]]:
    """Installed voices for the settings picker. Never raises."""
    try:
        return get_engine().voices()
    except Exception as exc:  # noqa: BLE001
        print(f"[tts] engine error listing voices: {exc!r}")
        return []
