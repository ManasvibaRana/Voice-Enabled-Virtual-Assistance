"""Speech to text with faster-whisper, running locally on CPU.

The browser's MediaRecorder produces WebM/Opus; faster-whisper decodes it via
PyAV, so the bytes are simply written to a temp file and handed over.
"""

import os
import tempfile
import threading
from typing import Any

from app import config

_model: Any = None
_lock = threading.Lock()


def get_model():
    """Load the whisper model once. First call downloads ~75MB."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from faster_whisper import WhisperModel

                print(
                    f"[stt] loading faster-whisper '{config.WHISPER_MODEL_SIZE}' "
                    f"({config.WHISPER_DEVICE}/{config.WHISPER_COMPUTE_TYPE}) ..."
                )
                _model = WhisperModel(
                    config.WHISPER_MODEL_SIZE,
                    device=config.WHISPER_DEVICE,
                    compute_type=config.WHISPER_COMPUTE_TYPE,
                )
                print("[stt] whisper ready")
    return _model


def transcribe(
    audio_bytes: bytes, suffix: str = ".webm", language: str | None = None
) -> tuple[str, str, str]:
    """Transcribe raw audio bytes. Returns (text, detected_language, error).

    `language` is an ISO code such as "en" or "hi". Pass None (the default) to
    let Whisper detect it, which is what makes mixed English/Hindi work.
    """
    if not audio_bytes:
        return "", "", "empty audio upload"

    lang = language or config.WHISPER_LANGUAGE  # None => auto-detect

    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(audio_bytes)
            path = handle.name

        model = get_model()
        segments, info = model.transcribe(
            path,
            language=lang,
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        detected = getattr(info, "language", "") or (lang or "")
        return text, detected, ""
    except Exception as exc:  # noqa: BLE001 - never crash the request
        print(f"[stt] transcription failed: {exc!r}")
        return "", "", str(exc)
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
