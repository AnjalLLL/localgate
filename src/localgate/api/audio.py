"""``/v1/audio/*`` — local speech, OpenAI-compatible where OpenAI has an equivalent.

* ``POST /v1/audio/speech`` — text → WAV with Piper. Body: ``{"input": "...",
  "voice": "en_US-lessac-medium"}`` (``model``/``response_format`` accepted for SDK
  compatibility; the output is always WAV).
* ``POST /v1/audio/transcriptions`` — audio → ``{"text": ...}`` with faster-whisper.
  Accepts the OpenAI SDK's multipart upload (``file`` field) or a raw audio body.
* ``GET /v1/audio/voices`` — what is installed and available.

Everything runs on this machine; audio is never stored. Both need the
``localgate[voice]`` extra — without it these routes answer 501 with the install
command, rather than failing in an unhelpful way.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field

from localgate.api.deps import enforce_rate_limit
from localgate.core.errors import InvalidRequestError, LocalgateError
from localgate.db.models import APIKey
from localgate.voice import INSTALL_HINT, VoiceUnavailable

router = APIRouter(tags=["audio"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # OpenAI's own transcription limit


class VoiceNotAvailable(LocalgateError):
    status_code = 501
    error_type = "not_implemented_error"


def _settings(request: Request) -> Any:
    settings = request.app.state.settings
    if not settings.voice_enabled:
        raise VoiceNotAvailable(
            "Voice is disabled on this server (LOCALGATE_VOICE_ENABLED).", code="voice_disabled"
        )
    return settings


def _tts(request: Request) -> Any:
    from localgate.voice.tts import PiperTTS, voices_dir  # noqa: PLC0415

    settings = _settings(request)
    engine = getattr(request.app.state, "tts", None)
    if engine is None:
        engine = PiperTTS(
            voices_dir(settings),
            settings.voice_tts_voice,
            download=settings.voice_auto_download,
        )
        request.app.state.tts = engine
    return engine


def _stt(request: Request) -> Any:
    from localgate.voice import stt  # noqa: PLC0415

    settings = _settings(request)
    engine = getattr(request.app.state, "stt", None)
    if engine is None:
        if not stt.available():
            raise VoiceNotAvailable(
                f"Speech recognition is not installed. {INSTALL_HINT}", code="voice_not_installed"
            )
        engine = stt.WhisperSTT(settings.voice_stt_model, settings.voice_stt_device)
        request.app.state.stt = engine
    return engine


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=5000)
    voice: str | None = None
    model: str | None = None  # accepted for OpenAI SDK compatibility; Piper has one engine
    response_format: str | None = None


@router.post("/v1/audio/speech")
async def speech(
    body: SpeechRequest,
    request: Request,
    _key: APIKey = Depends(enforce_rate_limit),
) -> Response:
    if body.response_format not in (None, "wav"):
        raise InvalidRequestError(
            "Only response_format 'wav' is supported.", code="unsupported_format"
        )
    engine = _tts(request)
    # OpenAI voice names ("alloy", ...) mean nothing to Piper: use the default.
    voice = body.voice if body.voice and "-" in body.voice else None
    try:
        audio = await engine.synthesize(body.input, voice)
    except VoiceUnavailable as exc:
        raise VoiceNotAvailable(str(exc), code="voice_not_installed") from exc
    except ValueError as exc:
        raise InvalidRequestError(str(exc), code="invalid_input") from exc
    return Response(audio, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@router.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    _key: APIKey = Depends(enforce_rate_limit),
) -> dict:
    engine = _stt(request)
    language: str | None = None
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        try:
            form = await request.form()
        except AssertionError as exc:  # starlette: python-multipart missing
            raise VoiceNotAvailable(
                f"Multipart uploads need python-multipart. {INSTALL_HINT}"
            ) from exc
        upload: Any = form.get("file")
        if upload is None or isinstance(upload, str):
            raise InvalidRequestError("Send the audio as the 'file' field.", code="missing_file")
        data = await upload.read(MAX_UPLOAD_BYTES + 1)
        lang = form.get("language")
        language = lang if isinstance(lang, str) and lang else None
    else:
        data = await request.body()
        language = request.query_params.get("language") or None
    if not data:
        raise InvalidRequestError("No audio received.", code="missing_file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidRequestError("Audio is larger than 25 MB.", code="file_too_large")
    try:
        result = await engine.transcribe(data, language)
    except VoiceUnavailable as exc:
        raise VoiceNotAvailable(str(exc), code="voice_not_installed") from exc
    except ValueError as exc:
        raise InvalidRequestError(str(exc), code="invalid_audio") from exc
    return {"text": result.text, "language": result.language, "duration": round(result.duration, 2)}


@router.get("/v1/audio/voices")
async def voices(request: Request, _key: APIKey = Depends(enforce_rate_limit)) -> dict:
    from localgate.voice import stt  # noqa: PLC0415
    from localgate.voice.tts import engine_name, installed_voices, voices_dir  # noqa: PLC0415

    settings = request.app.state.settings
    return {
        "enabled": settings.voice_enabled,
        "tts_engine": engine_name(),
        "stt_available": stt.available(),
        "default_voice": settings.voice_tts_voice,
        "installed": installed_voices(voices_dir(settings)),
        "stt_model": settings.voice_stt_model,
    }
