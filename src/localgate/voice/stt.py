"""Speech-to-text with faster-whisper (local Whisper models, CPU or GPU).

Audio is decoded here with PyAV (which faster-whisper already depends on), into
16 kHz mono float32 samples, rather than by faster-whisper itself: its own
decoder breaks on recent PyAV releases, and decoding ourselves accepts whatever a
browser records (WebM/Opus in Chromium and Firefox, MP4/AAC in Safari) as well as
WAV, MP3 and the raw samples the CLI records.

Model names are faster-whisper's: ``tiny``, ``base``, ``small``, ``medium``,
``large-v3``, ``turbo`` and their English-only ``.en`` variants. The model is
downloaded from Hugging Face on first use and cached.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import threading
from dataclasses import dataclass
from typing import Any

from localgate.voice import INSTALL_HINT, VoiceUnavailable

SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 10 * 60


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None
    duration: float


def available() -> bool:
    try:
        import av  # noqa: F401, PLC0415
        import faster_whisper  # noqa: F401, PLC0415

        importlib.import_module("numpy")
    except ImportError:
        return False
    return True


def decode_audio(data: bytes) -> Any:
    """Any audio container/codec PyAV understands → 16 kHz mono float32 array."""
    try:
        import av  # noqa: PLC0415

        np = importlib.import_module("numpy")
    except ImportError as exc:
        raise VoiceUnavailable(f"Speech recognition is not installed. {INSTALL_HINT}") from exc
    chunks: list[Any] = []
    total = 0
    try:
        with av.open(io.BytesIO(data)) as container:
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    samples = out.to_ndarray().reshape(-1)
                    chunks.append(samples)
                    total += len(samples)
                if total > MAX_AUDIO_SECONDS * SAMPLE_RATE:
                    break
            for out in resampler.resample(None):
                chunks.append(out.to_ndarray().reshape(-1))
    except Exception as exc:  # noqa: BLE001 — PyAV raises many error types for bad input
        raise ValueError(f"couldn't decode the audio: {exc}") from exc
    if not chunks:
        raise ValueError("the audio contained no samples")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


class WhisperSTT:
    """Lazily loads one faster-whisper model and transcribes with it."""

    def __init__(self, model: str = "base", device: str = "auto", compute_type: str = "auto"):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel  # noqa: PLC0415
                except ImportError as exc:
                    raise VoiceUnavailable(
                        f"Speech recognition is not installed. {INSTALL_HINT}"
                    ) from exc
                compute = self.compute_type
                if compute == "auto":
                    compute = "int8"  # fast and small on every CPU; GPUs accept it too
                self._model = WhisperModel(
                    self.model_name, device=self.device, compute_type=compute
                )
            return self._model

    def transcribe_samples(self, samples: Any, language: str | None = None) -> Transcript:
        model = self._load()
        segments, info = model.transcribe(
            samples, language=language or None, beam_size=1, vad_filter=True
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return Transcript(
            text=text, language=getattr(info, "language", None), duration=len(samples) / SAMPLE_RATE
        )

    def transcribe_sync(self, data: bytes, language: str | None = None) -> Transcript:
        return self.transcribe_samples(decode_audio(data), language)

    async def transcribe(self, data: bytes, language: str | None = None) -> Transcript:
        return await asyncio.to_thread(self.transcribe_sync, data, language)
