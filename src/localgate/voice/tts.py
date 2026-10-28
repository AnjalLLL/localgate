"""Text-to-speech with Piper (https://github.com/OHF-Voice/piper1-gpl).

Two ways to run it, tried in this order:

1. the ``piper-tts`` Python package (``localgate[voice]``) — voices load once and
   stay in memory, so each sentence takes a fraction of a second;
2. a ``piper`` executable on PATH (the standalone release binary, or the package's
   own CLI) — for people who prefer not to install the Python package.

Voices are ``<language>_<REGION>-<name>-<quality>`` names such as
``en_US-lessac-medium``, stored as ``.onnx`` + ``.onnx.json`` pairs in the voice
directory (``<data dir>/voices`` by default) and downloaded on first use.

Piper is GPL-3.0 licensed; it is an optional extra and is never bundled.
"""

from __future__ import annotations

import asyncio
import io
import re
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

from localgate import paths
from localgate.voice import INSTALL_HINT, VoiceUnavailable

DEFAULT_VOICE = "en_US-lessac-medium"
MAX_TTS_CHARS = 5000
VOICE_NAME_RE = re.compile(r"^[a-z]{2,3}_[A-Z]{2}-[A-Za-z0-9_]+-(x_low|low|medium|high)$")

#: espeak-ng keeps its data path in a fixed-size buffer; a longer path (deep
#: virtualenvs, Windows profiles) makes it silently fall back to a build-time
#: path and fail. Above this length the data is copied somewhere shorter.
_ESPEAK_SAFE_PATH_LEN = 120


def voices_dir(settings: Any | None = None) -> Path:
    configured = getattr(settings, "voice_dir", None)
    return Path(configured) if configured else paths.data_dir() / "voices"


def installed_voices(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(
        p.name[: -len(".onnx")]
        for p in directory.glob("*.onnx")
        if p.with_name(p.name + ".json").is_file()
    )


def _piper_package() -> Any | None:
    try:
        import piper  # noqa: PLC0415
    except ImportError:
        return None
    return piper


def _piper_executable() -> str | None:
    return shutil.which("piper")


def engine_name() -> str | None:
    """Which Piper is available: "python", "executable" or None."""
    if _piper_package() is not None:
        return "python"
    if _piper_executable() is not None:
        return "executable"
    return None


def ensure_voice(name: str, directory: Path, *, download: bool = True) -> Path:
    """Path to ``name``'s model, downloading it first if allowed and missing."""
    if not VOICE_NAME_RE.match(name):
        raise VoiceUnavailable(
            f"{name!r} is not a Piper voice name (expected e.g. {DEFAULT_VOICE!r})"
        )
    model = directory / f"{name}.onnx"
    if model.is_file() and model.with_name(model.name + ".json").is_file():
        return model
    if not download:
        raise VoiceUnavailable(
            f"Voice {name!r} is not installed. Run: localgate voice download {name}"
        )
    package = _piper_package()
    if package is None:
        raise VoiceUnavailable(
            f"Voice {name!r} is not installed and can't be downloaded without the piper "
            f"package. {INSTALL_HINT}, or put {name}.onnx and {name}.onnx.json in {directory}"
        )
    from piper.download_voices import download_voice  # noqa: PLC0415

    directory.mkdir(parents=True, exist_ok=True)
    try:
        download_voice(name, directory)
    except Exception as exc:  # noqa: BLE001 — network/HTTP errors surface as one message
        raise VoiceUnavailable(f"Couldn't download voice {name!r}: {exc}") from exc
    return model


def _espeak_data_dir() -> Path | None:
    package = _piper_package()
    if package is None:
        return None
    bundled = Path(package.__file__).parent / "espeak-ng-data"
    if not bundled.is_dir() or len(str(bundled)) <= _ESPEAK_SAFE_PATH_LEN:
        return None
    short = paths.data_dir() / "espeak-ng-data"
    if not (short / "phontab").is_file():
        short.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundled, short, dirs_exist_ok=True)
    return short


class PiperTTS:
    """Synthesizes WAV audio. Thread-safe; loaded voices are cached."""

    def __init__(
        self, directory: Path, default_voice: str = DEFAULT_VOICE, *, download: bool = True
    ):
        self.directory = directory
        self.default_voice = default_voice
        self.download = download
        self._voices: dict[str, Any] = {}
        self._lock = threading.Lock()

    @property
    def engine(self) -> str | None:
        return engine_name()

    def _load(self, name: str) -> Any:
        with self._lock:
            if name in self._voices:
                return self._voices[name]
            model = ensure_voice(name, self.directory, download=self.download)
            from piper import PiperVoice  # noqa: PLC0415

            espeak = _espeak_data_dir()
            voice = (
                PiperVoice.load(model, espeak_data_dir=espeak)
                if espeak is not None
                else PiperVoice.load(model)
            )
            self._voices[name] = voice
            return voice

    def synthesize_sync(self, text: str, voice: str | None = None) -> bytes:
        text = text.strip()[:MAX_TTS_CHARS]
        if not text:
            raise ValueError("nothing to say")
        name = voice or self.default_voice
        engine = self.engine
        if engine == "python":
            piper_voice = self._load(name)
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                piper_voice.synthesize_wav(text, wav)
            return buffer.getvalue()
        if engine == "executable":
            return self._synthesize_with_executable(text, name)
        raise VoiceUnavailable(f"Piper is not installed. {INSTALL_HINT}")

    def _synthesize_with_executable(self, text: str, name: str) -> bytes:
        model = ensure_voice(name, self.directory, download=False)
        executable = _piper_executable()
        assert executable is not None
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "speech.wav"
            completed = subprocess.run(  # noqa: S603 — fixed argv, text on stdin
                [executable, "--model", str(model), "--output_file", str(out)],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=120,
                check=False,
            )
            if completed.returncode != 0 or not out.is_file():
                detail = completed.stderr.decode("utf-8", "replace").strip()[-300:]
                raise VoiceUnavailable(f"piper failed: {detail or completed.returncode}")
            return out.read_bytes()

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        return await asyncio.to_thread(self.synthesize_sync, text, voice)


_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def speakable(text: str, limit: int = 800) -> str:
    """The part of a reply worth reading aloud: code blocks and markdown dropped."""
    text = _FENCE_RE.sub(" (code omitted) ", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"[*_#>]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = " ".join(text.split())
    if len(text) > limit:
        cut = text.rfind(". ", 0, limit)
        text = text[: cut + 1 if cut > limit // 2 else limit]
    return text
