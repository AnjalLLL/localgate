"""Playing and recording audio in the terminal, on every OS.

Playback, first that works:

* Windows: ``winsound`` (standard library) plays WAV bytes directly;
* ``sounddevice`` (``localgate[voice]``; PortAudio is bundled on macOS/Windows,
  ``libportaudio2`` on Linux);
* a system player: ``afplay`` (macOS), ``paplay``/``aplay``/``pw-play`` (Linux), or
  ``ffplay`` anywhere.

Recording needs ``sounddevice``: the microphone is read until the user presses
Enter (or a time limit), at 16 kHz mono — exactly what Whisper wants.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

from localgate.voice import INSTALL_HINT, VoiceUnavailable
from localgate.voice.stt import SAMPLE_RATE

_PLAYERS: tuple[tuple[str, ...], ...] = (
    ("afplay",),
    ("pw-play",),
    ("paplay",),
    ("aplay", "-q"),
    ("ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"),
)


def play_wav(data: bytes) -> None:
    """Play WAV bytes and return when playback finishes."""
    if sys.platform == "win32":
        import winsound  # noqa: PLC0415

        winsound.PlaySound(data, winsound.SND_MEMORY)
        return
    if _play_with_sounddevice(data):
        return
    for player in _PLAYERS:
        executable = shutil.which(player[0])
        if executable is None:
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(data)
            path = Path(handle.name)
        try:
            subprocess.run([executable, *player[1:], str(path)], check=False, timeout=600)  # noqa: S603
        finally:
            with contextlib.suppress(OSError):
                path.unlink()
        return
    raise VoiceUnavailable(f"No audio player found. {INSTALL_HINT}")


def _play_with_sounddevice(data: bytes) -> bool:
    try:
        np = importlib.import_module("numpy")
        import sounddevice as sd  # noqa: PLC0415
    except (ImportError, OSError):
        return False
    with wave.open(io.BytesIO(data)) as wav:
        frames = wav.readframes(wav.getnframes())
        rate = wav.getframerate()
        channels = wav.getnchannels()
    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    try:
        sd.play(samples, rate)
        sd.wait()
    except Exception:  # noqa: BLE001 — no output device etc.; fall back to a player
        return False
    return True


def record_until_enter(max_seconds: float = 120.0, prompt: Any = input) -> Any:
    """Record from the default microphone until Enter is pressed. Returns 16 kHz
    mono float32 samples, at most ``max_seconds`` of them."""
    try:
        np = importlib.import_module("numpy")
        import sounddevice as sd  # noqa: PLC0415
    except (ImportError, OSError) as exc:
        raise VoiceUnavailable(f"Recording needs sounddevice. {INSTALL_HINT}") from exc

    chunks: list[Any] = []
    limit = int(max_seconds * SAMPLE_RATE)
    recorded = 0

    def callback(indata: Any, _frames: int, _time: Any, _status: Any) -> None:
        nonlocal recorded
        if recorded < limit:
            chunk = indata.copy().reshape(-1)
            chunks.append(chunk)
            recorded += len(chunk)

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
        )
    except Exception as exc:  # noqa: BLE001 — no device, permission denied, ...
        raise VoiceUnavailable(f"Couldn't open the microphone: {exc}") from exc
    with stream:
        prompt()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)[:limit]
