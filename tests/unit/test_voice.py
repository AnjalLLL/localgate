"""Voice helpers: text prepared for speech, voice-name validation, audio decoding,
the REPL commands, and — when a Piper voice is installed — a real round trip."""

import io
import math
import struct
import wave

import pytest

from localgate.voice import VoiceUnavailable, stt
from localgate.voice.tts import PiperTTS, ensure_voice, installed_voices, speakable


def tone_wav(seconds=0.3, rate=44100) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        frames = b"".join(
            struct.pack("<hh", v, v)
            for v in (
                int(8000 * math.sin(2 * math.pi * 440 * i / rate))
                for i in range(int(seconds * rate))
            )
        )
        wav.writeframes(frames)
    return buffer.getvalue()


def test_speakable_drops_code_and_markdown():
    text = "**Done.** Changed `calc.py`:\n```python\nprint(1)\n```\nSee [docs](https://x.y)."
    spoken = speakable(text)
    assert "print(1)" not in spoken and "(code omitted)" in spoken
    assert "**" not in spoken and "`" not in spoken and "docs" in spoken and "https" not in spoken


def test_speakable_cuts_long_text_at_a_sentence():
    text = "This is a sentence. " * 100
    spoken = speakable(text, limit=200)
    assert len(spoken) <= 200 and spoken.endswith(".")


def test_voice_names_are_validated_and_missing_voices_explained(tmp_path):
    with pytest.raises(VoiceUnavailable, match="not a Piper voice"):
        ensure_voice("../../etc/passwd", tmp_path)
    with pytest.raises(VoiceUnavailable, match="localgate voice download"):
        ensure_voice("en_US-lessac-medium", tmp_path, download=False)


def test_installed_voices_need_model_and_config(tmp_path):
    (tmp_path / "en_US-a-low.onnx").write_bytes(b"x")
    (tmp_path / "en_US-a-low.onnx.json").write_text("{}")
    (tmp_path / "en_US-b-low.onnx").write_bytes(b"x")  # config missing
    assert installed_voices(tmp_path) == ["en_US-a-low"]


@pytest.mark.skipif(not stt.available(), reason="voice extra not installed")
def test_decode_audio_resamples_any_wav_to_16k_mono():
    samples = stt.decode_audio(tone_wav(seconds=0.5, rate=44100))
    assert samples.dtype.name == "float32"
    assert abs(len(samples) - 8000) < 400  # 0.5 s at 16 kHz
    assert 0.1 < float(abs(samples).max()) <= 1.0


@pytest.mark.skipif(not stt.available(), reason="voice extra not installed")
def test_decode_audio_rejects_garbage():
    with pytest.raises(ValueError, match="couldn't decode"):
        stt.decode_audio(b"definitely not audio")


def test_tts_without_any_piper_says_how_to_install(tmp_path, monkeypatch):
    from localgate.voice import tts  # noqa: PLC0415

    monkeypatch.setattr(tts, "engine_name", lambda: None)
    with pytest.raises(VoiceUnavailable, match="localgate\\[voice\\]"):
        PiperTTS(tmp_path).synthesize_sync("hello")
