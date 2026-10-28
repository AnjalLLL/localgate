"""/v1/audio/*: Piper speech and Whisper transcription, with stand-in engines so
the suite never downloads a model. A real round trip is covered separately when a
voice is installed (see tests/unit/test_voice.py)."""

import io
import wave

from localgate.voice.stt import Transcript


def wav_bytes(seconds: float = 0.5, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(seconds * rate))
    return buffer.getvalue()


class FakeTTS:
    def __init__(self):
        self.calls = []

    async def synthesize(self, text, voice=None):
        self.calls.append((text, voice))
        return wav_bytes()


class FakeSTT:
    def __init__(self):
        self.received = []

    async def transcribe(self, data, language=None):
        self.received.append((data, language))
        return Transcript(text="hello there", language=language or "en", duration=0.5)


async def test_speech_returns_wav_and_maps_openai_voice_names(app, client, auth_headers):
    app.state.tts = FakeTTS()
    resp = await client.post(
        "/v1/audio/speech", headers=auth_headers, json={"input": "hi", "voice": "alloy"}
    )
    assert resp.status_code == 200 and resp.headers["content-type"] == "audio/wav"
    assert resp.content.startswith(b"RIFF")
    # "alloy" is an OpenAI voice name; Piper falls back to its default voice.
    assert app.state.tts.calls == [("hi", None)]
    await client.post(
        "/v1/audio/speech", headers=auth_headers, json={"input": "hi", "voice": "en_GB-alan-low"}
    )
    assert app.state.tts.calls[-1] == ("hi", "en_GB-alan-low")


async def test_speech_rejects_other_formats_and_requires_a_key(app, client, auth_headers):
    app.state.tts = FakeTTS()
    bad = await client.post(
        "/v1/audio/speech", headers=auth_headers, json={"input": "hi", "response_format": "mp3"}
    )
    assert bad.status_code == 400
    assert (await client.post("/v1/audio/speech", json={"input": "hi"})).status_code == 401


async def test_transcription_accepts_multipart_and_raw_bodies(app, client, auth_headers):
    app.state.stt = FakeSTT()
    audio = wav_bytes()
    multipart = await client.post(
        "/v1/audio/transcriptions",
        headers=auth_headers,
        files={"file": ("a.wav", audio, "audio/wav")},
        data={"language": "ne"},
    )
    assert multipart.status_code == 200, multipart.text
    assert multipart.json()["text"] == "hello there"
    raw = await client.post(
        "/v1/audio/transcriptions?language=en",
        headers={**auth_headers, "content-type": "audio/webm"},
        content=audio,
    )
    assert raw.status_code == 200
    assert app.state.stt.received == [(audio, "ne"), (audio, "en")]


async def test_empty_upload_is_rejected(app, client, auth_headers):
    app.state.stt = FakeSTT()
    resp = await client.post(
        "/v1/audio/transcriptions",
        headers={**auth_headers, "content-type": "audio/wav"},
        content=b"",
    )
    assert resp.status_code == 400


async def test_voice_can_be_disabled(app, client, auth_headers):
    app.state.settings.voice_enabled = False
    resp = await client.post("/v1/audio/speech", headers=auth_headers, json={"input": "hi"})
    assert resp.status_code == 501 and "disabled" in resp.text


async def test_voices_endpoint_reports_what_is_available(client, auth_headers):
    body = (await client.get("/v1/audio/voices", headers=auth_headers)).json()
    assert body["default_voice"] == "en_US-lessac-medium"
    assert set(body) >= {"tts_engine", "stt_available", "installed", "stt_model"}


async def test_dashboard_csp_allows_playing_generated_audio(client):
    resp = await client.get("/dashboard/")
    assert "media-src 'self' blob:" in resp.headers["content-security-policy"]
