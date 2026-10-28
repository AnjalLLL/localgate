"""Voice in the terminal: read replies aloud with Piper, and dictate with Whisper.

Shared by `localgate code` (``--voice``, ``/voice``, ``/listen``) and the
`localgate voice` commands. Engines load lazily, on first use, so a session that
never speaks or listens pays nothing for having voice available.
"""

from __future__ import annotations

import asyncio
from typing import Any

from localgate.voice import VoiceUnavailable


class VoiceIO:
    def __init__(self, settings: Any, console: Any, *, speak: bool = False) -> None:
        self.settings = settings
        self.console = console
        self.speak_enabled = speak
        self._tts: Any | None = None
        self._stt: Any | None = None

    def tts(self) -> Any:
        if self._tts is None:
            from localgate.voice.tts import PiperTTS, voices_dir  # noqa: PLC0415

            self._tts = PiperTTS(
                voices_dir(self.settings),
                self.settings.voice_tts_voice,
                download=self.settings.voice_auto_download,
            )
        return self._tts

    def stt(self) -> Any:
        if self._stt is None:
            from localgate.voice.stt import WhisperSTT  # noqa: PLC0415

            self._stt = WhisperSTT(self.settings.voice_stt_model, self.settings.voice_stt_device)
        return self._stt

    async def say(self, text: str) -> None:
        """Speak ``text`` (code blocks and markdown removed). Never raises."""
        from localgate.voice.audio import play_wav  # noqa: PLC0415
        from localgate.voice.tts import speakable  # noqa: PLC0415

        spoken = speakable(text)
        if not spoken:
            return
        try:
            audio = await self.tts().synthesize(spoken)
            await asyncio.to_thread(play_wav, audio)
        except (VoiceUnavailable, ValueError, OSError) as exc:
            self.console.print(f"[yellow]voice: {exc}[/yellow]")

    async def listen(self) -> str | None:
        """Record until Enter, then transcribe. None if nothing usable was heard."""
        from localgate.voice.audio import record_until_enter  # noqa: PLC0415

        self.console.print("[bold cyan]● listening — press Enter to stop[/bold cyan]")
        try:
            samples = await asyncio.to_thread(record_until_enter)
        except VoiceUnavailable as exc:
            self.console.print(f"[yellow]voice: {exc}[/yellow]")
            return None
        if len(samples) < 1600:  # under 0.1 s
            self.console.print("[dim]nothing recorded[/dim]")
            return None
        with self.console.status("transcribing…"):
            try:
                result = await asyncio.to_thread(self.stt().transcribe_samples, samples)
            except VoiceUnavailable as exc:
                self.console.print(f"[yellow]voice: {exc}[/yellow]")
                return None
        text = str(result.text).strip()
        if not text:
            self.console.print("[dim]didn't catch that[/dim]")
            return None
        return text
