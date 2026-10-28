"""Local voice: Piper text-to-speech and faster-whisper speech-to-text.

Both run entirely on this machine. They are an optional extra::

    pip install 'localgate[voice]'

Used by the gateway's ``/v1/audio/*`` endpoints (and so the dashboard's chat), and
by ``localgate code --voice`` / ``localgate voice``. Nothing here is imported unless
voice is actually used, so the base install neither needs nor loads these
dependencies.
"""

from __future__ import annotations


class VoiceUnavailable(RuntimeError):
    """A voice feature was asked for but its dependency or model is missing."""


INSTALL_HINT = "Install voice support with: pip install 'localgate[voice]'"
