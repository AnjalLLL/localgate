"""llama.cpp server adapter.

``llama-server`` exposes an OpenAI-compatible surface on port 8080. It serves a
single model per process and reports it under whatever alias it was launched
with, so the ``model`` field in a request is effectively ignored — which is why
model aliasing (see ``core/model_router.py``) matters most for this backend.
"""

from __future__ import annotations

from localgate.backends.openai_compat import OpenAICompatBackend


class LlamaCppBackend(OpenAICompatBackend):
    name = "llamacpp"
    default_base_url = "http://localhost:8080"
