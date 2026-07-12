"""Ollama HTTP adapter — talks to the native /api/chat endpoint."""
from localgate.backends.base import InferenceBackend


class OllamaBackend(InferenceBackend):
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    async def chat(self, request):
        raise NotImplementedError

    async def chat_stream(self, request):
        raise NotImplementedError

    async def list_models(self):
        raise NotImplementedError

    async def health(self):
        raise NotImplementedError
