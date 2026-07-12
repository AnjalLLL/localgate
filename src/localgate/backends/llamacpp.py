"""llama.cpp server adapter."""
from localgate.backends.base import InferenceBackend


class LlamaCppBackend(InferenceBackend):
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
