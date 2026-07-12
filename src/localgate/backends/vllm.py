"""vLLM adapter — vLLM already speaks OpenAI's API, so this is a thin pass-through."""
from localgate.backends.base import InferenceBackend


class VLLMBackend(InferenceBackend):
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
