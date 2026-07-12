"""Generic adapter for any OpenAI-compatible backend not covered above."""
from localgate.backends.base import InferenceBackend


class OpenAICompatBackend(InferenceBackend):
    def __init__(self, base_url: str):
        self.base_url = base_url
