"""POST /v1/chat/completions — the main OpenAI-compatible endpoint."""
from fastapi import APIRouter

router = APIRouter()

# TODO: implement request validation, auth dependency, call into core.streaming,
# forward to backend adapter, return OpenAI-shaped response.
