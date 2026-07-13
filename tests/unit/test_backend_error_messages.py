"""Backend failures must produce a message the operator can act on.

A raw "Connection refused" traceback tells a user nothing about what to do next.
Each of the common failures has to name its own remedy.
"""

import httpx

from localgate.core.errors import describe_backend_failure


def _status_error(status: int, body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
    response = httpx.Response(status, request=request, text=body)
    return httpx.HTTPStatusError("error", request=request, response=response)


def test_connect_error_names_the_url_and_how_to_start_ollama():
    detail = describe_backend_failure(
        httpx.ConnectError("Connection refused"), "http://localhost:11434", "ollama"
    )
    assert "http://localhost:11434" in detail
    assert "ollama serve" in detail


def test_connect_error_for_another_backend_does_not_talk_about_ollama():
    detail = describe_backend_failure(
        httpx.ConnectError("Connection refused"), "http://localhost:8000", "vllm"
    )
    assert "ollama" not in detail.lower()
    assert "vllm" in detail


def test_404_suggests_pulling_the_model():
    detail = describe_backend_failure(
        _status_error(404, "model not found"), "http://localhost:11434", "ollama"
    )
    assert "ollama pull" in detail


def test_other_status_errors_carry_the_code_and_body():
    detail = describe_backend_failure(
        _status_error(500, "internal backend error"), "http://localhost:11434", "ollama"
    )
    assert "500" in detail
    assert "internal backend error" in detail


def test_timeout_points_at_the_timeout_setting():
    detail = describe_backend_failure(
        httpx.ReadTimeout("timed out"), "http://localhost:11434", "ollama"
    )
    assert "LOCALGATE_BACKEND_TIMEOUT" in detail
