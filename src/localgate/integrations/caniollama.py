"""Read-only client for the caniollama compatibility registry.

This module provides best-effort lookups of model compatibility data from the
public caniollama registry. All failures (timeout, network error, 404) degrade
gracefully to None rather than raising, since the registry is an enhancement to
the model picker, not a dependency — localgate must work when the registry is
unreachable or has no data for a given model.

IMPORTANT: This integration only performs GET requests to read public data.
localgate never submits user data or inference runs to the registry on the
user's behalf — that path requires explicit consent in the caniollama CLI tool.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ModelCompatSummary:
    """Registry's view of a model's tool-calling reliability."""

    model_name: str
    total_reports: int
    pass_rate_by_case: dict[str, float]  # case_id -> pass_rate (0.0–1.0)
    last_updated: str


class CaniollamaClient:
    """Thin async client for GET /models/{name} with caching and graceful degradation."""

    def __init__(
        self,
        registry_url: str,
        enabled: bool = True,
        timeout: float = 1.5,
        cache_ttl: float = 300.0,  # 5 minutes
    ):
        self.registry_url = registry_url.rstrip("/")
        self.enabled = enabled
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[ModelCompatSummary | None, datetime]] = {}
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Lazy initialization of the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def get_compat(self, model: str) -> ModelCompatSummary | None:
        """Fetch compatibility data for ``model``, returning None on any failure.

        A 404 means the registry has no data yet for this model (not an error).
        Any other failure (timeout, network error, non-2xx) also returns None —
        this sits in the model picker's render path and must never block or raise.
        """
        if not self.enabled:
            return None

        # Check cache first
        now = datetime.now()
        if model in self._cache:
            cached_result, cached_at = self._cache[model]
            if now - cached_at < timedelta(seconds=self.cache_ttl):
                return cached_result

        # Fetch from registry
        try:
            client = await self._ensure_client()
            url = f"{self.registry_url}/models/{model}"
            resp = await client.get(url)

            if resp.status_code == 404:
                # Registry has no data for this model yet — cache the absence
                self._cache[model] = (None, now)
                return None

            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

            # Parse the registry's response into our summary type
            summary = ModelCompatSummary(
                model_name=data["model_name"],
                total_reports=data["total_reports"],
                pass_rate_by_case=data.get("pass_rate_by_case", {}),
                last_updated=data.get("last_updated", "unknown"),
            )
            self._cache[model] = (summary, now)
            return summary

        except httpx.TimeoutException:
            logger.debug("caniollama registry timeout", model=model)
            return None
        except httpx.HTTPError as exc:
            logger.debug("caniollama registry error", model=model, error=str(exc))
            return None
        except Exception as exc:
            # Catch-all for parse errors, etc. — never let this raise into the picker
            logger.warning(
                "unexpected error fetching caniollama data", model=model, error=str(exc)
            )
            return None

    async def aclose(self) -> None:
        """Release HTTP client resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def overall_pass_rate(summary: ModelCompatSummary) -> float:
    """Compute a single pass rate from the per-case rates.

    The registry's pass_rate_by_case maps case IDs to their individual pass
    rates. We weight these by case importance: structured_pass (the core
    tool-calling signal) counts for 60%, and the remaining cases split the
    other 40% evenly.
    """
    if not summary.pass_rate_by_case:
        return 0.0

    # The registry's core signal is whether the model produces structured tool_calls
    structured = summary.pass_rate_by_case.get("structured_pass", 0.0)

    # Other cases (correctness of the call itself, follow-through, etc.) matter
    # less than whether it even emits tool_calls, but they're still signal
    other_cases = [
        rate for case_id, rate in summary.pass_rate_by_case.items() if case_id != "structured_pass"
    ]
    other_avg = sum(other_cases) / len(other_cases) if other_cases else 0.0

    # Weight: 60% structured, 40% other
    return 0.6 * structured + 0.4 * other_avg


async def get_compat_for_models(
    client: CaniollamaClient, models: list[str]
) -> dict[str, ModelCompatSummary | None]:
    """Fan out compatibility lookups for multiple models concurrently.

    Returns a dict mapping model name -> summary (or None if no data / error).
    This ensures a single slow/unreachable registry call doesn't multiply into
    N × timeout when rendering a picker with N models.
    """
    tasks = [client.get_compat(model) for model in models]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return dict(zip(models, results))
