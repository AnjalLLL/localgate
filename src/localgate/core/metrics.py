"""Prometheus metrics.

Collectors are defined against a private registry rather than the global default
one. The default registry is process-global state: creating two apps in one
process (which every test that builds an app does) would raise "Duplicated
timeseries in CollectorRegistry" on the second. A registry owned by this module,
cleared per app, keeps the metrics honest and the tests independent.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

REGISTRY = CollectorRegistry()

requests_total = Counter(
    "localgate_requests_total",
    "HTTP requests handled by the gateway.",
    ["method", "path", "status"],
    registry=REGISTRY,
)

request_duration_seconds = Histogram(
    "localgate_request_duration_seconds",
    "End-to-end latency of gateway requests.",
    ["method", "path"],
    registry=REGISTRY,
)

tokens_total = Counter(
    "localgate_tokens_total",
    "Tokens accounted by the gateway, by direction.",
    ["model", "direction"],  # direction: prompt | completion
    registry=REGISTRY,
)

backend_errors_total = Counter(
    "localgate_backend_errors_total",
    "Failed calls to the inference backend.",
    ["backend"],
    registry=REGISTRY,
)

cache_events_total = Counter(
    "localgate_cache_events_total",
    "Prompt cache lookups, by outcome.",
    ["outcome"],  # hit | miss
    registry=REGISTRY,
)

rate_limited_total = Counter(
    "localgate_rate_limited_total",
    "Requests rejected because the calling key exceeded its rate limit.",
    registry=REGISTRY,
)

backend_up = Gauge(
    "localgate_backend_up",
    "1 if the inference backend answered its last health check, else 0.",
    registry=REGISTRY,
)


def render() -> bytes:
    """Serialize the current metric values in the Prometheus text format."""
    return generate_latest(REGISTRY)
