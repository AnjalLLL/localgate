"""ASGI middleware.

Only concerns that genuinely wrap *every* request live here. Auth, rate limiting
and token accounting are dependencies in ``api/deps.py`` instead — see
docs/decisions/0001-dependencies-over-middleware.md.
"""

from localgate.middleware.logging_middleware import REQUEST_ID_HEADER, RequestContextMiddleware

__all__ = ["REQUEST_ID_HEADER", "RequestContextMiddleware"]
