# 1. Auth, rate limiting and token accounting are dependencies, not middleware

- **Status:** Accepted
- **Date:** 2026-07-13

## Context

The original folder layout reserved a file per cross-cutting concern under
`middleware/`: `auth_middleware.py`, `rate_limit_middleware.py`,
`token_counting_middleware.py`, `logging_middleware.py`. Three of the four were
never written, and it is worth recording why they should not be.

## Decision

Only request-context logging is middleware. Authentication, rate limiting and
token accounting are FastAPI dependencies in `api/deps.py`, applied per route.

## Rationale

ASGI middleware runs before routing, which means it sees a raw `Request` and
nothing else. Each of the three concerns needs more than that:

- **Auth** must resolve the bearer token to an `APIKey` row and hand that row to
  the handler. A middleware can stash it on `request.state`, but then the
  handler's signature no longer declares what it depends on, and it cannot be
  tested without constructing an HTTP request. `api_key: APIKey = Depends(require_api_key)`
  says exactly what the route needs and gives OpenAPI the security scheme for free.

- **Rate limiting** is *per key*, so it can only run after auth has resolved the
  key. Expressing that ordering as middleware means one middleware reaching into
  another's `request.state` — an implicit dependency the type system can't see.

- **Token accounting** needs the parsed request body *and* the backend's response
  usage block. Middleware would have to re-read and re-parse the body (Starlette
  streams it, so this needs buffering) and re-parse the response — including
  reassembling a streamed one. The handler already holds both.

Logging is different: it needs nothing but the method, path, status and duration,
and it must wrap requests that never reach a route at all (404s, malformed
bodies). That is exactly what middleware is for.

## Consequences

- Route handlers declare their own security and limits, and read as such.
- A route that forgets `Depends(require_api_key)` is unauthenticated. This is
  guarded in two ways: admin routers apply the dependency at the router level
  (`APIRouter(dependencies=[Depends(require_admin)])`), and
  `tests/integration/test_key_management.py::test_every_route_is_guarded` asserts
  that no route escapes both auth dependencies.
- `middleware/` holds one file. That is the honest size of the problem.
