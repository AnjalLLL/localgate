# Deployment

## Before you expose it to anything

```bash
LOCALGATE_ENVIRONMENT=production
LOCALGATE_ADMIN_KEY=$(openssl rand -hex 32)
```

Setting `ENVIRONMENT=production` makes the gateway **refuse to start** with the placeholder
admin key. That is the point: the worst failure mode is the quiet one, where a gateway is
reachable on a network using the key that's printed in the documentation.

Also worth setting:

```bash
LOCALGATE_LOG_FORMAT=json          # structured logs for ingestion
LOCALGATE_DATABASE_URL=postgresql+asyncpg://...   # SQLite serializes writers
```

## Docker

```bash
docker compose up
```

The bundled `docker-compose.yml` runs localgate against an Ollama on the host.
`examples/docker-compose.postgres.yml` brings up Postgres alongside it.

Images are published to GHCR on tag:

```bash
docker run -p 8000:8000 --env-file .env ghcr.io/anjalll/localgate:latest
```

## Workers, and what doesn't scale with them

```bash
localgate serve --workers 4
```

Two things live in process memory and do **not** coordinate across workers:

- **Rate limits.** Each worker keeps its own counters, so a key limited to 60/min can
  actually make 60 × N requests per minute across N workers.
- **The prompt cache.** N workers means N independent caches. Correct, just less effective —
  a hit in one worker is a miss in the next.

Neither is a bug; both are the honest tradeoff for a tool whose primary deployment is a
single process on a single machine. **If precise per-key limits matter to you, run one
worker** (a gateway is I/O-bound on the inference backend anyway, so this costs less than
you'd think), or put the limiting in front of localgate at your reverse proxy.

Both sit behind small interfaces — `allow(key_id, limit) -> bool` and the cache's
`get`/`set` — that a Redis-backed implementation would satisfy without touching any code
above them. That is the shape of the fix if you need it.

## Reverse proxy

Nginx buffers proxied responses by default, which would hold an entire SSE stream until it
completed and defeat the point of streaming. localgate sends `X-Accel-Buffering: no` on
streaming responses, which Nginx honours — but set the timeouts too, because local models
on CPU are slow:

```nginx
location / {
    proxy_pass http://localhost:8000;
    proxy_buffering off;
    proxy_read_timeout 300s;
    proxy_set_header X-Request-ID $request_id;   # localgate adopts this, tying the logs together
}
```

Keep `/metrics` off the public listener. It exposes no prompts or key material, but it is
unauthenticated and there's no reason for the internet to have it.

## Health checks

Two endpoints, because orchestrators ask two different questions, and conflating them causes
outages.

| Probe | Endpoint | Checks |
|---|---|---|
| Liveness | `/health/live` | Nothing external. Is this process wedged? |
| Readiness | `/health` | Backend reachable, database connected. Can it serve? |

**Do not point a liveness probe at `/health`.** It would fail when *Ollama* is down, and
Kubernetes would respond by restarting a perfectly healthy gateway, over and over, fixing
nothing. `/health` returns 503 when a hard dependency is down, so a load balancer can act on
the status code without parsing the body.

```yaml
livenessProbe:
  httpGet: { path: /health/live, port: 8000 }
readinessProbe:
  httpGet: { path: /health, port: 8000 }
```

## Graceful shutdown

Uvicorn stops accepting connections on SIGTERM, drains in-flight requests, and only then
runs localgate's shutdown — which closes the backend's HTTP client and disposes the database
engine. Nothing is still using them by the time they close.

Give the orchestrator enough grace period to cover a slow inference call, or you'll kill
requests mid-generation:

```yaml
terminationGracePeriodSeconds: 130   # > LOCALGATE_BACKEND_TIMEOUT
```

## Migrations

The server applies pending migrations at startup, so a normal deploy needs no extra step.
If you'd rather gate deploys on an explicit migration:

```bash
localgate db upgrade    # safe to re-run; no-ops when already current
localgate db current    # prints the current revision
```

Upgrading from localgate 0.1–0.2 is handled automatically — see
[Database Setup](database-setup.md#upgrading-from-localgate-01-02).

## Monitoring

`/metrics` serves Prometheus:

| Metric | Use |
|---|---|
| `localgate_requests_total{method,path,status}` | Traffic and error rate |
| `localgate_request_duration_seconds{method,path}` | Latency distribution |
| `localgate_tokens_total{model,direction}` | Token spend per model |
| `localgate_backend_errors_total{backend}` | Backend health, as your users experience it |
| `localgate_rate_limited_total` | Keys hitting their ceiling |
| `localgate_cache_events_total{outcome}` | Cache hit rate |
| `localgate_backend_up` | 1/0 from the last health check |

Alerts worth having: `localgate_backend_up == 0`, a rising
`rate(localgate_backend_errors_total[5m])`, and p99 of `request_duration_seconds` against
whatever your users will tolerate.

## Logging

`LOCALGATE_LOG_FORMAT=json` emits one JSON object per line, each carrying a `request_id`.
Set `X-Request-ID` at your proxy and localgate adopts it, so a request keeps a single
identity across every service that touched it.

```json
{"event": "request_completed", "request_id": "a3f9...", "method": "POST",
 "path": "/v1/chat/completions", "status": 200, "duration_ms": 1843.2, "level": "info"}
```

Prompts and completions are **not** logged. Sessions and keys are referenced by id.

## Backups

The database holds everything: keys, usage, conversations, memory. Back it up the way you'd
back up any Postgres database.

For a portable snapshot, `GET /admin/export` returns every row as JSON. Nobody should feel
locked into a self-hosted tool.

## A production checklist

- [ ] `LOCALGATE_ENVIRONMENT=production` and a real `LOCALGATE_ADMIN_KEY`
- [ ] Postgres, not SQLite
- [ ] `LOCALGATE_LOG_FORMAT=json`
- [ ] `/admin` and `/metrics` not exposed publicly
- [ ] Liveness → `/health/live`, readiness → `/health`
- [ ] `proxy_buffering off` and a generous `proxy_read_timeout`
- [ ] Grace period longer than `LOCALGATE_BACKEND_TIMEOUT`
- [ ] One worker, or rate limiting at the proxy — see above
- [ ] Database backups
