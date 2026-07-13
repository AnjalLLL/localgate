# API Reference

Interactive docs are served at `/docs` (Swagger) and `/redoc` while the gateway is
running. This page is the narrative version, including the things a schema can't tell you.

## Authentication

Two credentials, two audiences:

| Surface | Header | Credential |
|---|---|---|
| `/v1/*` — client routes | `Authorization: Bearer <api-key>` | A key from `localgate keys create` |
| `/admin/*` — admin routes | `X-Admin-Key: <admin-key>` | `LOCALGATE_ADMIN_KEY` |

`/health`, `/health/live` and `/metrics` are deliberately open: a readiness probe that
required a key would be useless to a load balancer, and Prometheus has no good way to
hold a bearer token per target. None of them expose prompts or key material.

## Errors

Every error — including validation failures and rate limits — comes back in the OpenAI
envelope, so the OpenAI SDK can parse it:

```json
{
  "error": {
    "message": "Could not reach the inference backend at http://localhost:11434. Is Ollama running? Start it with `ollama serve`.",
    "type": "backend_error",
    "code": null
  }
}
```

| Status | `type` | Means |
|---|---|---|
| 401 | `authentication_error` | Missing, invalid, or revoked key |
| 400 | `invalid_request_error` | A malformed request the gateway could still parse |
| 422 | `invalid_request_error` | The body failed schema validation |
| 429 | `rate_limit_error` | The calling key exceeded its per-minute limit |
| 502 | `backend_error` | The inference backend was unreachable or failed |
| 503 | — | `/health` only: a hard dependency is down |

---

## `POST /v1/chat/completions`

The main endpoint. OpenAI-compatible, streaming and non-streaming.

**Headers**

| Header | Required | Purpose |
|---|---|---|
| `Authorization: Bearer <key>` | yes | |
| `X-Session-ID: <id>` | no | Groups requests into one conversation for memory. **Generated per-request if omitted**, which means memory does nothing — an easy thing to trip over. |

**Body** — the OpenAI schema. `messages` is required and must be non-empty. **Fields
localgate doesn't know about are forwarded to the backend verbatim**, so backend-specific
sampling knobs (`top_k`, `repeat_penalty`, `min_p`, …) keep working: a gateway that
dropped them would quietly change your results.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer lg_..." \
  -H "X-Session-ID: conversation-1" \
  -d '{"model":"llama3","messages":[{"role":"user","content":"Hello"}],"temperature":0.7}'
```

**Streaming** (`"stream": true`) returns Server-Sent Events, terminated by `data: [DONE]`.

A backend failure *mid-stream* cannot become a 502 — the 200 and its headers are already
on the wire. It is reported inside the stream instead, in the same error envelope, and
the stream still terminates with `[DONE]` so the client is never left hanging:

```
data: {"error": {"message": "Could not reach the inference backend...", "type": "backend_error"}}
data: [DONE]
```

**Token accounting.** Non-streamed responses use the backend's own `usage` block, which
was counted with the model's real tokenizer over the full augmented prompt. Streamed
responses carry no usage block, so those counts are tiktoken's `cl100k_base`
approximation — consistent, but not byte-exact for a Llama or Qwen tokenizer.

---

## `POST /v1/completions`

The legacy text-completion endpoint. Most local backends have dropped it, so localgate
implements it by translating the prompt into a single user message, forwarding it to the
chat route, and translating the answer back. That keeps older clients and LangChain's
non-chat `OpenAI` LLM working against any backend.

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer lg_..." \
  -d '{"model": "llama3", "prompt": "Once upon a time"}'
```

---

## `POST /v1/embeddings`

Embeds text with the configured embedding model. Billed against the calling key like any
other request — embeddings consume backend capacity, and leaving them out would let a
client embed without limit while the usage dashboard under-reported the load.

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer lg_..." \
  -d '{"input": ["hello", "world"]}'
```

---

## `GET /v1/models`

Lists the backend's models **plus any configured aliases**, so a caller passing
`model: "fast"` can see that it's a name they're allowed to use.

---

## `GET /v1/conversations`

The calling key's sessions, most recently active first.

## `GET /v1/conversations/{session_id}`

Full history for one session, plus its rolling summary and memory-chunk count.

```json
{
  "session_id": "conversation-1",
  "messages": [{"role": "user", "content": "...", "created_at": "..."}],
  "summary": "The user is Ana, prefers Postgres...",
  "memory_chunks": 14
}
```

A session belonging to a *different* key returns **404, not 403**. 403 would confirm the
session exists, letting a caller enumerate other keys' session ids by probing. Someone
else's session and a session that never existed look identical from the outside.

---

## `POST /admin/keys`

Creates a key. **This is the only response that ever contains the raw key** — only a
hash is stored, so it cannot be shown again.

```bash
curl -X POST http://localhost:8000/admin/keys \
  -H "X-Admin-Key: $ADMIN_KEY" \
  -d '{"name": "my-app", "rate_limit_per_min": 120}'
```

```json
{
  "id": "04e42b62-...",
  "name": "my-app",
  "api_key": "lg_9f3a...",
  "key_prefix": "lg_9f3a2b1",
  "rate_limit_per_min": 120,
  "revoked": false
}
```

`key_prefix` is kept in plaintext so a key can be identified in a listing — without it
there is no way to tell which row corresponds to the key in your hand.

## `GET /admin/keys` · `GET /admin/keys/{id}`

Lists keys. Never includes secret material.

## `PATCH /admin/keys/{id}`

Changes a key's rate limit without reissuing it.

```bash
curl -X PATCH http://localhost:8000/admin/keys/04e42b62-... \
  -H "X-Admin-Key: $ADMIN_KEY" -d '{"rate_limit_per_min": 10}'
```

## `DELETE /admin/keys/{id}`

Revokes a key. This sets a flag — it does not delete the row, because usage records
reference it and deleting it would silently rewrite the history your dashboard reports.

---

## `GET /admin/usage`

Everything the dashboard needs in one round trip: totals, per-key, per-model, and daily
buckets. `?days=N` sets the window for the daily series (default 14).

Per-key breakdown uses an outer join, so **keys with zero requests still appear** — that
is exactly the key you're looking for when auditing.

## `GET /admin/usage/{api_key_id}`

Totals for one key: request count, prompt/completion tokens, average latency.

---

## `GET /admin/config` · `PUT /admin/config/database-url`

Reports the running configuration, with database credentials redacted. The `PUT` proves
a connection works *before* persisting it — see [Database Setup](database-setup.md).
The database is not swapped live; a restart is required, and the response says so.

---

## `GET /admin/export`

Every row localgate holds — keys (metadata only), usage, conversations, summaries — as
one JSON document. Nobody should feel locked into a self-hosted tool.

Key hashes and embedding vectors are excluded: the hashes are secret material with no
value outside this database, and the vectors would multiply the export size while being
reproducible from the text with the same embedding model.

---

## `GET /health`

Full readiness: backend reachability, database connectivity, memory config, cache stats,
and configuration warnings. Returns **503** when a hard dependency is down, so a load
balancer can act on the status code alone.

```json
{
  "status": "ok",
  "version": "0.6.0",
  "backend": {"type": "ollama", "url": "http://localhost:11434", "reachable": true},
  "database": {"dialect": "sqlite+aiosqlite", "connected": true, "error": null},
  "memory": {"enabled": true, "embedding_model": "nomic-embed-text"},
  "warnings": []
}
```

## `GET /health/live`

Liveness. Checks **nothing external**, by design: a liveness probe that failed because
*Ollama* was down would have Kubernetes restart a perfectly healthy gateway, forever,
fixing nothing.

## `GET /metrics`

Prometheus metrics: `localgate_requests_total`, `localgate_request_duration_seconds`,
`localgate_tokens_total`, `localgate_backend_errors_total`, `localgate_cache_events_total`,
`localgate_rate_limited_total`, `localgate_backend_up`.

Paths are labelled with the route *template* (`/v1/conversations/{session_id}`), never the
concrete path — the raw path would mint a new time series per session id and blow up
cardinality.
