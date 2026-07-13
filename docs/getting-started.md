# Getting Started

From nothing to a working, authenticated, memory-backed gateway in about five minutes.
This assumes [Ollama](https://ollama.com) as the inference backend;
[Configuration](configuration.md) covers the others.

## 1. Install

localgate needs Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/AnjalLLL/localgate.git
cd localgate
uv sync --all-extras
```

## 2. Start the inference backend

localgate does not serve models. It manages access to something that does.

```bash
ollama serve                    # in another terminal
ollama pull llama3              # the chat model
ollama pull nomic-embed-text    # the embedding model, for RAG memory
```

The embedding model is what makes memory work. Without it chat still works and memory
quietly does not — `/health` will say so rather than leaving you guessing.

## 3. Configure

```bash
cp .env.example .env
```

The defaults already point at Ollama on its usual port and use SQLite, so for a local
trial the only line worth changing is the admin key:

```bash
LOCALGATE_ADMIN_KEY=$(openssl rand -hex 32)
```

## 4. Create the database and your first key

```bash
uv run localgate db upgrade
uv run localgate keys create --name my-app
```

That prints the key **once**:

```
  lg_9f3a2b1c8d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a

  id           04e42b62-fe40-4c2d-b6cc-72362e9bf1bc
  name         my-app
  rate limit   60/min

  Store it now — only its hash is kept, so it cannot be shown again.
```

Only a SHA-256 hash is stored, so there is genuinely no way to recover it later.
Losing it means revoking that key and creating another.

## 5. Start the gateway

```bash
uv run localgate serve
```

Confirm it is all actually wired together:

```bash
uv run localgate health
```

```
✓ backend  ollama at http://localhost:11434 (2 models)
✓ database sqlite+aiosqlite — connected (migration 0002)
```

## 6. Use it

localgate speaks the OpenAI API, so any OpenAI client works unchanged. Point `base_url`
at it and pass the key you just created.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="lg_9f3a...")

response = client.chat.completions.create(
    model="llama3",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

Or with curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer lg_9f3a..." \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## 7. Give the model a memory

Memory is what makes localgate more than a proxy. Send an `X-Session-ID` header and the
gateway stores each turn, embeds it, and retrieves the relevant parts on later turns —
so an 8K model can hold a conversation far longer than 8K tokens.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer lg_9f3a..." \
  -H "X-Session-ID: my-conversation" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3","messages":[{"role":"user","content":"My name is Ana and I prefer Postgres."}]}'
```

Then, in a **separate request carrying no history at all**:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer lg_9f3a..." \
  -H "X-Session-ID: my-conversation" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3","messages":[{"role":"user","content":"What database do I prefer?"}]}'
```

The model answers "Postgres" — not because you sent the history, but because the
gateway retrieved it. With the OpenAI SDK, set the header once:

```python
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="lg_9f3a...",
    default_headers={"X-Session-ID": "my-conversation"},
)
```

[RAG Memory](rag-memory.md) explains how retrieval works and how to tune it.

## 8. Watch it

Open <http://localhost:8000/dashboard/> and paste in your admin key to see keys, token
usage, and stored conversations. Interactive API docs are at <http://localhost:8000/docs>.

## Where to go next

- [Configuration](configuration.md) — every setting and what it does
- [Database Setup](database-setup.md) — moving from SQLite to Postgres or Neon
- [RAG Memory](rag-memory.md) — how memory works, and how to tune retrieval
- [API Reference](api-reference.md) — every endpoint
- [Deployment](deployment.md) — running this somewhere real

## Troubleshooting

**"Could not reach the inference backend"** — Ollama isn't running. `ollama serve`.

**"The backend returned 404 — the model is probably not available"** — the model isn't
pulled. `ollama pull llama3`.

**The model doesn't remember anything** — either the request had no `X-Session-ID`
(every request without one is its own session), or the embedding model isn't pulled.
`localgate health` and `GET /health` each tell you which.

**"This database has no localgate schema yet"** — run `localgate db upgrade`.
