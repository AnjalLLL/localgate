# localgate

> A local-first API gateway for open-source LLMs — authentication, RAG-powered memory extension, database connectors, and token accounting, on top of Ollama, llama.cpp, or vLLM.

## Why

Ollama, LM Studio, and LocalAI solve model *serving*. Nobody has built the management layer on top. localgate adds:

- **Real API key management** — issue, revoke, rate-limit, and track per-key usage
- **Context extension via RAG** — make a small local model "remember" far more than its native context window by chunking, embedding, and retrieving conversation history
- **Database connector abstraction** — SQLite by default, or plug in Postgres/Neon with one env var
- **Token accounting** — per-key prompt/completion token stats out of the box

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (a fast Python package/project manager).

```bash
uv sync --all-extras
cp .env.example .env
uv run localgate serve --reload
```

Point any OpenAI-compatible client at `http://localhost:8000/v1`.

## Status

Early scaffolding — see [ROADMAP.md](./ROADMAP.md) for the full plan.

## License

MIT
