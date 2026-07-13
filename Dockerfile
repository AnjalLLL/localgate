# Multi-stage: the build stage carries uv and the lockfile; the runtime stage carries
# neither. What ships is the interpreter, the venv, and the source — nothing else.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first, in their own layer: they change far less often than the source, so
# this layer stays cached across almost every rebuild.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra postgres

# The postgres extra is included deliberately. Without asyncpg the image cannot talk to
# Postgres at all, and Postgres is what the deployment docs tell people to use — an image
# that only works with SQLite would be a trap.

COPY src/ src/
COPY README.md ./
RUN uv sync --frozen --no-dev --extra postgres


FROM python:3.12-slim AS runtime

# Don't run as root. A gateway is network-facing by definition.
RUN useradd --create-home --uid 1000 localgate

# /data is where the SQLite database lives when a volume is mounted there (see
# docker-compose.yml). It must be owned by the runtime user, or the process cannot create
# the database file and the container dies on first boot with a permissions error.
RUN mkdir -p /data && chown localgate:localgate /data
VOLUME ["/data"]

WORKDIR /app

COPY --from=builder --chown=localgate:localgate /app/.venv /app/.venv
COPY --from=builder --chown=localgate:localgate /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    LOCALGATE_HOST=0.0.0.0 \
    LOCALGATE_PORT=8000 \
    LOCALGATE_LOG_FORMAT=json

USER localgate
EXPOSE 8000

# Liveness, not readiness: this answers "is the process wedged", and must not fail merely
# because the inference backend is down. See docs/deployment.md.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"

# `serve` applies pending migrations at startup, so no separate migrate step is needed.
CMD ["localgate", "serve"]
