FROM python:3.12-slim

# Install uv (single static binary, no pip bootstrap needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project --no-dev

COPY src/ src/
RUN uv sync --no-dev

EXPOSE 8000
CMD ["uv", "run", "localgate", "serve", "--host", "0.0.0.0", "--port", "8000"]
