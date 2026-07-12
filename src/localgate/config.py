"""Pydantic Settings — every value overridable via env var (LOCALGATE_*) or .env file."""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Inference backend
    backend_type: str = "ollama"  # ollama | llamacpp | vllm | openai_compat
    backend_url: str = "http://localhost:11434"
    default_model: str = "llama3"

    # Database (SQLite by default; swap for Postgres/Neon connection string)
    database_url: str = "sqlite+aiosqlite:///./localgate.db"

    # Memory / RAG (context extension)
    memory_enabled: bool = True
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_retrieved_chunks: int = 5

    # Auth
    admin_key: str = Field(default="change-me-in-production")

    model_config = SettingsConfigDict(env_prefix="LOCALGATE_", env_file=".env")
