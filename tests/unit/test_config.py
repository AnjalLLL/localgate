"""Unit tests for config.Settings — defaults and env var overrides."""
import os

from localgate.config import Settings


def test_defaults_are_sensible():
    settings = Settings(_env_file=None)
    assert settings.backend_type == "ollama"
    assert settings.memory_enabled is True
    assert settings.database_url.startswith("sqlite")


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("LOCALGATE_BACKEND_TYPE", "vllm")
    monkeypatch.setenv("LOCALGATE_PORT", "9999")
    settings = Settings(_env_file=None)
    assert settings.backend_type == "vllm"
    assert settings.port == 9999


def test_constructor_kwargs_override_defaults():
    settings = Settings(_env_file=None, default_model="custom-model")
    assert settings.default_model == "custom-model"
