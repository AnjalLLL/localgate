"""config.Settings — defaults, env overrides, and fail-fast validation."""

import pytest
from pydantic import ValidationError

from localgate.config import INSECURE_ADMIN_KEY, Settings


def test_defaults_are_sensible():
    settings = Settings(_env_file=None)
    assert settings.backend_type == "ollama"
    assert settings.memory_enabled is True
    assert settings.database_url.startswith("sqlite")
    assert settings.environment == "development"


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("LOCALGATE_BACKEND_TYPE", "vllm")
    monkeypatch.setenv("LOCALGATE_PORT", "9999")
    settings = Settings(_env_file=None)
    assert settings.backend_type == "vllm"
    assert settings.port == 9999


def test_constructor_kwargs_override_defaults():
    assert Settings(_env_file=None, default_model="custom-model").default_model == "custom-model"


def test_production_refuses_to_start_with_the_placeholder_admin_key():
    """The worst failure mode is the quiet one: a gateway serving production traffic
    with the key that is printed in the docs. It must not be possible."""
    with pytest.raises(ValidationError, match="LOCALGATE_ADMIN_KEY"):
        Settings(_env_file=None, environment="production", admin_key=INSECURE_ADMIN_KEY)


def test_production_starts_with_a_real_admin_key():
    settings = Settings(_env_file=None, environment="production", admin_key="a-real-secret")
    assert settings.uses_insecure_admin_key is False


def test_development_tolerates_the_placeholder_key_but_flags_it():
    """Local development has to stay zero-config; a warning is the right response there."""
    assert Settings(_env_file=None).uses_insecure_admin_key is True


def test_chunk_overlap_at_or_above_chunk_size_is_rejected():
    """An overlap >= chunk_size never advances the window, so chunking would not
    terminate. Refusing the config beats hanging on the first request."""
    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_model_aliases_parse_from_a_json_env_var(monkeypatch):
    monkeypatch.setenv("LOCALGATE_MODEL_ALIASES", '{"fast": "phi4-mini", "smart": "llama3:70b"}')
    assert Settings(_env_file=None).model_aliases == {"fast": "phi4-mini", "smart": "llama3:70b"}


def test_cors_origins_accept_the_comma_separated_form_people_actually_type(monkeypatch):
    monkeypatch.setenv("LOCALGATE_CORS_ORIGINS", "http://a.test, http://b.test")
    assert Settings(_env_file=None).cors_origins == ["http://a.test", "http://b.test"]


def test_cors_origins_also_accept_json(monkeypatch):
    monkeypatch.setenv("LOCALGATE_CORS_ORIGINS", '["http://a.test"]')
    assert Settings(_env_file=None).cors_origins == ["http://a.test"]


def test_resolve_model_maps_an_alias_to_its_target():
    settings = Settings(_env_file=None, model_aliases={"fast": "phi4-mini"})
    assert settings.resolve_model("fast") == "phi4-mini"


def test_resolve_model_passes_through_an_unaliased_name():
    settings = Settings(_env_file=None, model_aliases={"fast": "phi4-mini"})
    assert settings.resolve_model("llama3") == "llama3"


def test_resolve_model_falls_back_to_the_default_when_none_is_requested():
    assert Settings(_env_file=None, default_model="llama3").resolve_model(None) == "llama3"


def test_an_out_of_range_port_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, port=99999)
