import pytest
from pydantic import ValidationError

from medbridge.config import Settings, get_settings

_ISOLATED_VARS = [
    "LLM_PROVIDER", "LLM_MODEL", "LLM_MODEL_GATES", "LLM_MODEL_GENERATOR", "LOCAL_ONLY_MODE",
    "OLLAMA_BASE_URL", "DATABASE_URL", "CORS_ORIGINS", "HOST", "GROQ_API_KEY",
    "POSTGRES_PASSWORD", "LLM_TEMPERATURE_GATES", "LLM_MAX_TOKENS_GATES",
]


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Ignore the developer .env and restore the get_settings cache afterwards."""
    orig_env_file = Settings.model_config.get("env_file", ".env")
    Settings.model_config["env_file"] = None
    for var in _ISOLATED_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
    Settings.model_config["env_file"] = orig_env_file
    get_settings.cache_clear()


def test_defaults_are_local_first():
    settings = Settings()

    assert settings.LLM_PROVIDER == "ollama"
    assert settings.LOCAL_ONLY_MODE is True
    assert settings.OLLAMA_BASE_URL == "http://127.0.0.1:11434"
    assert settings.LLM_TEMPERATURE_GATES == 0.0
    assert settings.LLM_SEED == 42
    assert settings.SOFT_ASK_MAX_COUNT == 2
    assert settings.SPECULATIVE_RETRIEVAL is True
    assert settings.HOST == "127.0.0.1"
    assert settings.QDRANT_URL == "http://127.0.0.1:6333"
    assert settings.GROQ_API_KEY == ""


def test_no_remote_api_key_required():
    # A fresh checkout must start without any paid-API credentials.
    settings = Settings()
    assert settings.OPENAI_API_KEY == ""


def test_role_models_fall_back_to_llm_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "qwen3:8b")
    settings = Settings()
    assert settings.gates_model == "qwen3:8b"
    assert settings.generator_model == "qwen3:8b"

    monkeypatch.setenv("LLM_MODEL_GATES", "qwen3.5:4b")
    settings = Settings()
    assert settings.gates_model == "qwen3.5:4b"
    assert settings.generator_model == "qwen3:8b"


def test_invalid_provider_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "someone-elses-cloud")
    with pytest.raises(ValidationError):
        Settings()


def test_type_coercion(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE_GATES", "0.5")
    monkeypatch.setenv("LLM_MAX_TOKENS_GATES", "1024")
    monkeypatch.setenv("CORS_ORIGINS", '["http://test.com"]')

    settings = Settings()

    assert settings.LLM_TEMPERATURE_GATES == 0.5
    assert settings.LLM_MAX_TOKENS_GATES == 1024
    assert settings.CORS_ORIGINS == ["http://test.com"]


def test_singleton_behavior():
    get_settings.cache_clear()
    settings_1 = get_settings()
    settings_2 = get_settings()

    assert settings_1 is settings_2
