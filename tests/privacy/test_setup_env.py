"""scripts/setup_env.py — secret generation for a fresh .env."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from urllib.parse import urlsplit

import pytest
from cryptography.fernet import Fernet

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "setup_env.py"
EXAMPLE = """# example
GROQ_API_KEY=
POSTGRES_PASSWORD=
DATABASE_URL=postgresql+asyncpg://medbridge_app:password@127.0.0.1:5433/medbridge
LLM_MODEL=qwen3:8b
"""


@pytest.fixture(scope="module")
def setup_env() -> ModuleType:
    spec = importlib.util.spec_from_file_location("setup_env_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read(path: Path, setup_env: ModuleType) -> dict[str, str]:
    return setup_env.parse_env(path.read_text(encoding="utf-8").splitlines())


def test_creates_env_with_generated_secrets(tmp_path: Path, setup_env: ModuleType) -> None:
    example, env = tmp_path / ".env.example", tmp_path / ".env"
    example.write_text(EXAMPLE, encoding="utf-8")
    setup_env.setup_env(env, example)
    values = _read(env, setup_env)
    Fernet(values["ENCRYPTION_MASTER_KEY"].encode())  # valid Fernet key
    assert len(values["SESSION_TOKEN_SECRET"]) >= 40
    assert len(values["ADMIN_PASSWORD"]) >= 16
    assert urlsplit(values["DATABASE_URL"]).password == values["POSTGRES_PASSWORD"]
    assert values["LLM_MODEL"] == "qwen3:8b"  # untouched keys preserved
    assert env.read_text(encoding="utf-8").startswith("# example")
    if os.name == "posix":
        assert (env.stat().st_mode & 0o777) == 0o600


def test_existing_values_are_kept_unless_forced(tmp_path: Path, setup_env: ModuleType) -> None:
    example, env = tmp_path / ".env.example", tmp_path / ".env"
    example.write_text(EXAMPLE, encoding="utf-8")
    setup_env.setup_env(env, example)
    first = _read(env, setup_env)
    notes = setup_env.setup_env(env, example)
    assert _read(env, setup_env) == first
    assert "Kept existing ENCRYPTION_MASTER_KEY." in notes
    setup_env.setup_env(env, example, force=True)
    forced = _read(env, setup_env)
    for key in ("ENCRYPTION_MASTER_KEY", "SESSION_TOKEN_SECRET", "ADMIN_PASSWORD",
                "POSTGRES_PASSWORD"):
        assert forced[key] != first[key]
    assert urlsplit(forced["DATABASE_URL"]).password == forced["POSTGRES_PASSWORD"]


def test_mismatched_database_password_is_reported(tmp_path: Path, setup_env: ModuleType) -> None:
    env = tmp_path / ".env"
    env.write_text("POSTGRES_PASSWORD=abc\n"
                   "DATABASE_URL=postgresql+asyncpg://u:other@db:5432/x\n", encoding="utf-8")
    notes = setup_env.setup_env(env, tmp_path / "missing.example")
    assert any("WARNING" in n for n in notes)
    assert _read(env, setup_env)["DATABASE_URL"].endswith("u:other@db:5432/x")


def test_cli_does_not_print_secrets(tmp_path: Path, setup_env: ModuleType,
                                    capsys: pytest.CaptureFixture[str]) -> None:
    env = tmp_path / ".env"
    assert setup_env.main(["--env-file", str(env), "--example", str(tmp_path / "none")]) == 0
    out = capsys.readouterr().out
    for key, value in _read(env, setup_env).items():
        if key != "DATABASE_URL":
            assert value not in out
    assert "alembic upgrade head" in out


def test_existing_database_password_is_adopted(tmp_path: Path, setup_env: ModuleType) -> None:
    env = tmp_path / ".env"
    env.write_text("DATABASE_URL=postgresql+asyncpg://medbridge_app:s3cret@127.0.0.1:5433/"
                   "medbridge\n", encoding="utf-8")
    notes = setup_env.setup_env(env, tmp_path / "missing.example")
    values = _read(env, setup_env)
    assert values["POSTGRES_PASSWORD"] == "s3cret"
    assert values["DATABASE_URL"].endswith("medbridge_app:s3cret@127.0.0.1:5433/medbridge")
    assert "Adopted POSTGRES_PASSWORD from the existing DATABASE_URL." in notes
    assert not any("WARNING" in n for n in notes)
