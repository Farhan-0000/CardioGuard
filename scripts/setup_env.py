#!/usr/bin/env python3
"""
Create or complete ``.env`` with freshly generated secrets (cross-platform).

    python scripts/setup_env.py            # create .env from .env.example, fill empty secrets
    python scripts/setup_env.py --force    # regenerate the secrets even if already set

Generated values (never printed):

* ``ENCRYPTION_MASTER_KEY`` — Fernet key (master key for encryption at rest)
* ``SESSION_TOKEN_SECRET``  — HMAC secret for signed session/admin tokens
* ``ADMIN_PASSWORD``        — dashboard password (username: ``ADMIN_USERNAME``, default admin)
* ``POSTGRES_PASSWORD``     — database password used by docker-compose
* ``DATABASE_URL``          — rebuilt with the generated database password

Existing non-empty values are kept unless ``--force`` is given. Comments and
the order of lines in the file are preserved; missing keys are appended.
"""
from __future__ import annotations

import argparse
import base64
import os
import secrets
import sys
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_USER = "medbridge_app"
DB_HOST_PORT = "127.0.0.1:5433"
DB_NAME = "medbridge"
# DATABASE_URL passwords that are clearly placeholders and may be replaced.
PLACEHOLDER_PASSWORDS = {"", "password", "change-me", "changeme", "postgres"}


def fernet_key() -> str:
    """A urlsafe-base64 32-byte key (the Fernet key format)."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")


GENERATORS = {
    "ENCRYPTION_MASTER_KEY": fernet_key,
    "SESSION_TOKEN_SECRET": lambda: secrets.token_urlsafe(48),
    "ADMIN_PASSWORD": lambda: secrets.token_urlsafe(15),
    "POSTGRES_PASSWORD": lambda: secrets.token_urlsafe(24),
}


def parse_env(lines: list[str]) -> dict[str, str]:
    """Return KEY → value for ``KEY=value`` lines (last occurrence wins)."""
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def set_value(lines: list[str], key: str, value: str) -> list[str]:
    """Replace ``key``'s line in place, or append it."""
    out: list[str] = []
    replaced = False
    for line in lines:
        if line.strip().partition("=")[0].strip() == key and not line.lstrip().startswith("#"):
            if not replaced:
                out.append(f"{key}={value}")
                replaced = True
            continue
        out.append(line)
    if not replaced:
        out.append(f"{key}={value}")
    return out


def database_url_password(url: str) -> str | None:
    try:
        return urlsplit(url).password
    except ValueError:
        return None


def build_database_url(existing: str, password: str) -> str:
    """Return ``existing`` with its password replaced (or the default local URL)."""
    if existing:
        parts = urlsplit(existing)
        if parts.hostname:
            user = parts.username or DB_USER
            host = parts.hostname + (f":{parts.port}" if parts.port else "")
            netloc = f"{quote(user, safe='')}:{quote(password, safe='')}@{host}"
            return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return f"postgresql+asyncpg://{DB_USER}:{quote(password, safe='')}@{DB_HOST_PORT}/{DB_NAME}"


def setup_env(env_path: Path, example_path: Path, force: bool = False) -> list[str]:
    """Create/complete ``env_path``. Returns human-readable notes (no secret values)."""
    notes: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        notes.append(f"Updating existing {env_path.name}.")
    elif example_path.exists():
        lines = example_path.read_text(encoding="utf-8").splitlines()
        notes.append(f"Created {env_path.name} from {example_path.name}.")
    else:
        lines = ["# MedBridge local configuration"]
        notes.append(f"Created {env_path.name} (no {example_path.name} found).")

    values = parse_env(lines)
    url_password = database_url_password(values.get("DATABASE_URL", ""))
    if (not force and not values.get("POSTGRES_PASSWORD")
            and url_password and url_password not in PLACEHOLDER_PASSWORDS):
        # An existing database already uses this password: adopt it, never replace it.
        lines = set_value(lines, "POSTGRES_PASSWORD", url_password)
        values["POSTGRES_PASSWORD"] = url_password
        notes.append("Adopted POSTGRES_PASSWORD from the existing DATABASE_URL.")
    generated: set[str] = set()
    for key, generate in GENERATORS.items():
        if force or not values.get(key):
            lines = set_value(lines, key, generate())
            generated.add(key)
            notes.append(f"Generated {key}.")
        else:
            notes.append(f"Kept existing {key}.")
    values = parse_env(lines)

    db_password = values["POSTGRES_PASSWORD"]
    url = values.get("DATABASE_URL", "")
    current = database_url_password(url) if url else None
    if "POSTGRES_PASSWORD" in generated or force or (current or "") in PLACEHOLDER_PASSWORDS:
        if current != db_password:
            lines = set_value(lines, "DATABASE_URL", build_database_url(url, db_password))
            notes.append("Set DATABASE_URL to use POSTGRES_PASSWORD.")
    elif current != db_password:
        notes.append(
            "WARNING: DATABASE_URL uses a different password than POSTGRES_PASSWORD; "
            "edit one of them so they match."
        )
    if "POSTGRES_PASSWORD" in generated:
        notes.append(
            "NOTE: if a PostgreSQL volume already exists it keeps its old password. "
            "Either keep the old POSTGRES_PASSWORD/DATABASE_URL or recreate the volume "
            "(`docker compose down -v` deletes ALL local data)."
        )

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if os.name == "posix":
        env_path.chmod(0o600)
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true",
                        help="regenerate secrets even if they are already set")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--example", type=Path, default=REPO_ROOT / ".env.example")
    args = parser.parse_args(argv)

    for note in setup_env(args.env_file, args.example, force=args.force):
        print(f"  - {note}")
    print(f"\nSecrets were written to {args.env_file} (they are not printed here).")
    print("Next steps:")
    print("  1. docker compose up -d                 # PostgreSQL (127.0.0.1:5433) + Qdrant")
    print("  2. python -m alembic upgrade head       # create/upgrade the database schema")
    print("  3. ollama pull <LLM_MODEL from .env>    # local model(s)")
    print("  4. python -m uvicorn medbridge.main:app --host 127.0.0.1 --port 8000")
    print("Keep .env private: losing ENCRYPTION_MASTER_KEY makes stored conversations unreadable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
