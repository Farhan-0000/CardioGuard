"""
MedBridge command-line launcher (cross-platform: Windows 11, macOS, Linux).

Run every command from the repository root, inside the virtual environment::

    python -m medbridge.cli doctor             # check the machine and print a fix for every problem
    python -m medbridge.cli up                 # services + migrations + (first-run) ingestion + server
    python -m medbridge.cli up --open          # ... and open the web UI in the browser
    python -m medbridge.cli up --no-server     # prepare everything, do not start uvicorn
    python -m medbridge.cli ingest --recreate  # = python -m medbridge.ingestion --recreate
    python -m medbridge.cli smoke              # = python scripts/smoke_test.py --base-url http://HOST:PORT
    python -m medbridge.cli --dry-run up       # print what would run, change nothing

``up`` performs, in order:

1. create ``.env`` with ``scripts/setup_env.py`` when it does not exist yet;
2. ``docker compose up -d`` (PostgreSQL on 127.0.0.1:5433, Qdrant on 127.0.0.1:6333);
3. wait until PostgreSQL reports *healthy* and Qdrant answers ``/readyz``;
4. ``python -m alembic upgrade head``;
5. ``python -m medbridge.ingestion --no-manifest-update`` **only** when the Qdrant
   collection is missing or empty (incremental, so it is safe to repeat);
6. warn (not fail) when Ollama or a configured model is missing;
7. ``python -m uvicorn medbridge.main:app --host HOST --port PORT`` in the
   foreground (Ctrl+C stops it; the Docker services keep running).

The module imports only the standard library at import time, so ``doctor`` also
works before ``pip install -r requirements.txt`` (it then reports the missing
packages). Every external program is started with an argument list
(``subprocess`` without a shell) and ``cwd`` = the repository root. Secrets from
``.env`` are never printed.

Exit codes: 0 = success, 1 = a check or step failed, 2 = usage error.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import importlib.util
import ipaddress
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"
MANIFEST_FILE = REPO_ROOT / "data" / "corpus_manifest.json"
SETUP_ENV_SCRIPT = REPO_ROOT / "scripts" / "setup_env.py"
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_test.py"

SUPPORTED_PYTHON: tuple[tuple[int, int], ...] = ((3, 11), (3, 12))
COMPOSE_SERVICES: tuple[str, ...] = ("postgres", "qdrant")
DEFAULT_WAIT_SECONDS = 180.0
POLL_SECONDS = 2.0

# import name -> pip distribution name (runtime packages from requirements.txt).
REQUIRED_MODULES: dict[str, str] = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn[standard]",
    "pydantic_settings": "pydantic-settings",
    "sqlalchemy": "sqlalchemy[asyncio]",
    "asyncpg": "asyncpg",
    "alembic": "alembic",
    "psycopg2": "psycopg2-binary",
    "qdrant_client": "qdrant-client",
    "fastembed": "fastembed",
    "sentence_transformers": "sentence-transformers",
    "httpx": "httpx",
    "pymupdf": "pymupdf",
    "structlog": "structlog",
    "cryptography": "cryptography",
}
# Corpus documents that cannot be fetched by script (publisher browser check), see data/README.md.
MANUAL_DOCUMENTS: frozenset[str] = frozenset({"AHA_ACC_HTN_2025", "ESC_HTN_2024"})
# DATABASE_URL passwords that are obviously placeholders (scripts/setup_env.py replaces them).
PLACEHOLDER_PASSWORDS: frozenset[str] = frozenset(
    {"", "password", "change-me", "changeme", "postgres"}
)

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"
_LABEL = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[SKIP]"}


# ---------------------------------------------------------------------------
# Configuration view (what the launcher needs from medbridge.config.Settings)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    """The subset of the application settings the launcher uses."""

    database_url: str = "postgresql+asyncpg://medbridge_app:change-me@127.0.0.1:5433/medbridge"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "clinical_guidelines"
    ollama_base_url: str = "http://127.0.0.1:11434"
    models: tuple[str, ...] = ()
    llm_provider: str = "ollama"
    local_only: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    admin_username: str = "admin"
    source: str = "settings"

    @property
    def browse_host(self) -> str:
        """Host to put in a browser URL (a wildcard bind address is not browsable)."""
        return "127.0.0.1" if self.host in ("0.0.0.0", "::", "") else self.host

    @property
    def base_url(self) -> str:
        return f"http://{self.browse_host}:{self.port}"


def normalize_ollama_url(url: str) -> str:
    """``http://h:11434/v1/`` -> ``http://h:11434`` (the native API lives at the root)."""
    trimmed = url.strip().rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    return trimmed.rstrip("/")


def read_env_file(path: Path) -> dict[str, str]:
    """``KEY=value`` pairs of a dotenv file (comments ignored, quotes stripped)."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _as_bool(value: str, default: bool) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    return default


def _config_from_settings(env_file: Path) -> AppConfig:
    from medbridge.config import Settings  # needs pydantic-settings

    settings = Settings(_env_file=str(env_file) if env_file.is_file() else None)
    models = tuple(dict.fromkeys([settings.gates_model, settings.generator_model]))
    return AppConfig(
        database_url=settings.DATABASE_URL,
        qdrant_url=settings.QDRANT_URL.rstrip("/"),
        qdrant_collection=settings.QDRANT_COLLECTION,
        ollama_base_url=normalize_ollama_url(settings.OLLAMA_BASE_URL),
        models=models,
        llm_provider=settings.LLM_PROVIDER,
        local_only=settings.LOCAL_ONLY_MODE,
        host=settings.HOST,
        port=int(settings.PORT),
        admin_username=settings.ADMIN_USERNAME,
        source="settings",
    )


_CONFIG_KEYS: frozenset[str] = frozenset({
    "DATABASE_URL", "QDRANT_URL", "QDRANT_COLLECTION", "OLLAMA_BASE_URL", "LLM_MODEL",
    "LLM_MODEL_GATES", "LLM_MODEL_GENERATOR", "LLM_PROVIDER", "LOCAL_ONLY_MODE", "HOST", "PORT",
    "ADMIN_USERNAME",
})


def _config_fallback(env_file: Path, environ: Mapping[str, str]) -> AppConfig:
    """Best-effort view when ``medbridge.config`` cannot be imported (packages missing)."""
    values = read_env_file(env_file)
    values.update({k: v for k, v in environ.items() if k in _CONFIG_KEYS})
    default = AppConfig()

    def get(key: str, fallback: str) -> str:
        return values.get(key) or fallback

    model = get("LLM_MODEL", "")
    models = tuple(
        m for m in dict.fromkeys([get("LLM_MODEL_GATES", model), get("LLM_MODEL_GENERATOR", model)])
        if m
    )
    try:
        port = int(get("PORT", str(default.port)))
    except ValueError:
        port = default.port
    return AppConfig(
        database_url=get("DATABASE_URL", default.database_url),
        qdrant_url=get("QDRANT_URL", default.qdrant_url).rstrip("/"),
        qdrant_collection=get("QDRANT_COLLECTION", default.qdrant_collection),
        ollama_base_url=normalize_ollama_url(get("OLLAMA_BASE_URL", default.ollama_base_url)),
        models=models,
        llm_provider=get("LLM_PROVIDER", default.llm_provider),
        local_only=_as_bool(get("LOCAL_ONLY_MODE", "true"), True),
        host=get("HOST", default.host),
        port=port,
        admin_username=get("ADMIN_USERNAME", default.admin_username),
        source="fallback",
    )


def load_config(env_file: Path = ENV_FILE, environ: Mapping[str, str] | None = None) -> AppConfig:
    """Application settings (environment variables override ``env_file``, as in the app)."""
    try:
        return _config_from_settings(env_file)
    except Exception:  # ImportError before installation, or an invalid value in .env
        return _config_fallback(env_file, os.environ if environ is None else environ)


# ---------------------------------------------------------------------------
# Side-effect boundary (injectable for hermetic tests)
# ---------------------------------------------------------------------------


def tcp_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """True when something accepts TCP connections on ``host:port``."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def http_get_json(url: str, timeout: float = 3.0) -> tuple[int | None, Any]:
    """``(status, parsed JSON or None)``; status ``None`` when the server is unreachable."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code), None
    except (urllib.error.URLError, OSError, ValueError):
        return None, None
    try:
        return status, json.loads(body.decode("utf-8")) if body else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status, None


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    # Child processes print non-ASCII text (e.g. dashes in document titles); never crash on
    # a legacy Windows console code page.
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _default_run(
    cmd: Sequence[str], capture: bool, timeout: float | None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=REPO_ROOT,
        env=_child_env(),
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _default_serve(cmd: Sequence[str]) -> int:
    """Run a long-lived foreground process; Ctrl+C stops it cleanly."""
    proc = subprocess.Popen(list(cmd), cwd=REPO_ROOT, env=_child_env())
    try:
        return proc.wait()
    except KeyboardInterrupt:
        try:
            return proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.terminate()
            return proc.wait()


@dataclass
class Tools:
    """Everything that touches the outside world. Tests replace these callables."""

    run: Callable[[Sequence[str], bool, float | None], subprocess.CompletedProcess[str]] = (
        _default_run
    )
    serve: Callable[[Sequence[str]], int] = _default_serve
    which: Callable[[str], str | None] = shutil.which
    tcp_open: Callable[[str, int], bool] = tcp_open
    http_get_json: Callable[[str], tuple[int | None, Any]] = http_get_json
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic
    out: Callable[[str], None] = print
    open_browser: Callable[[str], Any] = webbrowser.open
    python: str = sys.executable
    environ: Mapping[str, str] = field(default_factory=lambda: os.environ)
    version_info: tuple[int, int, int] = (
        sys.version_info.major, sys.version_info.minor, sys.version_info.micro
    )
    in_venv: bool = sys.prefix != sys.base_prefix or bool(os.environ.get("VIRTUAL_ENV"))
    platform_system: str = platform.system()
    find_spec: Callable[[str], object | None] = importlib.util.find_spec
    torch_probe: Callable[[], dict[str, Any]] | None = None
    env_file: Path = ENV_FILE
    manifest_file: Path = MANIFEST_FILE
    repo_root: Path = REPO_ROOT


def format_cmd(cmd: Sequence[str]) -> str:
    """Human-readable command line for logs and ``--dry-run``."""
    if os.name == "nt":
        return subprocess.list2cmdline(list(cmd))
    return shlex.join(list(cmd))


class Launcher:
    """Runs commands (or prints them in dry-run mode) and reports progress."""

    def __init__(self, tools: Tools, dry_run: bool = False) -> None:
        self.tools = tools
        self.dry_run = dry_run

    def say(self, message: str = "") -> None:
        self.tools.out(message)

    def step(self, message: str) -> None:
        self.say(f"==> {message}")

    def run(
        self, cmd: Sequence[str], *, capture: bool = False, timeout: float | None = None,
        note: str = "",
    ) -> subprocess.CompletedProcess[str]:
        if self.dry_run:
            suffix = f"   ({note})" if note else ""
            self.say(f"[dry-run] {format_cmd(cmd)}{suffix}")
            return subprocess.CompletedProcess(list(cmd), 0, "", "")
        if not capture:
            self.say(f"$ {format_cmd(cmd)}")
        try:
            return self.tools.run(cmd, capture, timeout)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(list(cmd), 124, "", f"timed out after {timeout} s")
        except OSError as exc:  # program not found / not executable
            return subprocess.CompletedProcess(list(cmd), 127, "", str(exc))

    def serve(self, cmd: Sequence[str]) -> int:
        if self.dry_run:
            self.say(f"[dry-run] {format_cmd(cmd)}   (foreground; Ctrl+C stops it)")
            return 0
        self.say(f"$ {format_cmd(cmd)}")
        return self.tools.serve(cmd)

    def py(self, *args: str) -> list[str]:
        return [self.tools.python, *args]


# ---------------------------------------------------------------------------
# Helpers shared by doctor and up
# ---------------------------------------------------------------------------


def split_host_port(url: str, default_port: int) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(url.replace("+asyncpg", ""))
    return parsed.hostname or "127.0.0.1", parsed.port or default_port


def url_password(url: str) -> str | None:
    try:
        return urllib.parse.urlparse(url).password
    except ValueError:
        return None


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def model_present(model: str, available: Sequence[str]) -> bool:
    """Ollama tag matching: ``llama3`` means ``llama3:latest``."""
    wanted = model if ":" in model else f"{model}:latest"
    return any(name in (model, wanted) for name in available)


def ollama_models(tools: Tools, cfg: AppConfig) -> list[str] | None:
    """Model tags known to the Ollama server, or ``None`` when it is unreachable."""
    status, body = tools.http_get_json(f"{cfg.ollama_base_url}/api/tags")
    if status != 200 or not isinstance(body, dict):
        return None
    return [
        str(m.get("name") or m.get("model"))
        for m in body.get("models", [])
        if isinstance(m, dict)
    ]


def qdrant_points(tools: Tools, cfg: AppConfig) -> tuple[str, int | None]:
    """``("unreachable"|"missing"|"ok"|"error", points)`` for the guideline collection."""
    status, body = tools.http_get_json(f"{cfg.qdrant_url}/collections/{cfg.qdrant_collection}")
    if status is None:
        return "unreachable", None
    if status == 404:
        return "missing", None
    if status != 200 or not isinstance(body, dict):
        return "error", None
    result = body.get("result") or {}
    points = result.get("points_count")
    if points is None:
        points = result.get("vectors_count")
    return "ok", int(points or 0)


def qdrant_ready(tools: Tools, cfg: AppConfig) -> bool:
    status, _ = tools.http_get_json(f"{cfg.qdrant_url}/readyz")
    if status == 200:
        return True
    if status == 404:  # older Qdrant without /readyz
        status, _ = tools.http_get_json(f"{cfg.qdrant_url}/collections")
        return status == 200
    return False


def compose_command(tools: Tools) -> list[str] | None:
    """``[docker, compose]`` (Compose v2) or ``[docker-compose]``; ``None`` if unavailable."""
    docker = tools.which("docker")
    if docker:
        try:
            probe = tools.run([docker, "compose", "version"], True, 30)
            if probe.returncode == 0:
                return [docker, "compose"]
        except (OSError, subprocess.SubprocessError):
            pass
    legacy = tools.which("docker-compose")
    return [legacy] if legacy else None


def container_health(tools: Tools, compose: Sequence[str], docker: str, service: str) -> str:
    """Docker health of a compose service: healthy | starting | unhealthy | none | missing."""
    try:
        ids = tools.run([*compose, "ps", "-q", service], True, 30)
    except (OSError, subprocess.SubprocessError):
        return "missing"
    container = (ids.stdout or "").strip().splitlines()
    if ids.returncode != 0 or not container:
        return "missing"
    try:
        inspected = tools.run(
            [docker, "inspect", "-f", "{{if .State.Health}}{{.State.Health.Status}}"
             "{{else}}none{{end}}", container[0]],
            True, 30,
        )
    except (OSError, subprocess.SubprocessError):
        return "missing"
    status = (inspected.stdout or "").strip().lower()
    return status if inspected.returncode == 0 and status else "missing"


def corpus_status(manifest_file: Path, repo_root: Path) -> tuple[int, list[str]]:
    """``(documents in the manifest, ids whose file is missing)``; ``(-1, [])`` if unreadable."""
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return -1, []
    documents = manifest.get("documents", []) if isinstance(manifest, dict) else []
    missing = [
        str(doc.get("guideline_id"))
        for doc in documents
        if isinstance(doc, dict) and not (repo_root / str(doc.get("file", ""))).is_file()
    ]
    return len(documents), missing


def activation_hint(system: str) -> str:
    if system == "Windows":
        return r".\.venv\Scripts\Activate.ps1"
    return "source .venv/bin/activate"


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    fix: str = ""


def check_python(tools: Tools) -> CheckResult:
    major, minor, micro = tools.version_info
    version = f"{major}.{minor}.{micro}"
    if (major, minor) in SUPPORTED_PYTHON:
        return CheckResult("Python", OK, f"{version} ({tools.python})")
    if (major, minor) < SUPPORTED_PYTHON[0]:
        return CheckResult(
            "Python", FAIL, f"{version} is too old (3.11 or 3.12 required)",
            "Install Python 3.11 and recreate the virtual environment "
            "(Windows: py -3.11 -m venv .venv; macOS: python3.11 -m venv .venv).",
        )
    return CheckResult(
        "Python", WARN, f"{version} is newer than the tested versions (3.11, 3.12)",
        "If installation fails, use Python 3.11.",
    )


def check_venv(tools: Tools) -> CheckResult:
    if tools.in_venv:
        return CheckResult("Virtual env", OK, "running inside a virtual environment")
    return CheckResult(
        "Virtual env", WARN, "not active (packages would go to the system Python)",
        f"Activate it first: {activation_hint(tools.platform_system)}",
    )


def check_packages(tools: Tools) -> CheckResult:
    missing = []
    for module, dist in REQUIRED_MODULES.items():
        try:
            spec = tools.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(dist)
    if not missing:
        return CheckResult("Python packages", OK, f"{len(REQUIRED_MODULES)} runtime packages found")
    return CheckResult(
        "Python packages", FAIL, "missing: " + ", ".join(missing),
        "python -m pip install -r requirements.txt  (inside the activated virtual environment)",
    )


def _probe_torch() -> dict[str, Any]:
    import torch  # heavy import, only on demand

    info: dict[str, Any] = {"version": str(torch.__version__), "device": "cpu", "name": ""}
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["name"] = str(torch.cuda.get_device_name(0))
    else:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            info["device"] = "mps"
    info["cuda_build"] = str(getattr(torch.version, "cuda", None) or "")
    return info


def check_torch(tools: Tools) -> CheckResult:
    try:
        spec = tools.find_spec("torch")
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return CheckResult(
            "PyTorch", FAIL, "not installed (needed by the reranker and the NLI models)",
            "python -m pip install -r requirements.txt (Windows + NVIDIA: install the CUDA "
            "build of torch first, see HOW_TO_RUN.md).",
        )
    try:
        info = (tools.torch_probe or _probe_torch)()
    except Exception as exc:
        return CheckResult("PyTorch", WARN, f"import failed: {type(exc).__name__}: {exc}",
                           "Reinstall torch (see HOW_TO_RUN.md, 'CUDA not used by torch').")
    device = str(info.get("device", "cpu"))
    detail = f"{info.get('version', '?')} on {device}"
    if info.get("name"):
        detail += f" ({info['name']})"
    has_nvidia = tools.which("nvidia-smi") is not None
    if device == "cpu" and has_nvidia and tools.platform_system == "Windows":
        return CheckResult(
            "PyTorch", WARN, detail + " - an NVIDIA GPU is present but this torch build is CPU-only",
            "pip uninstall -y torch, then pip install torch --index-url "
            "https://download.pytorch.org/whl/cu126 (optional: the reranker also works on CPU).",
        )
    return CheckResult("PyTorch", OK, detail)


def _fernet_key_ok(key: str) -> bool:
    try:
        return len(base64.urlsafe_b64decode(key.encode("ascii"))) == 32
    except (binascii.Error, ValueError, UnicodeEncodeError):
        return False


def check_env(tools: Tools) -> list[CheckResult]:
    """Inspect ``.env`` without printing any secret value."""
    path = tools.env_file
    if not path.is_file():
        return [CheckResult(
            ".env", FAIL, f"{path.name} not found",
            "python scripts/setup_env.py   (creates .env from .env.example with fresh secrets)",
        )]
    file_values = read_env_file(path)

    def value(key: str) -> str:
        if key in tools.environ:
            return str(tools.environ[key])
        return file_values.get(key, "")

    results: list[CheckResult] = []
    problems: list[str] = []
    missing = [k for k in ("ENCRYPTION_MASTER_KEY", "SESSION_TOKEN_SECRET", "POSTGRES_PASSWORD")
               if not value(k)]
    if missing:
        problems.append("empty: " + ", ".join(missing))
    key = value("ENCRYPTION_MASTER_KEY")
    if key and not _fernet_key_ok(key):
        results.append(CheckResult(
            ".env encryption key", WARN,
            "ENCRYPTION_MASTER_KEY is not a 32-byte urlsafe-base64 (Fernet) key; the app derives one",
            "Prefer a generated key: python scripts/setup_env.py (only before any data is stored).",
        ))
    db_password = url_password(value("DATABASE_URL"))
    pg_password = value("POSTGRES_PASSWORD")
    if pg_password and db_password is not None and db_password != pg_password:
        if db_password in PLACEHOLDER_PASSWORDS:
            problems.append("DATABASE_URL still has a placeholder password")
        else:
            problems.append("DATABASE_URL password differs from POSTGRES_PASSWORD")
    if problems:
        results.insert(0, CheckResult(
            ".env", FAIL, "; ".join(problems),
            "python scripts/setup_env.py fills empty secrets and aligns DATABASE_URL "
            "(never change POSTGRES_PASSWORD after the database volume exists).",
        ))
    else:
        results.insert(0, CheckResult(".env", OK, "secrets present, DATABASE_URL matches"))
    if not value("ADMIN_PASSWORD"):
        results.append(CheckResult(
            ".env admin", WARN, "ADMIN_PASSWORD is empty: the /dashboard login is disabled",
            "python scripts/setup_env.py generates one (username: ADMIN_USERNAME, default admin).",
        ))
    host = value("HOST") or "127.0.0.1"
    if host in ("0.0.0.0", "::"):
        results.append(CheckResult(
            ".env host", WARN, f"HOST={host} exposes the API to your network",
            "Use HOST=127.0.0.1 (the default) unless you really need LAN access.",
        ))
    return results


def check_docker(tools: Tools) -> tuple[CheckResult, list[str] | None]:
    docker = tools.which("docker")
    if not docker:
        return CheckResult(
            "Docker", FAIL, "docker command not found",
            "Install Docker Desktop (Windows: WSL 2 backend) and open it once.",
        ), None
    try:
        info = tools.run([docker, "info", "--format", "{{.ServerVersion}}"], True, 30)
    except (OSError, subprocess.SubprocessError) as exc:
        info = subprocess.CompletedProcess([docker], 1, "", str(exc))
    if info.returncode != 0:
        hint = ("Start Docker Desktop and wait until it says 'Engine running'"
                + (" (Settings > General > 'Use the WSL 2 based engine')."
                   if tools.platform_system == "Windows" else "."))
        return CheckResult("Docker", FAIL, "the Docker engine is not running", hint), None
    compose = compose_command(tools)
    if compose is None:
        return CheckResult(
            "Docker", FAIL, "Docker Compose v2 not available",
            "Update Docker Desktop (it ships 'docker compose').",
        ), None
    return CheckResult("Docker", OK, f"engine {(info.stdout or '').strip() or 'running'}"), compose


def check_postgres(tools: Tools, cfg: AppConfig) -> CheckResult:
    host, port = split_host_port(cfg.database_url, 5432)
    if tools.tcp_open(host, port):
        return CheckResult("PostgreSQL", OK, f"{host}:{port} reachable")
    return CheckResult(
        "PostgreSQL", FAIL, f"nothing listens on {host}:{port}",
        "docker compose up -d   (then: docker compose ps  -> postgres 'healthy')",
    )


def check_migrations(
    tools: Tools, launcher: Launcher, postgres_ok: bool, packages_ok: bool = True
) -> CheckResult:
    if not packages_ok:
        return CheckResult("Migrations", SKIP, "Python packages missing (install them first)")
    if not postgres_ok:
        return CheckResult("Migrations", SKIP, "PostgreSQL not reachable")
    result = launcher.run(launcher.py("-m", "alembic", "current"), capture=True, timeout=90)
    output = f"{result.stdout or ''}\n{result.stderr or ''}"
    if result.returncode == 0 and "(head)" in output:
        return CheckResult("Migrations", OK, "database schema at head")
    if result.returncode == 0:
        return CheckResult(
            "Migrations", WARN, "database schema is not at the latest revision",
            "python -m alembic upgrade head",
        )
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    last = lines[-1][:160] if lines else f"exit code {result.returncode}"
    fix = "python -m alembic upgrade head"
    if "password authentication failed" in output:
        fix = ("The database volume was created with another password: restore the old "
               "POSTGRES_PASSWORD/DATABASE_URL, or reset (docker compose down -v deletes ALL data).")
    return CheckResult("Migrations", FAIL, last, fix)


def check_qdrant(tools: Tools, cfg: AppConfig) -> CheckResult:
    state, points = qdrant_points(tools, cfg)
    if state == "unreachable":
        return CheckResult("Qdrant", FAIL, f"{cfg.qdrant_url} not reachable",
                           "docker compose up -d   (then: docker compose ps -> qdrant 'healthy')")
    if state == "missing":
        return CheckResult("Qdrant", FAIL, f"collection '{cfg.qdrant_collection}' does not exist",
                           "python -m medbridge.ingestion   (after downloading the corpus)")
    if state == "error":
        return CheckResult("Qdrant", WARN, "unexpected answer from Qdrant",
                           "docker compose logs qdrant")
    if not points:
        return CheckResult("Qdrant", FAIL, f"collection '{cfg.qdrant_collection}' is empty",
                           "python -m medbridge.ingestion")
    return CheckResult("Qdrant", OK, f"collection '{cfg.qdrant_collection}': {points} points")


def check_corpus(tools: Tools) -> CheckResult:
    total, missing = corpus_status(tools.manifest_file, tools.repo_root)
    if total < 0:
        return CheckResult("Corpus", FAIL, "data/corpus_manifest.json not readable",
                           "Restore it from git (git checkout -- data/corpus_manifest.json).")
    if not missing:
        return CheckResult("Corpus", OK, f"all {total} documents present in data/guidelines")
    manual = sorted(set(missing) & MANUAL_DOCUMENTS)
    fix = "python scripts/download_corpus.py"
    if manual:
        fix += "; manual browser download (data/README.md): " + ", ".join(manual)
    status = FAIL if len(missing) == total else WARN
    shown = ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")
    return CheckResult("Corpus", status, f"{len(missing)}/{total} files missing: {shown}", fix)


def check_ollama(tools: Tools, cfg: AppConfig) -> CheckResult:
    if cfg.llm_provider != "ollama":
        return CheckResult("Ollama", SKIP, f"LLM_PROVIDER={cfg.llm_provider}")
    host, _port = split_host_port(cfg.ollama_base_url, 11434)
    if cfg.local_only and not is_loopback_host(host):
        return CheckResult(
            "Ollama", FAIL, f"OLLAMA_BASE_URL host '{host}' is not loopback; "
            "LOCAL_ONLY_MODE=true blocks every LLM call",
            "Use OLLAMA_BASE_URL=http://127.0.0.1:11434",
        )
    available = ollama_models(tools, cfg)
    installed = tools.which("ollama") is not None
    if available is None:
        fix = ("Start Ollama: open the Ollama app (or run 'ollama serve' in a separate terminal)."
               if installed else "Install Ollama from https://ollama.com/download and start it.")
        return CheckResult("Ollama", FAIL, f"{cfg.ollama_base_url} not reachable", fix)
    missing = [m for m in cfg.models if not model_present(m, available)]
    if missing:
        return CheckResult(
            "Ollama", FAIL, "model(s) not pulled: " + ", ".join(missing),
            "; ".join(f"ollama pull {m}" for m in missing),
        )
    return CheckResult("Ollama", OK, "models available: " + ", ".join(cfg.models or ("-",)))


def check_port(tools: Tools, cfg: AppConfig) -> CheckResult:
    if tools.tcp_open(cfg.browse_host, cfg.port):
        return CheckResult(
            "API port", WARN, f"{cfg.browse_host}:{cfg.port} is already in use",
            f"MedBridge may already be running (open {cfg.base_url}); otherwise stop the other "
            f"program or start on another port: python -m medbridge.cli up --port {cfg.port + 1}",
        )
    return CheckResult("API port", OK, f"{cfg.browse_host}:{cfg.port} is free")


def run_doctor(tools: Tools, launcher: Launcher) -> list[CheckResult]:
    cfg = load_config(tools.env_file, tools.environ)
    packages = check_packages(tools)
    results = [check_python(tools), check_venv(tools), packages, check_torch(tools)]
    results += check_env(tools)
    docker_result, _compose = check_docker(tools)
    results.append(docker_result)
    postgres = check_postgres(tools, cfg)
    results.append(postgres)
    results.append(check_migrations(tools, launcher, postgres.status == OK, packages.status == OK))
    results.append(check_qdrant(tools, cfg))
    results.append(check_corpus(tools))
    results.append(check_ollama(tools, cfg))
    results.append(check_port(tools, cfg))
    return results


DOCTOR_PLAN = (
    "Python version (3.11/3.12)", "virtual environment active", "runtime packages importable",
    "PyTorch device (cuda / mps / cpu)", ".env secrets (values are never printed)",
    "Docker engine + Compose v2", "PostgreSQL port", "Alembic revision (python -m alembic current)",
    "Qdrant collection and point count", "corpus files listed in data/corpus_manifest.json",
    "Ollama server + configured models", "API port free",
)


def cmd_doctor(args: argparse.Namespace, tools: Tools) -> int:
    launcher = Launcher(tools, dry_run=False)
    if args.dry_run:
        tools.out("[dry-run] doctor would check (read-only):")
        for item in DOCTOR_PLAN:
            tools.out(f"  - {item}")
        return 0
    results = run_doctor(tools, launcher)
    if args.json:
        tools.out(json.dumps([asdict(r) for r in results], indent=2))
    else:
        tools.out(f"MedBridge doctor  ({tools.platform_system}, repo: {tools.repo_root.name})")
        width = max(len(r.name) for r in results)
        for r in results:
            tools.out(f"{_LABEL[r.status]} {r.name.ljust(width)}  {r.detail}")
            if r.fix and r.status in (WARN, FAIL):
                tools.out(f"       {' ' * width}  fix: {r.fix}")
        counts = {s: sum(1 for r in results if r.status == s) for s in (OK, WARN, FAIL, SKIP)}
        tools.out("")
        tools.out(f"{counts[OK]} ok, {counts[WARN]} warning(s), {counts[FAIL]} problem(s), "
                  f"{counts[SKIP]} skipped")
        if counts[FAIL]:
            tools.out("Fix the [FAIL] items from top to bottom, then run the doctor again.")
    return 1 if any(r.status == FAIL for r in results) else 0


# ---------------------------------------------------------------------------
# up
# ---------------------------------------------------------------------------


def wait_for_services(
    launcher: Launcher, cfg: AppConfig, compose: Sequence[str] | None, timeout: float
) -> bool:
    """Wait until PostgreSQL is healthy/reachable and Qdrant is ready."""
    tools = launcher.tools
    if launcher.dry_run:
        launcher.say(f"[dry-run] wait until postgres is healthy and qdrant answers /readyz "
                     f"(timeout {timeout:.0f} s)")
        return True
    docker = tools.which("docker") if compose else None
    pg_host, pg_port = split_host_port(cfg.database_url, 5432)
    deadline = tools.monotonic() + timeout
    last = ""
    while True:
        pg_health = (container_health(tools, compose, docker, "postgres")
                     if compose and docker else "none")
        if pg_health in ("none", "missing"):
            pg_ok = tools.tcp_open(pg_host, pg_port)
        else:
            pg_ok = pg_health == "healthy"
        qd_ok = qdrant_ready(tools, cfg)
        state = f"postgres={'ready' if pg_ok else pg_health}, qdrant={'ready' if qd_ok else 'waiting'}"
        if state != last:
            launcher.say(f"    {state}")
            last = state
        if pg_ok and qd_ok:
            return True
        if tools.monotonic() >= deadline:
            return False
        tools.sleep(POLL_SECONDS)


def _open_when_ready(tools: Tools, cfg: AppConfig, timeout: float = 180.0) -> None:
    deadline = tools.monotonic() + timeout
    while tools.monotonic() < deadline:
        if tools.tcp_open(cfg.browse_host, cfg.port):
            tools.open_browser(cfg.base_url)
            return
        tools.sleep(1.0)


def cmd_up(args: argparse.Namespace, tools: Tools) -> int:
    launcher = Launcher(tools, dry_run=args.dry_run)

    # 1. .env
    if not tools.env_file.is_file():
        launcher.step(".env not found: creating it with fresh secrets")
        result = launcher.run(launcher.py(str(SETUP_ENV_SCRIPT)))
        if result.returncode != 0:
            launcher.say("ERROR: scripts/setup_env.py failed; fix the error above and retry.")
            return 1

    cfg = load_config(tools.env_file, tools.environ)
    host = args.host or cfg.host
    port = int(args.port or cfg.port)
    cfg = replace(cfg, host=host, port=port)

    # 2. Docker services
    compose: list[str] | None = None
    if args.skip_docker:
        launcher.step("Skipping Docker (--skip-docker): using the services in DATABASE_URL/QDRANT_URL")
    else:
        launcher.step("Starting PostgreSQL + Qdrant (docker compose up -d; the first run "
                      "downloads the images)")
        compose = ["docker", "compose"] if launcher.dry_run else compose_command(tools)
        if compose is None:
            launcher.say("ERROR: Docker Compose is not available. Install/start Docker Desktop, "
                         "then run: python -m medbridge.cli doctor")
            return 1
        result = launcher.run([*compose, "up", "-d"])
        if result.returncode != 0:
            launcher.say("ERROR: 'docker compose up -d' failed. Is Docker Desktop running? "
                         "Is port 5433 or 6333 used by another program? "
                         "(python -m medbridge.cli doctor shows details)")
            return 1

    # 3. wait for health
    launcher.step("Waiting for the services to become healthy")
    if not wait_for_services(launcher, cfg, compose, args.wait):
        launcher.say(f"ERROR: services not healthy after {args.wait:.0f} s. "
                     "Check: docker compose ps   and   docker compose logs postgres qdrant")
        return 1

    # 4. migrations
    launcher.step("Applying database migrations (alembic upgrade head)")
    result = launcher.run(launcher.py("-m", "alembic", "upgrade", "head"))
    if result.returncode != 0:
        launcher.say("ERROR: migrations failed. If the log says 'password authentication failed', "
                     "the database volume was created with a different POSTGRES_PASSWORD "
                     "(see HOW_TO_RUN.md, Troubleshooting).")
        return 1

    # 5. first-run ingestion
    ingest_cmd = launcher.py("-m", "medbridge.ingestion", "--no-manifest-update")
    if args.no_ingest:
        launcher.step("Skipping ingestion (--no-ingest)")
    elif launcher.dry_run:
        launcher.step("Indexing the corpus if the Qdrant collection is empty")
        launcher.run(ingest_cmd, note="only if the collection is missing or empty")
    else:
        state, points = qdrant_points(tools, cfg)
        if state == "ok" and points:
            launcher.step(f"Knowledge base ready: {points} chunks in '{cfg.qdrant_collection}'")
        else:
            total, missing = corpus_status(tools.manifest_file, tools.repo_root)
            if total > 0 and len(missing) == total:
                launcher.say("WARNING: the Qdrant collection is empty and no corpus file is "
                             "present. Run: python scripts/download_corpus.py (and the manual "
                             "downloads in data/README.md), then: python -m medbridge.cli ingest")
            else:
                launcher.step("Qdrant collection is empty: indexing the corpus (first run takes "
                              "a few minutes; embedding models are downloaded once)")
                result = launcher.run(ingest_cmd)
                if result.returncode != 0:
                    launcher.say("WARNING: ingestion reported failures (often a missing manual "
                                 "download). The server starts anyway; see the log above.")

    # 6. LLM availability (warning only: the pipeline degrades to safe templates)
    if not launcher.dry_run:
        check = check_ollama(tools, cfg)
        if check.status == FAIL:
            launcher.say(f"WARNING: {check.detail}. Fix: {check.fix}")
            launcher.say("         The server starts, but answers fall back to safe templates "
                         "until the model is available.")

    if args.no_server:
        launcher.step("Done (--no-server). Start the API later with: python -m medbridge.cli up")
        return 0

    # 7. API server
    if not launcher.dry_run and tools.tcp_open(cfg.browse_host, port):
        launcher.say(f"ERROR: {cfg.browse_host}:{port} is already in use. MedBridge may already "
                     f"be running (open {cfg.base_url}); otherwise use --port {port + 1}.")
        return 1
    launcher.step(f"Starting the API on {cfg.base_url}  (web UI: {cfg.base_url}/ , "
                  f"dashboard: {cfg.base_url}/dashboard, user '{cfg.admin_username}', password = "
                  "ADMIN_PASSWORD in .env)")
    launcher.say("    The first request after start-up can be slow while models warm up. "
                 "Press Ctrl+C to stop.")
    if args.open and not launcher.dry_run:
        threading.Thread(target=_open_when_ready, args=(tools, cfg), daemon=True).start()
    return launcher.serve(launcher.py(
        "-m", "uvicorn", "medbridge.main:app", "--host", host, "--port", str(port),
    ))


# ---------------------------------------------------------------------------
# ingest / smoke
# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace, tools: Tools, extra: Sequence[str]) -> int:
    launcher = Launcher(tools, dry_run=args.dry_run)
    return launcher.run(launcher.py("-m", "medbridge.ingestion", *extra)).returncode


def cmd_smoke(args: argparse.Namespace, tools: Tools, extra: Sequence[str]) -> int:
    launcher = Launcher(tools, dry_run=args.dry_run)
    forwarded = list(extra)
    if not any(a == "--base-url" or a.startswith("--base-url=") for a in forwarded):
        cfg = load_config(tools.env_file, tools.environ)
        forwarded = ["--base-url", cfg.base_url, *forwarded]
    return launcher.run(launcher.py(str(SMOKE_SCRIPT), *forwarded)).returncode


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m medbridge.cli",
        allow_abbrev=False,
        description="MedBridge launcher: check the machine, start the stack, ingest, smoke-test.",
        epilog="Run from the repository root inside the virtual environment. "
               "See HOW_TO_RUN.md for the full guide.",
    )
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="print the commands that would run; change nothing")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text, allow_abbrev=False)
        # SUPPRESS: '--dry-run' is accepted before or after the sub-command.
        p.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS,
                       help="print the commands that would run; change nothing")
        return p

    doctor = add("doctor", "check Python, packages, .env, Docker, PostgreSQL, migrations, Qdrant, "
                           "corpus, Ollama models and the API port; print a fix for each problem")
    doctor.add_argument("--json", action="store_true", help="machine-readable output")

    up = add("up", "docker compose up -d, wait until healthy, alembic upgrade head, ingest if the "
                   "collection is empty, then start uvicorn")
    up.add_argument("--host", default=None, help="bind address (default: HOST from .env)")
    up.add_argument("--port", type=int, default=None, help="port (default: PORT from .env)")
    up.add_argument("--wait", type=float, default=DEFAULT_WAIT_SECONDS,
                    help=f"seconds to wait for healthy services (default {DEFAULT_WAIT_SECONDS:.0f})")
    up.add_argument("--no-ingest", action="store_true", help="never run the ingestion step")
    up.add_argument("--no-server", action="store_true", help="prepare everything, do not start uvicorn")
    up.add_argument("--skip-docker", action="store_true",
                    help="do not call docker compose (services are managed elsewhere)")
    up.add_argument("--open", action="store_true", help="open the web UI in the browser when ready")

    add("ingest", "run 'python -m medbridge.ingestion' with any extra arguments "
                  "(e.g. --recreate, --stats, --only ID ...)")
    add("smoke", "run scripts/smoke_test.py against the running API "
                 "(extra arguments are forwarded, e.g. --timeout 300)")
    return parser


def main(argv: Sequence[str] | None = None, tools: Tools | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    tools = tools or Tools()
    if args.command is None:
        parser.print_help()
        return 2
    if extra and args.command not in ("ingest", "smoke"):
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if extra[:1] == ["--"]:  # 'ingest -- --dry-run' forwards ingestion's own --dry-run
        extra = extra[1:]
    if args.command == "doctor":
        return cmd_doctor(args, tools)
    if args.command == "up":
        return cmd_up(args, tools)
    if args.command == "ingest":
        return cmd_ingest(args, tools, extra)
    return cmd_smoke(args, tools, extra)


if __name__ == "__main__":
    sys.exit(main())
