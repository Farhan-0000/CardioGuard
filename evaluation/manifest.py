"""
Reproducibility metadata for evaluation runs (CLAUDE_AGENT_PROMPT §6.5):
git commit, model names + Ollama digests/quantisation, hardware, seeds and a
settings snapshot with every secret removed.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evaluation.paths import REPO_ROOT

_SECRET_KEY_RE = re.compile(r"(KEY|SECRET|PASSWORD|TOKEN)", re.IGNORECASE)
_URL_CREDENTIALS_RE = re.compile(r"(://[^:/@]+:)([^@]+)(@)")

EVAL_PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "matplotlib",
    "openpyxl",
    "sentence-transformers",
    "torch",
    "fastembed",
    "qdrant-client",
    "pydantic",
)


def utc_now() -> str:
    """Current UTC time in ISO-8601."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_info(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Commit hash and dirty flag (read-only git commands; empty dict when git is unavailable)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return {"commit": commit, "dirty": bool(status.strip())}


def _total_ram_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        pass
    if sys.platform == "win32":  # pragma: no cover - Windows only
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.ullTotalPhys)
    return None


def _cpu_brand() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return platform.processor()
    return platform.processor()


def hardware_info() -> dict[str, Any]:
    """Platform, CPU, RAM and Python version of the machine running the evaluation."""
    ram = _total_ram_bytes()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_brand(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(ram / 1024**3, 1) if ram else None,
        "python": sys.version.split()[0],
    }


def package_versions(packages: tuple[str, ...] = EVAL_PACKAGES) -> dict[str, str | None]:
    """Installed versions of the evaluation dependencies."""
    out: dict[str, str | None] = {}
    for name in packages:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def ollama_models(
    base_url: str, names: list[str] | None = None, timeout: float = 5.0
) -> dict[str, Any]:
    """Digest, size, parameter count and quantisation of local Ollama models (``/api/tags``).

    Returns ``{"error": ...}`` when Ollama is unreachable.
    """
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    try:
        resp = httpx.get(f"{url}/api/tags", timeout=timeout)
        resp.raise_for_status()
        models = resp.json().get("models", [])
    except (httpx.HTTPError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    out: dict[str, Any] = {}
    for model in models:
        name = str(model.get("name") or model.get("model") or "")
        if names is not None and name not in names and name.removesuffix(":latest") not in names:
            continue
        details = model.get("details") or {}
        out[name] = {
            "digest": model.get("digest"),
            "size_bytes": model.get("size"),
            "family": details.get("family"),
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "modified_at": model.get("modified_at"),
        }
    return out


def mask_url_credentials(text: str) -> str:
    """Replace the password of any ``scheme://user:password@host`` URL with ``***``."""
    return _URL_CREDENTIALS_RE.sub(r"\1***\3", text)


def redact_secrets(data: dict[str, Any]) -> dict[str, Any]:
    """Drop secret-looking keys and mask credentials embedded in URLs."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if _SECRET_KEY_RE.search(key):
            continue
        if isinstance(value, str):
            value = mask_url_credentials(value)
        out[key] = value
    return out


def apply_settings_env(values: dict[str, str | None]) -> None:
    """Override ``medbridge.config.Settings`` fields for this process (``None`` = keep).

    Environment variables take precedence over ``.env`` in pydantic-settings;
    the cached settings are dropped so every object built afterwards sees the
    overrides. Call this before any pipeline object is created.
    """
    for key, value in values.items():
        if value is not None:
            os.environ[key] = str(value)
    from medbridge.config import get_settings

    get_settings.cache_clear()


def settings_snapshot() -> dict[str, Any]:
    """Current ``medbridge.config.Settings`` without secrets."""
    from medbridge.config import get_settings

    return redact_secrets(get_settings().model_dump(mode="json"))
