"""Hermetic tests for the ``python -m medbridge.cli`` launcher.

No subprocess, socket, HTTP request or browser is real: every side effect goes
through an injected :class:`medbridge.cli.Tools` whose callables are fakes, and
``load_config`` is replaced so the developer ``.env`` is never read.
"""

from __future__ import annotations

import base64
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from medbridge import cli

GOOD_KEY = base64.urlsafe_b64encode(b"k" * 32).decode("ascii")
PG_PASSWORD = "s3cr3t-pg-password-value"
SECRET_VALUES = (GOOD_KEY, PG_PASSWORD, "token-secret-value-123", "admin-pass-value-456")


def write_env(path: Path, **overrides: str) -> Path:
    values = {
        "ENCRYPTION_MASTER_KEY": GOOD_KEY,
        "SESSION_TOKEN_SECRET": "token-secret-value-123",
        "ADMIN_PASSWORD": "admin-pass-value-456",
        "POSTGRES_PASSWORD": PG_PASSWORD,
        "DATABASE_URL": f"postgresql+asyncpg://medbridge_app:{PG_PASSWORD}@127.0.0.1:5433/medbridge",
        "HOST": "127.0.0.1",
    }
    values.update(overrides)
    path.write_text("# test env\n" + "\n".join(f"{k}={v}" for k, v in values.items()) + "\n",
                    encoding="utf-8")
    return path


def write_manifest(root: Path, present: Sequence[str], missing: Sequence[str]) -> Path:
    docs = []
    (root / "data" / "guidelines").mkdir(parents=True, exist_ok=True)
    for gid in present:
        rel = f"data/guidelines/{gid}.pdf"
        (root / rel).write_bytes(b"%PDF")
        docs.append({"guideline_id": gid, "file": rel})
    for gid in missing:
        docs.append({"guideline_id": gid, "file": f"data/guidelines/{gid}.pdf"})
    manifest = root / "data" / "corpus_manifest.json"
    manifest.write_text(json.dumps({"documents": docs}), encoding="utf-8")
    return manifest


class FakeWorld:
    """State behind the fake tools: services, model list, recorded commands."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.env_file = write_env(tmp_path / ".env")
        self.manifest = write_manifest(tmp_path, ["DOC_A", "DOC_B"], [])
        self.commands: list[list[str]] = []
        self.served: list[list[str]] = []
        self.output: list[str] = []
        self.opened: list[str] = []
        self.returncodes: dict[str, int] = {}
        self.open_ports: set[tuple[str, int]] = {("127.0.0.1", 5433)}
        self.qdrant_points: int | None = 2747  # None = collection missing
        self.qdrant_up = True
        self.ollama_up = True
        self.ollama_models = ["qwen3.5:4b", "gemma3:4b"]
        self.health = "healthy"
        self.health_after_polls = 0
        self.polls = 0
        self.clock = 0.0
        self.binaries = {"docker": "/usr/bin/docker", "ollama": "/usr/bin/ollama"}
        self.alembic_current = "h2b8d4f6a1c3 (head)"

    # -- fakes ------------------------------------------------------------
    def run(self, cmd: Sequence[str], capture: bool, timeout: float | None
            ) -> subprocess.CompletedProcess[str]:
        cmd = list(cmd)
        self.commands.append(cmd)
        joined = " ".join(cmd)
        for needle, code in self.returncodes.items():
            if needle in joined:
                return subprocess.CompletedProcess(cmd, code, "", f"{needle} failed")
        stdout = ""
        if cmd[1:3] == ["compose", "ps"]:
            stdout = "abc123\n"
        elif cmd[1:2] == ["inspect"]:
            self.polls += 1
            stdout = self.health if self.polls > self.health_after_polls else "starting"
        elif cmd[1:2] == ["info"]:
            stdout = "29.0.0"
        elif "alembic" in cmd and "current" in cmd:
            stdout = self.alembic_current
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    def serve(self, cmd: Sequence[str]) -> int:
        self.served.append(list(cmd))
        return 0

    def which(self, name: str) -> str | None:
        return self.binaries.get(name)

    def tcp_open(self, host: str, port: int) -> bool:
        return (host, port) in self.open_ports

    def http_get_json(self, url: str) -> tuple[int | None, Any]:
        if ":6333" in url:
            if not self.qdrant_up:
                return None, None
            if url.endswith("/readyz"):
                return 200, None
            if self.qdrant_points is None:
                return 404, None
            return 200, {"result": {"points_count": self.qdrant_points}}
        if ":11434" in url:
            if not self.ollama_up:
                return None, None
            return 200, {"models": [{"name": m} for m in self.ollama_models]}
        return None, None

    def sleep(self, seconds: float) -> None:
        self.clock += seconds

    def monotonic(self) -> float:
        return self.clock

    def tools(self, **overrides: Any) -> cli.Tools:
        params: dict[str, Any] = dict(
            run=self.run, serve=self.serve, which=self.which, tcp_open=self.tcp_open,
            http_get_json=self.http_get_json, sleep=self.sleep, monotonic=self.monotonic,
            out=self.output.append, open_browser=self.opened.append, python="python",
            environ={}, version_info=(3, 11, 9), in_venv=True, platform_system="Windows",
            find_spec=lambda name: object(),
            torch_probe=lambda: {"version": "2.6.0+cu124", "device": "cuda",
                                 "name": "NVIDIA GeForce RTX 4050 Laptop GPU"},
            env_file=self.env_file, manifest_file=self.manifest, repo_root=self.root,
        )
        params.update(overrides)
        return cli.Tools(**params)

    @property
    def text(self) -> str:
        return "\n".join(self.output)


DEFAULT_CONFIG = cli.AppConfig(models=("qwen3.5:4b",), source="test")


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeWorld:
    monkeypatch.setattr(cli, "load_config", lambda env_file=None, environ=None: DEFAULT_CONFIG)
    return FakeWorld(tmp_path)


def index_of(commands: list[list[str]], needle: str) -> int:
    for i, cmd in enumerate(commands):
        if needle in " ".join(cmd):
            return i
    raise AssertionError(f"{needle!r} not run; commands: {commands}")


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def test_no_command_prints_help_and_returns_usage_error(world: FakeWorld) -> None:
    assert cli.main([], tools=world.tools()) == 2


def test_unknown_option_for_doctor_is_rejected(world: FakeWorld) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["doctor", "--bogus"], tools=world.tools())
    assert exc.value.code == 2


@pytest.mark.parametrize("argv", [["--dry-run", "up"], ["up", "--dry-run"]])
def test_dry_run_flag_accepted_before_or_after_subcommand(world: FakeWorld, argv: list[str]) -> None:
    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("dry-run must not execute anything")

    tools = world.tools(run=forbidden, serve=forbidden, tcp_open=forbidden, http_get_json=forbidden)
    assert cli.main(argv, tools=tools) == 0
    text = world.text
    assert "[dry-run] docker compose up -d" in text
    assert "python -m alembic upgrade head" in text
    assert "python -m medbridge.ingestion --no-manifest-update" in text
    assert "python -m uvicorn medbridge.main:app --host 127.0.0.1 --port 8000" in text
    assert text.index("compose up -d") < text.index("alembic upgrade head") < text.index("uvicorn")


def test_abbreviated_options_are_not_swallowed(world: FakeWorld) -> None:
    assert cli.main(["ingest", "--dry"], tools=world.tools()) == 0
    assert world.commands == [["python", "-m", "medbridge.ingestion", "--dry"]]


# ---------------------------------------------------------------------------
# up
# ---------------------------------------------------------------------------


def test_up_runs_steps_in_order_and_skips_ingest_when_indexed(world: FakeWorld) -> None:
    assert cli.main(["up"], tools=world.tools()) == 0
    cmds = world.commands
    assert index_of(cmds, "compose up -d") < index_of(cmds, "alembic upgrade head")
    assert not any("medbridge.ingestion" in " ".join(c) for c in cmds)
    assert world.served == [["python", "-m", "uvicorn", "medbridge.main:app",
                             "--host", "127.0.0.1", "--port", "8000"]]
    assert "2747 chunks" in world.text
    assert "http://127.0.0.1:8000/dashboard" in world.text


@pytest.mark.parametrize("points", [None, 0])
def test_up_ingests_when_collection_missing_or_empty(world: FakeWorld, points: int | None) -> None:
    world.qdrant_points = points
    assert cli.main(["up", "--no-server"], tools=world.tools()) == 0
    cmds = world.commands
    assert index_of(cmds, "alembic upgrade head") < index_of(cmds, "medbridge.ingestion")
    assert ["python", "-m", "medbridge.ingestion", "--no-manifest-update"] in cmds
    assert world.served == []


def test_up_does_not_ingest_without_corpus_files(world: FakeWorld) -> None:
    world.qdrant_points = 0
    manifest = write_manifest(world.root / "empty", [], ["DOC_A", "DOC_B"])
    tools = world.tools(manifest_file=manifest, repo_root=world.root / "empty")
    assert cli.main(["up", "--no-server"], tools=tools) == 0
    assert not any("medbridge.ingestion" in " ".join(c) for c in world.commands)
    assert "download_corpus.py" in world.text


def test_up_no_ingest_flag(world: FakeWorld) -> None:
    world.qdrant_points = 0
    assert cli.main(["up", "--no-ingest", "--no-server"], tools=world.tools()) == 0
    assert not any("medbridge.ingestion" in " ".join(c) for c in world.commands)


def test_up_stops_when_compose_fails(world: FakeWorld) -> None:
    world.returncodes["compose up"] = 1
    assert cli.main(["up"], tools=world.tools()) == 1
    assert not any("alembic" in " ".join(c) for c in world.commands)
    assert world.served == []
    assert "docker compose up -d' failed" in world.text


def test_up_stops_when_docker_missing(world: FakeWorld) -> None:
    world.binaries.pop("docker")
    assert cli.main(["up"], tools=world.tools()) == 1
    assert "Docker Compose is not available" in world.text


def test_up_stops_when_migrations_fail(world: FakeWorld) -> None:
    world.returncodes["alembic upgrade"] = 1
    assert cli.main(["up"], tools=world.tools()) == 1
    assert world.served == []
    assert "password authentication failed" in world.text  # the hint


def test_up_waits_until_postgres_healthy(world: FakeWorld) -> None:
    world.health_after_polls = 3
    assert cli.main(["up", "--no-server"], tools=world.tools()) == 0
    assert world.polls == 4
    assert world.clock == pytest.approx(3 * cli.POLL_SECONDS)


def test_up_times_out_when_services_never_healthy(world: FakeWorld) -> None:
    world.health = "unhealthy"
    assert cli.main(["up", "--wait", "10"], tools=world.tools()) == 1
    assert "not healthy after 10 s" in world.text
    assert not any("alembic" in " ".join(c) for c in world.commands)


def test_up_skip_docker_uses_tcp_probe(world: FakeWorld) -> None:
    assert cli.main(["up", "--skip-docker", "--no-server"], tools=world.tools()) == 0
    assert not any("compose" in " ".join(c) for c in world.commands)
    assert any("alembic upgrade head" in " ".join(c) for c in world.commands)


def test_up_refuses_busy_port(world: FakeWorld) -> None:
    world.open_ports.add(("127.0.0.1", 8000))
    assert cli.main(["up"], tools=world.tools()) == 1
    assert world.served == []
    assert "--port 8001" in world.text


def test_up_custom_host_and_port(world: FakeWorld) -> None:
    assert cli.main(["up", "--port", "8100", "--host", "0.0.0.0"], tools=world.tools()) == 0
    assert world.served[0][-4:] == ["--host", "0.0.0.0", "--port", "8100"]
    assert "http://127.0.0.1:8100" in world.text  # wildcard bind shown as a browsable URL


def test_up_warns_but_starts_when_model_missing(world: FakeWorld) -> None:
    world.ollama_models = ["gemma3:4b"]
    assert cli.main(["up"], tools=world.tools()) == 0
    assert "ollama pull qwen3.5:4b" in world.text
    assert world.served


def test_up_creates_env_when_missing(world: FakeWorld) -> None:
    world.env_file.unlink()
    assert cli.main(["up", "--no-server"], tools=world.tools()) == 0
    assert world.commands[0] == ["python", str(cli.SETUP_ENV_SCRIPT)]


def test_up_open_launches_browser_when_port_answers(world: FakeWorld, monkeypatch) -> None:
    started: list[Any] = []

    class InlineThread:
        def __init__(self, target, args, daemon):
            self.target, self.args = target, args

        def start(self) -> None:
            started.append(True)
            world.open_ports.add(("127.0.0.1", 8000))  # the server came up
            self.target(*self.args)

    monkeypatch.setattr(cli.threading, "Thread", InlineThread)
    cfg = cli.AppConfig(models=("qwen3.5:4b",))
    cli._open_when_ready(world.tools(), cfg, timeout=5)  # never answers -> gives up, no browser
    assert world.opened == []
    assert world.clock >= 5
    assert cli.main(["up", "--open"], tools=world.tools()) == 0
    assert started and world.opened == ["http://127.0.0.1:8000"]


# ---------------------------------------------------------------------------
# ingest / smoke
# ---------------------------------------------------------------------------


def test_ingest_forwards_arguments(world: FakeWorld) -> None:
    assert cli.main(["ingest", "--recreate"], tools=world.tools()) == 0
    assert world.commands == [["python", "-m", "medbridge.ingestion", "--recreate"]]


def test_ingest_double_dash_forwards_ingestion_dry_run(world: FakeWorld) -> None:
    assert cli.main(["ingest", "--", "--dry-run"], tools=world.tools()) == 0
    assert world.commands == [["python", "-m", "medbridge.ingestion", "--dry-run"]]


def test_ingest_returns_child_exit_code(world: FakeWorld) -> None:
    world.returncodes["medbridge.ingestion"] = 1
    assert cli.main(["ingest", "--stats"], tools=world.tools()) == 1


def test_smoke_adds_base_url_from_settings(world: FakeWorld) -> None:
    assert cli.main(["smoke", "--timeout", "60"], tools=world.tools()) == 0
    assert world.commands == [["python", str(cli.SMOKE_SCRIPT), "--base-url",
                               "http://127.0.0.1:8000", "--timeout", "60"]]


def test_smoke_keeps_explicit_base_url(world: FakeWorld) -> None:
    assert cli.main(["smoke", "--base-url", "http://127.0.0.1:8100"], tools=world.tools()) == 0
    assert world.commands == [["python", str(cli.SMOKE_SCRIPT), "--base-url",
                               "http://127.0.0.1:8100"]]


def test_command_not_found_is_reported_not_raised(world: FakeWorld) -> None:
    def missing(cmd, capture, timeout):
        raise FileNotFoundError("no such program")

    assert cli.main(["ingest"], tools=world.tools(run=missing)) == 127


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def test_doctor_all_green(world: FakeWorld) -> None:
    assert cli.main(["doctor"], tools=world.tools()) == 0
    assert "0 problem(s)" in world.text
    assert "[FAIL]" not in world.text
    assert "RTX 4050" in world.text


def test_doctor_never_prints_secrets(world: FakeWorld) -> None:
    cli.main(["doctor"], tools=world.tools())
    cli.main(["doctor", "--json"], tools=world.tools())
    for secret in SECRET_VALUES:
        assert secret not in world.text


def test_doctor_json_output(world: FakeWorld) -> None:
    assert cli.main(["doctor", "--json"], tools=world.tools()) == 0
    payload = json.loads(world.text)
    names = {item["name"] for item in payload}
    assert {"Python", "Docker", "PostgreSQL", "Migrations", "Qdrant", "Corpus", "Ollama"} <= names


def test_doctor_dry_run_runs_nothing(world: FakeWorld) -> None:
    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("must not run")

    tools = world.tools(run=forbidden, tcp_open=forbidden, http_get_json=forbidden)
    assert cli.main(["doctor", "--dry-run"], tools=tools) == 0
    assert "Ollama server" in world.text


def test_doctor_reports_fixes_for_every_problem(world: FakeWorld) -> None:
    world.binaries.pop("docker")
    world.open_ports.clear()
    world.qdrant_points = 0
    world.ollama_models = []
    assert cli.main(["doctor"], tools=world.tools()) == 1
    text = world.text
    assert "Install Docker Desktop" in text
    assert "docker compose up -d" in text
    assert "python -m medbridge.ingestion" in text
    assert "ollama pull qwen3.5:4b" in text
    assert "Migrations" in text and "[SKIP]" in text  # skipped because PostgreSQL is down


def test_doctor_docker_engine_not_running(world: FakeWorld) -> None:
    world.returncodes["info --format"] = 1
    results = cli.run_doctor(world.tools(), cli.Launcher(world.tools()))
    docker = next(r for r in results if r.name == "Docker")
    assert docker.status == cli.FAIL
    assert "WSL 2" in docker.fix  # Windows-specific hint


def test_doctor_migrations_not_at_head(world: FakeWorld) -> None:
    world.alembic_current = "f7737fab1c53"
    results = cli.run_doctor(world.tools(), cli.Launcher(world.tools()))
    migrations = next(r for r in results if r.name == "Migrations")
    assert migrations.status == cli.WARN
    assert migrations.fix == "python -m alembic upgrade head"


def test_doctor_skips_migrations_when_packages_missing(world: FakeWorld) -> None:
    tools = world.tools(find_spec=lambda name: None if name == "alembic" else object())
    results = cli.run_doctor(tools, cli.Launcher(tools))
    migrations = next(r for r in results if r.name == "Migrations")
    assert migrations.status == cli.SKIP
    assert not any("alembic" in " ".join(c) for c in world.commands)


def test_doctor_ollama_unreachable(world: FakeWorld) -> None:
    world.ollama_up = False
    result = cli.check_ollama(world.tools(), DEFAULT_CONFIG)
    assert result.status == cli.FAIL
    assert "ollama serve" in result.fix


def test_doctor_ollama_not_installed(world: FakeWorld) -> None:
    world.ollama_up = False
    world.binaries.pop("ollama")
    result = cli.check_ollama(world.tools(), DEFAULT_CONFIG)
    assert "https://ollama.com/download" in result.fix


def test_local_only_mode_rejects_remote_ollama_host(world: FakeWorld) -> None:
    cfg = cli.AppConfig(ollama_base_url="http://192.168.1.20:11434", models=("qwen3.5:4b",))
    result = cli.check_ollama(world.tools(), cfg)
    assert result.status == cli.FAIL
    assert "LOCAL_ONLY_MODE" in result.detail


def test_corpus_check_names_manual_downloads(world: FakeWorld) -> None:
    manifest = write_manifest(world.root / "partial", ["NICE_NG136"],
                              ["AHA_ACC_HTN_2025", "ESC_HTN_2024"])
    result = cli.check_corpus(world.tools(manifest_file=manifest, repo_root=world.root / "partial"))
    assert result.status == cli.WARN
    assert "download_corpus.py" in result.fix
    assert "AHA_ACC_HTN_2025" in result.fix and "ESC_HTN_2024" in result.fix


def test_corpus_check_unreadable_manifest(world: FakeWorld) -> None:
    result = cli.check_corpus(world.tools(manifest_file=world.root / "nope.json"))
    assert result.status == cli.FAIL


def test_port_in_use_is_a_warning(world: FakeWorld) -> None:
    world.open_ports.add(("127.0.0.1", 8000))
    result = cli.check_port(world.tools(), DEFAULT_CONFIG)
    assert result.status == cli.WARN
    assert "--port 8001" in result.fix


@pytest.mark.parametrize(
    ("version", "status"),
    [((3, 10, 12), cli.FAIL), ((3, 11, 9), cli.OK), ((3, 12, 4), cli.OK), ((3, 13, 1), cli.WARN)],
)
def test_python_version_check(world: FakeWorld, version: tuple[int, int, int], status: str) -> None:
    assert cli.check_python(world.tools(version_info=version)).status == status


def test_venv_hint_is_platform_specific(world: FakeWorld) -> None:
    win = cli.check_venv(world.tools(in_venv=False, platform_system="Windows"))
    mac = cli.check_venv(world.tools(in_venv=False, platform_system="Darwin"))
    assert win.status == cli.WARN and r".\.venv\Scripts\Activate.ps1" in win.fix
    assert "source .venv/bin/activate" in mac.fix


def test_missing_packages_are_listed(world: FakeWorld) -> None:
    tools = world.tools(find_spec=lambda name: None if name in ("fastembed", "psycopg2") else object())
    result = cli.check_packages(tools)
    assert result.status == cli.FAIL
    assert "fastembed" in result.detail and "psycopg2-binary" in result.detail
    assert "requirements.txt" in result.fix


def test_cpu_torch_on_windows_with_nvidia_gpu_warns(world: FakeWorld) -> None:
    world.binaries["nvidia-smi"] = "nvidia-smi.exe"
    tools = world.tools(torch_probe=lambda: {"version": "2.6.0+cpu", "device": "cpu"})
    result = cli.check_torch(tools)
    assert result.status == cli.WARN
    assert "download.pytorch.org/whl/cu126" in result.fix


def test_torch_on_mps_is_ok(world: FakeWorld) -> None:
    tools = world.tools(platform_system="Darwin",
                        torch_probe=lambda: {"version": "2.14.1", "device": "mps"})
    assert cli.check_torch(tools).status == cli.OK


# ---------------------------------------------------------------------------
# .env inspection
# ---------------------------------------------------------------------------


def env_results(world: FakeWorld, **overrides: str) -> dict[str, cli.CheckResult]:
    write_env(world.env_file, **overrides)
    return {r.name: r for r in cli.check_env(world.tools())}


def test_env_missing_file(world: FakeWorld) -> None:
    world.env_file.unlink()
    (result,) = cli.check_env(world.tools())
    assert result.status == cli.FAIL
    assert "scripts/setup_env.py" in result.fix


def test_env_empty_secrets(world: FakeWorld) -> None:
    results = env_results(world, ENCRYPTION_MASTER_KEY="", SESSION_TOKEN_SECRET="")
    assert results[".env"].status == cli.FAIL
    assert "ENCRYPTION_MASTER_KEY" in results[".env"].detail


def test_env_password_mismatch(world: FakeWorld) -> None:
    results = env_results(
        world, DATABASE_URL="postgresql+asyncpg://medbridge_app:other@127.0.0.1:5433/medbridge")
    assert results[".env"].status == cli.FAIL
    assert "differs" in results[".env"].detail


def test_env_placeholder_password(world: FakeWorld) -> None:
    results = env_results(
        world, DATABASE_URL="postgresql+asyncpg://medbridge_app:change-me@127.0.0.1:5433/medbridge")
    assert "placeholder" in results[".env"].detail


def test_env_non_fernet_key_and_open_host_and_no_admin(world: FakeWorld) -> None:
    results = env_results(world, ENCRYPTION_MASTER_KEY="not-a-fernet-key", HOST="0.0.0.0",
                          ADMIN_PASSWORD="")
    assert results[".env"].status == cli.OK
    assert results[".env encryption key"].status == cli.WARN
    assert results[".env host"].status == cli.WARN
    assert results[".env admin"].status == cli.WARN


def test_env_environment_variables_take_precedence(world: FakeWorld) -> None:
    write_env(world.env_file, ENCRYPTION_MASTER_KEY="")
    results = cli.check_env(world.tools(environ={"ENCRYPTION_MASTER_KEY": GOOD_KEY}))
    assert results[0].status == cli.OK


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
     ("http://127.0.0.1:11434/", "http://127.0.0.1:11434"),
     ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434"),
     ("http://localhost:11435/v1/", "http://localhost:11435")],
)
def test_normalize_ollama_url(raw: str, expected: str) -> None:
    assert cli.normalize_ollama_url(raw) == expected


def test_model_present_handles_latest_tag() -> None:
    assert cli.model_present("qwen3.5:4b", ["qwen3.5:4b", "gemma3:12b"])
    assert cli.model_present("llama3", ["llama3:latest"])
    assert not cli.model_present("qwen3.5:4b", ["qwen3.5:9b"])


@pytest.mark.parametrize(("host", "loopback"),
                         [("127.0.0.1", True), ("localhost", True), ("::1", True),
                          ("[::1]", True), ("10.0.0.5", False), ("host.docker.internal", False)])
def test_is_loopback_host(host: str, loopback: bool) -> None:
    assert cli.is_loopback_host(host) is loopback


def test_read_env_file_parses_quotes_comments_and_export(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text('# c\nA=1\nexport B="two"\nC=\'three\'\nbroken line\n', encoding="utf-8")
    assert cli.read_env_file(path) == {"A": "1", "B": "two", "C": "three"}
    assert cli.read_env_file(tmp_path / "missing") == {}


def test_load_config_reads_settings_from_env_file(tmp_path: Path, monkeypatch) -> None:
    for key in cli._CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)
    env = write_env(tmp_path / ".env", LLM_MODEL="qwen3.5:4b", PORT="8100",
                    OLLAMA_BASE_URL="http://127.0.0.1:11434/v1")
    cfg = cli.load_config(env, environ={})
    assert cfg.source == "settings"
    assert cfg.models == ("qwen3.5:4b",)
    assert cfg.port == 8100
    assert cfg.ollama_base_url == "http://127.0.0.1:11434"


def test_load_config_fallback_without_pydantic(tmp_path: Path, monkeypatch) -> None:
    def broken(env_file: Path) -> cli.AppConfig:
        raise ImportError("pydantic_settings not installed")

    monkeypatch.setattr(cli, "_config_from_settings", broken)
    env = write_env(tmp_path / ".env", LLM_MODEL="gemma3:4b", LLM_MODEL_GENERATOR="qwen3.5:4b",
                    PORT="not-a-number")
    cfg = cli.load_config(env, environ={"HOST": "127.0.0.1"})
    assert cfg.source == "fallback"
    assert cfg.models == ("gemma3:4b", "qwen3.5:4b")
    assert cfg.port == 8000


def test_qdrant_points_states(world: FakeWorld) -> None:
    tools = world.tools()
    assert cli.qdrant_points(tools, DEFAULT_CONFIG) == ("ok", 2747)
    world.qdrant_points = None
    assert cli.qdrant_points(tools, DEFAULT_CONFIG) == ("missing", None)
    world.qdrant_up = False
    assert cli.qdrant_points(tools, DEFAULT_CONFIG) == ("unreachable", None)


def test_default_runner_uses_argument_list_without_shell(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"], seen["kwargs"] = cmd, kwargs
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    result = cli._default_run(["docker", "compose", "up", "-d"], True, 5.0)
    assert result.returncode == 0
    assert seen["cmd"] == ["docker", "compose", "up", "-d"]
    assert seen["kwargs"]["cwd"] == cli.REPO_ROOT
    assert not seen["kwargs"].get("shell")
    assert seen["kwargs"]["env"]["PYTHONIOENCODING"]


def test_default_serve_stops_child_on_ctrl_c(monkeypatch) -> None:
    class FakeProc:
        def __init__(self) -> None:
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            return 0

    proc = FakeProc()
    monkeypatch.setattr(cli.subprocess, "Popen", lambda cmd, cwd, env: proc)
    assert cli._default_serve(["python", "-m", "uvicorn"]) == 0
    assert proc.calls == 2


def test_http_get_json_unreachable_returns_none(monkeypatch) -> None:
    def refuse(request, timeout):
        raise cli.urllib.error.URLError("connection refused")

    monkeypatch.setattr(cli.urllib.request, "urlopen", refuse)
    assert cli.http_get_json("http://127.0.0.1:9/never", timeout=0.2) == (None, None)


def test_http_get_json_http_error_and_json_body(monkeypatch) -> None:
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self) -> bytes:
            return b'{"result": {"points_count": 5}}'

    def not_found(request, timeout):
        raise cli.urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(cli.urllib.request, "urlopen", not_found)
    assert cli.http_get_json("http://127.0.0.1:6333/collections/x") == (404, None)
    monkeypatch.setattr(cli.urllib.request, "urlopen", lambda request, timeout: Response())
    assert cli.http_get_json("http://127.0.0.1:6333/collections/x") == (
        200, {"result": {"points_count": 5}})


def test_tcp_open_handles_refused_connection(monkeypatch) -> None:
    def refuse(address, timeout):
        raise ConnectionRefusedError

    monkeypatch.setattr(cli.socket, "create_connection", refuse)
    assert cli.tcp_open("127.0.0.1", 5433) is False
