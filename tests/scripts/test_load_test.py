"""Hermetic tests for scripts/load_test.py (async httpx.MockTransport, no server)."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import httpx
import pytest

from tests.scripts.fake_api import FakeMedBridge


async def _run(load: ModuleType, api: FakeMedBridge, levels: list[int], rpu: int) -> tuple:
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    async with httpx.AsyncClient(base_url="http://test", transport=api.transport()) as client:
        summaries, rows = await load.run_load_test(client, levels, rpu, sleep=fake_sleep)
    return summaries, rows, waits


async def test_levels_report_latency_throughput_and_actions(load: ModuleType) -> None:
    api = FakeMedBridge()
    summaries, rows, _ = await _run(load, api, [1, 3], 4)
    assert [s.users for s in summaries] == [1, 3]
    assert [s.requests for s in summaries] == [4, 12]
    for s in summaries:
        assert s.ok == s.requests and s.errors == 0 and s.rate_limited == 0
        assert s.error_rate == 0.0 and s.throughput_rps > 0
        assert s.p50_ms is not None and s.p95_ms is not None and s.max_ms >= s.p95_ms
        assert sum(s.actions.values()) == s.ok
    assert set(summaries[1].by_kind) <= {k for k, _ in load.QUERY_MIX}
    # each simulated user has its own session
    assert len(api.sessions) == 1 + 3
    assert len(rows) == 16


async def test_rate_limited_requests_are_counted_separately(load: ModuleType) -> None:
    api = FakeMedBridge(message_429_every=3)
    summaries, _, _ = await _run(load, api, [2], 3)
    s = summaries[0]
    assert s.rate_limited == 2 and s.errors == 0 and s.ok == 4
    assert s.error_rate == pytest.approx(2 / 6 * 100, abs=0.01)


async def test_session_creation_waits_for_retry_after(load: ModuleType) -> None:
    api = FakeMedBridge(session_429_first=2)
    summaries, _, waits = await _run(load, api, [1], 1)
    assert waits == [2.0, 2.0]
    assert summaries[0].session_wait_s == 4.0


async def test_transport_errors_are_errors(load: ModuleType) -> None:
    api = FakeMedBridge()

    def flaky(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/messages"):
            raise httpx.ReadTimeout("timed out", request=request)
        return api.handle(request)

    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(flaky)) as c:
        summaries, rows = await load.run_load_test(c, [2], 2)
    assert summaries[0].errors == 4 and summaries[0].ok == 0
    assert summaries[0].p50_ms is None and all(r.error for r in rows)


def test_percentile_and_required_rate_limit(load: ModuleType) -> None:
    assert load.percentile([], 50) is None
    assert load.percentile([10.0, 20.0, 30.0, 40.0], 50) == 20.0
    assert load.percentile([float(i) for i in range(1, 101)], 95) == 95.0
    assert load.required_rate_limit([1, 5, 10], 3) == 30


def test_print_server_env(load: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    assert load.main(["--users", "1,5,10", "--requests-per-user", "4", "--print-server-env"]) == 0
    out = capsys.readouterr().out
    assert "RATE_LIMIT_PER_MINUTE=40" in out and "per client IP" in out


async def test_outputs_contain_notes(load: ModuleType, tmp_path: Path) -> None:
    summaries, rows, _ = await _run(load, FakeMedBridge(), [1, 2], 2)
    meta = {
        "generated_at": "now",
        "base_url": "http://test",
        "requests_per_user": 2,
        "required_rate_limit": 4,
    }
    json_path, md_path = load.write_outputs(summaries, rows, meta, tmp_path)
    data = json.loads(json_path.read_text())
    assert len(data["levels"]) == 2 and len(data["requests"]) == 6
    md = md_path.read_text()
    assert "serialis" in md and "per client IP" in md and "| 2 |" in md


def test_main_end_to_end_with_mock_transport(
    load: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeMedBridge()
    real = httpx.AsyncClient
    monkeypatch.setattr(
        load.httpx, "AsyncClient", lambda **kw: real(transport=api.transport(), **kw)
    )
    code = load.main(
        [
            "--base-url",
            "http://test",
            "--users",
            "1,2",
            "--requests-per-user",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert (tmp_path / "load_test.json").exists() and (tmp_path / "load_test.md").exists()


def test_invalid_levels_rejected(load: ModuleType) -> None:
    with pytest.raises(SystemExit):
        load.main(["--users", "0,2"])
