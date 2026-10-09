"""Hermetic tests for scripts/latency_probe.py server-side timing capture."""

from __future__ import annotations

from types import ModuleType

import httpx

from tests.scripts.fake_api import FakeMedBridge


def test_server_timings_sum_executed_stages_only(probe: ModuleType) -> None:
    body = {
        "explanation": {
            "timings": [
                {"stage": "retrieval", "ms": 10.0, "status": "ok"},
                {"stage": "retrieval", "ms": 5.0, "status": "discarded"},
                {"stage": "rerank", "ms": 2.5, "status": "failed"},
                {"stage": "generation", "ms": 0.0, "status": "skipped"},
            ],
            "total_ms": 42.0,
        }
    }
    stages, total = probe.server_timings(body)
    assert stages == {"retrieval": 10.0, "rerank": 2.5}
    assert total == 42.0
    assert probe.server_timings({}) == (None, None)


def test_run_scenario_records_timings_and_total(probe: ModuleType) -> None:
    api = FakeMedBridge()
    with httpx.Client(base_url="http://test", transport=api.transport()) as client:
        rows = probe.run_scenario(client, probe.SCENARIOS[3])
    assert [r["turn"] for r in rows] == [1, 2]
    for row in rows:
        assert row["status"] == 200 and row["server_total_ms"] == 1010.5
        assert row["server_stage_ms"]["extraction"] == 900.0
        assert "retrieval" not in row["server_stage_ms"]  # discarded speculative run
        assert isinstance(row["timings"], list) and row["llm_calls"] == 1


def test_legacy_api_without_explanation(probe: ModuleType) -> None:
    api = FakeMedBridge(explanation=False)
    with httpx.Client(base_url="http://test", transport=api.transport()) as client:
        rows = probe.run_scenario(client, probe.SCENARIOS[0])
    assert rows[0]["server_total_ms"] is None and rows[0]["server_stage_ms"] is None


def test_summary_has_stage_medians(probe: ModuleType) -> None:
    api = FakeMedBridge()
    with httpx.Client(base_url="http://test", transport=api.transport()) as client:
        rows = [r for _ in range(3) for r in probe.run_scenario(client, probe.SCENARIOS[4])]
    summary = probe.summarise(rows)
    (entry,) = summary.values()
    assert entry["n"] == 3 and entry["server_total_ms_p50"] == 1010.5
    assert entry["server_stage_ms_p50"]["emergency"] == 1.1


def test_session_creation_retries_after_429(probe: ModuleType) -> None:
    api = FakeMedBridge(session_429_first=1)
    waits: list[float] = []
    with httpx.Client(base_url="http://test", transport=api.transport()) as client:
        session = probe.create_session(client, sleep=waits.append)
    assert session["session_token"] and waits == [2.0]
