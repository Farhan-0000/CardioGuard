"""Hermetic tests for scripts/smoke_test.py (httpx.MockTransport, no server)."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from tests.scripts.fake_api import FakeMedBridge


def _run(smoke: ModuleType, api: FakeMedBridge) -> list[Any]:
    sleeps: list[float] = []
    with httpx.Client(base_url="http://test", transport=api.transport()) as client:
        checks = smoke.SmokeRunner(client, sleep=sleeps.append).run()
    api.sleeps = sleeps  # type: ignore[attr-defined]
    return checks


def _by_name(checks: list[Any]) -> dict[str, Any]:
    return {c.name: c for c in checks}


def test_healthy_stack_passes_every_check(smoke: ModuleType) -> None:
    checks = _by_name(_run(smoke, FakeMedBridge()))
    expected = {
        "health",
        "create_session",
        "escalate_fastpath",
        "abstain_out_of_domain",
        "generalize",
        "pii_redaction",
        "sse_stream",
        "feedback",
        "context",
        "summary",
        "auth_missing_token",
        "soft_ask",
        "follow_up_answer",
        "delete_session",
        "after_delete_gone",
    }
    assert expected <= set(checks)
    failing = {n: (c.observed, c.note) for n, c in checks.items() if c.status != "PASS"}
    assert failing == {}
    assert checks["escalate_fastpath"].observed == "ESCALATE"
    assert "redaction_count=2" in checks["pii_redaction"].observed
    assert "done" in checks["sse_stream"].observed


def test_wrong_action_fails_and_acceptable_variation_passes_with_note(smoke: ModuleType) -> None:
    api = FakeMedBridge()
    api.actions["chest pain"] = "ANSWER"  # emergency not escalated → must fail
    api.actions["French Revolution"] = "GENERALIZE"  # acceptable, but not the primary
    checks = _by_name(_run(smoke, api))
    assert checks["escalate_fastpath"].status == "FAIL"
    assert checks["abstain_out_of_domain"].status == "PASS"
    assert "primary expectation ABSTAIN" in checks["abstain_out_of_domain"].note


def test_missing_redaction_and_broken_stream_fail(smoke: ModuleType) -> None:
    checks = _by_name(_run(smoke, FakeMedBridge(redact=False, stream_ok=False)))
    assert checks["pii_redaction"].status == "FAIL"
    assert checks["sse_stream"].status == "FAIL"


def test_session_rate_limit_is_retried(smoke: ModuleType) -> None:
    api = FakeMedBridge(session_429_first=1)
    checks = _by_name(_run(smoke, api))
    assert checks["create_session"].status == "PASS"
    assert api.sleeps == [2.0]  # type: ignore[attr-defined]


def test_unreachable_server_reports_failures_without_crashing(smoke: ModuleType) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(refuse)) as client:
        checks = smoke.SmokeRunner(client, sleep=lambda s: None).run()
    assert [c.status for c in checks] == ["FAIL", "FAIL"]
    assert "ConnectError" in checks[0].observed


def test_main_writes_report_and_sets_exit_code(
    smoke: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_client = httpx.Client
    for api, expected_code in ((FakeMedBridge(), 0), (FakeMedBridge(health_status="unhealthy"), 1)):
        monkeypatch.setattr(
            smoke.httpx,
            "Client",
            lambda api=api, **kw: real_client(transport=api.transport(), **kw),
        )
        out = tmp_path / f"smoke_{expected_code}.json"
        code = smoke.main(["--base-url", "http://test", "--output", str(out)])
        assert code == expected_code
        report = json.loads(out.read_text())
        assert report["passed"] is (expected_code == 0)
        assert report["counts"]["PASS"] >= 10
    assert "CHECK" in capsys.readouterr().out  # PASS/FAIL table printed
