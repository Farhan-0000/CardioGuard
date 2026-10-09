"""
Tests for ``GET /api/metrics`` (medbridge/api/routes/metrics.py).

The aggregation is tested as a pure function; the route is tested with a fake
database session and an overridden admin dependency (plus one unauthenticated
request to prove the admin guard is wired).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from medbridge.api.middleware.error_handler import register_error_handlers
from medbridge.api.routes import metrics as metrics_module
from medbridge.api.routes.metrics import (
    MetricsResponse,
    compute_metrics,
    latency_stats,
    metrics_router,
    percentile,
)
from medbridge.db.connection import get_session
from medbridge.db.models import FinalActionEnum, Gate1ActionEnum, Gate2ActionEnum
from medbridge.privacy.deps import require_admin


def trace(**extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "gate1": {"action": "PROCEED", "source": "llm"},
        "gate2": {"action": "ANSWER", "source": "llm"},
        "timings": [
            {"stage": "extraction", "ms": 100.0, "status": "ok"},
            {"stage": "generation", "ms": 900.0, "status": "ok"},
            {"stage": "verification", "ms": 0.0, "status": "skipped"},
        ],
        "citation_validation": {"valid": 2, "stripped": 0, "downgraded": False},
        "speculative_retrieval": {
            "launched": True,
            "used": True,
            "discarded": False,
            "saved_ms": 40.0,
        },
        "redactions": 0,
    }
    base.update(extra)
    return base


ROWS = [
    (FinalActionEnum.ANSWER, Gate1ActionEnum.PROCEED, Gate2ActionEnum.ANSWER, 1000, trace()),
    (
        FinalActionEnum.GENERALIZE,
        Gate1ActionEnum.PROCEED,
        Gate2ActionEnum.ANSWER,
        2000,
        trace(
            citation_validation={"valid": 0, "stripped": 3, "downgraded": True},
            timings=[{"stage": "generation", "ms": 1500.0, "status": "failed"}],
            redactions=2,
        ),
    ),
    (
        FinalActionEnum.SOFT_ASK,
        Gate1ActionEnum.SOFT_ASK,
        None,
        500,
        trace(
            gate2=None,
            speculative_retrieval={
                "launched": True,
                "used": False,
                "discarded": True,
                "saved_ms": 0.0,
            },
            citation_validation=None,
        ),
    ),
    (
        FinalActionEnum.ESCALATE,
        None,
        Gate2ActionEnum.ESCALATE,
        5,
        {"gate2": {"action": "ESCALATE", "source": "fastpath"}, "timings": []},
    ),
    (
        FinalActionEnum.GENERALIZE,
        Gate1ActionEnum.SOFT_ASK,
        Gate2ActionEnum.GENERALIZE,
        3000,
        trace(
            gate1={"action": "PROCEED", "source": "fallback"},
            gate2={"action": "GENERALIZE", "source": "loop_breaker"},
            gate3={"n_claims": 3, "action_change": "ANSWER->GENERALIZE"},
        ),
    ),
    ("ABSTAIN", None, "ABSTAIN", None, None),  # legacy row without a trace
]


def test_percentile_interpolates():
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def test_latency_stats():
    stats = latency_stats([100.0, 200.0, 300.0])
    assert stats.count == 3 and stats.p50_ms == 200.0 and stats.mean_ms == 200.0
    assert stats.max_ms == 300.0
    assert latency_stats([]).p50_ms is None


def test_compute_metrics_aggregates_everything():
    result = compute_metrics(ROWS, [1, 1, -1, 0], window=100)
    assert result.turns == 6 and result.window == 100
    assert result.final_actions == {
        "ANSWER": 1,
        "GENERALIZE": 2,
        "SOFT-ASK": 1,
        "ESCALATE": 1,
        "ABSTAIN": 1,
    }
    assert result.gate_1 == {"PROCEED": 2, "SOFT-ASK": 2}
    assert result.gate_2 == {"ANSWER": 2, "ESCALATE": 1, "GENERALIZE": 1, "ABSTAIN": 1}
    assert result.gate_sources["gate2"] == {"llm": 2, "fastpath": 1, "loop_breaker": 1}
    assert result.latency.count == 5
    assert result.latency.p50_ms == 1000.0
    assert result.latency_by_action["GENERALIZE"].count == 2
    assert result.stage_latency["generation"].count == 4  # ok + failed, skipped excluded
    assert "verification" not in result.stage_latency
    assert result.fallbacks == {"generation": 1, "context_gate": 1}
    assert result.routing == {"loop_breaker": 1, "emergency_fastpath": 1, "small_talk": 0}
    assert result.citations == {"validated_turns": 3, "downgrades": 1, "stripped_markers": 3}
    assert result.speculative_retrieval == {
        "launched": 4,
        "used": 3,
        "discarded": 1,
        "discard_rate": 0.25,
        "mean_saved_ms": 40.0,
    }
    assert result.grounding_verifier == {"verified_turns": 1, "action_changes": 1}
    assert result.redaction == {"redacted_items": 2, "turns_with_redaction": 1}
    assert result.feedback.count == 4
    assert result.feedback.thumbs_up == 2 and result.feedback.thumbs_down == 1
    assert result.feedback.positive_rate == pytest.approx(0.667)


def test_compute_metrics_empty():
    result = compute_metrics([], [], window=10)
    assert result.turns == 0
    assert result.latency.p50_ms is None
    assert result.speculative_retrieval["discard_rate"] is None
    assert result.feedback.positive_rate is None


class FakeResult:
    def __init__(self, rows: list[Any] | None = None, scalars: list[Any] | None = None) -> None:
        self._rows = rows or []
        self._scalars = scalars or []

    def all(self) -> list[Any]:
        return self._rows

    def scalars(self) -> Any:
        return MagicMock(all=MagicMock(return_value=self._scalars))


class FakeDB:
    def __init__(self, rows: list[Any], ratings: list[int]) -> None:
        self.results = [FakeResult(rows=rows), FakeResult(scalars=ratings)]
        self.statements: list[Any] = []

    async def execute(self, statement: Any) -> FakeResult:
        self.statements.append(statement)
        return self.results.pop(0)


def make_app(db: FakeDB, admin: bool = True) -> TestClient:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(metrics_router, prefix="/api")

    async def override_session():
        yield db

    app.dependency_overrides[get_session] = override_session
    if admin:
        app.dependency_overrides[require_admin] = lambda: "admin"
    return TestClient(app, raise_server_exceptions=False)


def test_metrics_route_returns_aggregates():
    db = FakeDB([row[:5] for row in ROWS[:2]], ratings=[1, -1])
    response = make_app(db).get("/api/metrics?limit=50")
    assert response.status_code == 200
    body = MetricsResponse.model_validate(response.json())
    assert body.turns == 2 and body.window == 50
    assert body.feedback.count == 2
    assert "LIMIT" in str(db.statements[0]).upper()


def test_metrics_route_without_feedback_model(monkeypatch):
    monkeypatch.delattr(metrics_module.db_models, "Feedback", raising=False)
    db = FakeDB([row[:5] for row in ROWS[:1]], ratings=[])
    body = make_app(db).get("/api/metrics").json()
    assert body["feedback"]["count"] == 0
    assert len(db.statements) == 1


def test_metrics_requires_admin():
    response = make_app(FakeDB([], []), admin=False).get("/api/metrics")
    assert response.status_code in (401, 403)


def test_metrics_limit_validated():
    assert make_app(FakeDB([], [])).get("/api/metrics?limit=0").status_code == 422
