"""Latency probe used for the before/after performance table.

Runs the five Review-3 scenarios against a running MedBridge API and records,
per turn, the client-side end-to-end latency, the routing action and — when
the server returns it (current API) — the server-side ``explanation.timings``
per pipeline stage and ``explanation.total_ms``. Works with both the original
API (bare session UUIDs, no explanation) and the current API (signed session
tokens). Session creation is retried after ``Retry-After`` when rate limited.

Usage:
    python scripts/latency_probe.py --label before --repeats 5 --out reports/performance/before.json
    python scripts/latency_probe.py --label after --repeats 5 --out reports/performance/after.json \
        --base-url http://127.0.0.1:8100
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "S1",
        "name": "Emergency fast-path",
        "turns": ["My BP is 210/125 and I have chest pain"],
    },
    {"id": "S2", "name": "Out-of-domain", "turns": ["What caused the French Revolution?"]},
    {
        "id": "S3",
        "name": "Clarification gate",
        "turns": ["Should I take a higher dose of my BP medicine?"],
    },
    {
        "id": "S4",
        "name": "Clinical guidance (2 turns)",
        "turns": [
            "Should I take a higher dose of my BP medicine?",
            "My BP was 145/92 this morning and I take lisinopril 10 mg once daily. I am 54.",
        ],
    },
    {"id": "S5", "name": "Lifestyle", "turns": ["What dietary changes help lower blood pressure?"]},
]


def _headers(session: dict[str, Any]) -> dict[str, str]:
    token = session.get("session_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def create_session(
    client: httpx.Client, sleep: Callable[[float], None] = time.sleep, attempts: int = 8
) -> dict[str, Any]:
    """POST /api/sessions, waiting ``Retry-After`` on HTTP 429."""
    for _ in range(attempts):
        response = client.post("/api/sessions")
        if response.status_code == 429:
            sleep(float(response.headers.get("Retry-After", "6")))
            continue
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        return body
    raise RuntimeError("session creation kept being rate limited")


def server_timings(body: dict[str, Any]) -> tuple[dict[str, float] | None, float | None]:
    """Per-stage server time (ms, summed per stage, executed stages only) and total_ms."""
    explanation = body.get("explanation") or {}
    timings = explanation.get("timings")
    if not isinstance(timings, list):
        return None, None
    stages: dict[str, float] = {}
    for timing in timings:
        if timing.get("status") in ("ok", "failed"):
            name = str(timing.get("stage"))
            stages[name] = round(stages.get(name, 0.0) + float(timing.get("ms") or 0.0), 2)
    total = explanation.get("total_ms")
    return stages, float(total) if total is not None else None


def run_scenario(
    client: httpx.Client, scenario: dict[str, Any], sleep: Callable[[float], None] = time.sleep
) -> list[dict[str, Any]]:
    session = create_session(client, sleep)
    sid = session["session_id"]
    rows = []
    for i, text in enumerate(scenario["turns"]):
        t0 = time.perf_counter()
        r = client.post(
            f"/api/sessions/{sid}/messages", json={"message": text}, headers=_headers(session)
        )
        dt = time.perf_counter() - t0
        body = r.json() if r.status_code == 200 else {}
        stages, total_ms = server_timings(body)
        explanation = body.get("explanation") or {}
        rows.append(
            {
                "scenario": scenario["id"],
                "turn": i + 1,
                "status": r.status_code,
                "action": body.get("action"),
                "latency_s": round(dt, 3),
                "server_total_ms": total_ms,
                "server_stage_ms": stages,
                "timings": explanation.get("timings"),
                "llm_calls": len(explanation.get("llm_calls") or []) if explanation else None,
            }
        )
    return rows


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, round(0.95 * len(ordered)) - 1)]


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per scenario-turn-action: client latency stats and median server time per stage."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(f"{row['scenario']}-t{row['turn']}:{row['action']}", []).append(row)
    summary: dict[str, Any] = {}
    for key, items in sorted(groups.items()):
        latencies = [r["latency_s"] for r in items]
        entry: dict[str, Any] = {
            "n": len(latencies),
            "p50": round(statistics.median(latencies), 3),
            "p95": round(_p95(latencies), 3),
            "mean": round(statistics.mean(latencies), 3),
        }
        totals = [r["server_total_ms"] for r in items if r.get("server_total_ms") is not None]
        if totals:
            entry["server_total_ms_p50"] = round(statistics.median(totals), 1)
            stage_names = list(
                dict.fromkeys(s for r in items for s in (r.get("server_stage_ms") or {}))
            )
            entry["server_stage_ms_p50"] = {
                s: round(
                    statistics.median(
                        [(r.get("server_stage_ms") or {}).get(s, 0.0) for r in items]
                    ),
                    1,
                )
                for s in stage_names
            }
        summary[key] = entry
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cold-only", action="store_true")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    results: dict[str, Any] = {
        "label": args.label,
        "base_url": args.base_url,
        "cold": [],
        "warm": [],
    }
    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        # Cold pass: the first request of each scenario right after server start.
        for sc in SCENARIOS:
            results["cold"].extend(run_scenario(client, sc))
        if not args.cold_only:
            for _ in range(args.repeats):
                for sc in SCENARIOS:
                    results["warm"].extend(run_scenario(client, sc))

    results["cold_summary"] = summarise(results["cold"])
    results["warm_summary"] = summarise(results["warm"]) if results["warm"] else {}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            {
                "cold": [
                    (r["scenario"], r["turn"], r["action"], r["latency_s"], r["server_total_ms"])
                    for r in results["cold"]
                ],
                "warm_summary": results["warm_summary"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
