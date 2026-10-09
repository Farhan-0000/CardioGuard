"""Latency probe used for the before/after performance table.

Runs the five Review-3 scenarios against a running MedBridge API and records
end-to-end latency per action. Works with both the original API (bare session
UUIDs) and the current API (signed session tokens).

Usage:
    python scripts/latency_probe.py --label before --repeats 5 --out reports/performance/before.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

SCENARIOS: list[dict[str, Any]] = [
    {"id": "S1", "name": "Emergency fast-path", "turns": ["My BP is 210/125 and I have chest pain"]},
    {"id": "S2", "name": "Out-of-domain", "turns": ["What caused the French Revolution?"]},
    {"id": "S3", "name": "Clarification gate", "turns": ["Should I take a higher dose of my BP medicine?"]},
    {"id": "S4", "name": "Clinical guidance (2 turns)", "turns": [
        "Should I take a higher dose of my BP medicine?",
        "My BP was 145/92 this morning and I take lisinopril 10 mg once daily. I am 54.",
    ]},
    {"id": "S5", "name": "Lifestyle", "turns": ["What dietary changes help lower blood pressure?"]},
]


def _headers(session: dict[str, Any]) -> dict[str, str]:
    token = session.get("session_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def run_scenario(client: httpx.Client, scenario: dict[str, Any]) -> list[dict[str, Any]]:
    session = client.post("/api/sessions").json()
    sid = session["session_id"]
    rows = []
    for i, text in enumerate(scenario["turns"]):
        t0 = time.perf_counter()
        r = client.post(f"/api/sessions/{sid}/messages", json={"message": text}, headers=_headers(session))
        dt = time.perf_counter() - t0
        body = r.json() if r.status_code == 200 else {}
        rows.append({
            "scenario": scenario["id"], "turn": i + 1, "status": r.status_code,
            "action": body.get("action"), "latency_s": round(dt, 3),
            "trace": body.get("trace") or (body.get("explanation") or {}).get("timings_ms"),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--label", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cold-only", action="store_true")
    args = ap.parse_args()

    results: dict[str, Any] = {"label": args.label, "base_url": args.base_url, "cold": [], "warm": []}
    with httpx.Client(base_url=args.base_url, timeout=600.0) as client:
        # Cold pass: the first request of each scenario right after server start.
        for sc in SCENARIOS:
            results["cold"].extend(run_scenario(client, sc))
        if not args.cold_only:
            for _ in range(args.repeats):
                for sc in SCENARIOS:
                    results["warm"].extend(run_scenario(client, sc))

    summary: dict[str, Any] = {}
    for row in results["warm"]:
        key = f"{row['scenario']}-t{row['turn']}:{row['action']}"
        summary.setdefault(key, []).append(row["latency_s"])
    results["warm_summary"] = {
        k: {"n": len(v), "p50": round(statistics.median(v), 3),
            "p95": round(sorted(v)[max(0, int(round(0.95 * len(v))) - 1)], 3),
            "mean": round(statistics.mean(v), 3)}
        for k, v in sorted(summary.items())
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps({"cold": [(r["scenario"], r["turn"], r["action"], r["latency_s"]) for r in results["cold"]],
                      "warm_summary": results["warm_summary"]}, indent=1))


if __name__ == "__main__":
    main()
