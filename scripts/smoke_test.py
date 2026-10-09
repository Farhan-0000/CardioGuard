"""End-to-end smoke test against a running MedBridge stack.

Checks, in order: ``/health``; session creation (signed token); one query per
routing action — ESCALATE (deterministic fast-path), ABSTAIN (out-of-domain),
SOFT-ASK followed by a context-completing follow-up that should be answered,
GENERALIZE (population-level question); PII redaction (``redaction_count > 0``);
the SSE stream endpoint; feedback; the context card; the clinician summary;
the missing-token guard; and session erasure (DELETE → 204, then 410).

LLM-routed actions may legitimately vary between models and runs, so each case
has an *acceptable action set*; the observed action is always reported, with a
note when it differs from the primary expectation.

Usage::

    python scripts/smoke_test.py                         # http://127.0.0.1:8000
    python scripts/smoke_test.py --base-url http://127.0.0.1:8100 --timeout 300

Writes ``reports/smoke_test_latest.json``; the exit code is 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "smoke_test_latest.json"


@dataclass
class Check:
    name: str
    status: str  # PASS | FAIL | SKIP
    expected: str = ""
    observed: str = ""
    latency_ms: float = 0.0
    note: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MessageCase:
    name: str
    message: str
    primary: str
    acceptable: frozenset[str]


ESCALATE_CASE = MessageCase(
    "escalate_fastpath",
    "My BP is 210/125 and I have crushing chest pain right now.",
    "ESCALATE",
    frozenset({"ESCALATE"}),
)
ABSTAIN_CASE = MessageCase(
    "abstain_out_of_domain",
    "What caused the French Revolution?",
    "ABSTAIN",
    frozenset({"ABSTAIN", "GENERALIZE"}),
)
SOFT_ASK_CASE = MessageCase(
    "soft_ask",
    "Should I take a higher dose of my BP medicine?",
    "SOFT-ASK",
    frozenset({"SOFT-ASK", "GENERALIZE"}),
)
FOLLOW_UP_CASE = MessageCase(
    "follow_up_answer",
    "My BP was 145/92 this morning and I take lisinopril 10 mg once daily. I am 54.",
    "ANSWER",
    frozenset({"ANSWER", "GENERALIZE"}),
)
GENERALIZE_CASE = MessageCase(
    "generalize",
    "What dietary changes help lower blood pressure in general?",
    "GENERALIZE",
    frozenset({"GENERALIZE", "ANSWER"}),
)
PII_MESSAGE = (
    "My name is John Smith and my phone number is +91 98765 43210. "
    "Is 128/82 a normal blood pressure?"
)
STREAM_MESSAGE = "What is a normal blood pressure for an adult?"


class SmokeRunner:
    """Runs the checks with an ``httpx.Client`` (injectable for tests)."""

    def __init__(self, client: httpx.Client, sleep: Callable[[float], None] = time.sleep) -> None:
        self.client = client
        self.sleep = sleep
        self.checks: list[Check] = []

    # -- helpers -------------------------------------------------------------

    def _record(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    def _timed(
        self, method: str, url: str, **kwargs: Any
    ) -> tuple[httpx.Response | None, float, str]:
        started = time.perf_counter()
        try:
            response = self.client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            return None, (time.perf_counter() - started) * 1000.0, f"{type(exc).__name__}: {exc}"
        return response, (time.perf_counter() - started) * 1000.0, ""

    def create_session(self, attempts: int = 3) -> tuple[str, dict[str, str]] | None:
        for _ in range(attempts):
            response, ms, error = self._timed("POST", "/api/sessions")
            if response is not None and response.status_code == 429:
                self.sleep(float(response.headers.get("Retry-After", "5")))
                continue
            if response is None or response.status_code != 201:
                self._record(
                    Check(
                        "create_session",
                        "FAIL",
                        "201 + session_token",
                        error or str(response.status_code if response else ""),
                        ms,
                    )
                )
                return None
            body = response.json()
            token = body.get("session_token")
            ok = bool(body.get("session_id")) and bool(token)
            self._record(
                Check(
                    "create_session",
                    "PASS" if ok else "FAIL",
                    "201 + session_token",
                    "201" + (" + token" if token else " (no token)"),
                    ms,
                )
            )
            return body["session_id"], {"Authorization": f"Bearer {token}"} if token else {}
        self._record(Check("create_session", "FAIL", "201", "429 rate limited", 0.0))
        return None

    def send(self, sid: str, headers: dict[str, str], case: MessageCase) -> dict[str, Any] | None:
        response, ms, error = self._timed(
            "POST",
            f"/api/sessions/{sid}/messages",
            json={"message": case.message},
            headers=headers,
        )
        expected = "/".join(sorted(case.acceptable))
        if response is None or response.status_code != 200:
            observed = error or f"HTTP {response.status_code if response else '?'}"
            self._record(Check(case.name, "FAIL", expected, observed, ms))
            return None
        body: dict[str, Any] = response.json()
        action = str(body.get("action"))
        note = "" if action == case.primary else f"primary expectation {case.primary}"
        explanation = body.get("explanation") or {}
        self._record(
            Check(
                case.name,
                "PASS" if action in case.acceptable else "FAIL",
                expected,
                action,
                ms,
                note,
                {
                    "citations": len(body.get("citations") or []),
                    "soft_ask_count": body.get("soft_ask_count"),
                    "server_total_ms": explanation.get("total_ms"),
                },
            )
        )
        return body

    # -- checks ----------------------------------------------------------------

    def check_health(self) -> None:
        response, ms, error = self._timed("GET", "/health")
        if response is None:
            self._record(Check("health", "FAIL", "200 healthy|degraded", error, ms))
            return
        body = (
            response.json()
            if response.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        status = str(body.get("status", ""))
        ok = response.status_code == 200 and status in ("healthy", "degraded")
        self._record(
            Check(
                "health",
                "PASS" if ok else "FAIL",
                "200 healthy|degraded",
                f"{response.status_code} {status}",
                ms,
                "degraded: see details" if status == "degraded" else "",
                {
                    k: body.get(k)
                    for k in (
                        "postgres_connected",
                        "qdrant_connected",
                        "ollama_reachable",
                        "qdrant_points",
                        "models_loaded",
                    )
                },
            )
        )

    def check_stream(self, sid: str, headers: dict[str, str]) -> None:
        started = time.perf_counter()
        events: list[str] = []
        done: dict[str, Any] = {}
        try:
            with self.client.stream(
                "POST",
                f"/api/sessions/{sid}/messages/stream",
                json={"message": STREAM_MESSAGE},
                headers=headers,
            ) as resp:
                if resp.status_code != 200:
                    self._record(
                        Check(
                            "sse_stream",
                            "FAIL",
                            "200 text/event-stream",
                            f"HTTP {resp.status_code}",
                            (time.perf_counter() - started) * 1000.0,
                        )
                    )
                    return
                event = ""
                for line in resp.iter_lines():
                    if line.startswith("event:"):
                        event = line.split(":", 1)[1].strip()
                        events.append(event)
                    elif line.startswith("data:") and event == "done":
                        done = json.loads(line.split(":", 1)[1])
        except httpx.HTTPError as exc:
            self._record(
                Check(
                    "sse_stream",
                    "FAIL",
                    "stage..done events",
                    f"{exc}",
                    (time.perf_counter() - started) * 1000.0,
                )
            )
            return
        ms = (time.perf_counter() - started) * 1000.0
        ok = (
            "stage" in events
            and "done" in events
            and "error" not in events
            and bool(done.get("action"))
        )
        counts = {e: events.count(e) for e in dict.fromkeys(events)}
        self._record(
            Check(
                "sse_stream",
                "PASS" if ok else "FAIL",
                "stage..token..done",
                f"events {counts}; action {done.get('action')}",
                ms,
            )
        )

    def check_get(
        self,
        name: str,
        url: str,
        headers: dict[str, str],
        expected: int,
        validate: Callable[[httpx.Response], bool] | None = None,
    ) -> httpx.Response | None:
        response, ms, error = self._timed("GET", url, headers=headers)
        if response is None:
            self._record(Check(name, "FAIL", str(expected), error, ms))
            return None
        ok = response.status_code == expected and (validate is None or validate(response))
        self._record(
            Check(name, "PASS" if ok else "FAIL", str(expected), str(response.status_code), ms)
        )
        return response

    def run(self) -> list[Check]:
        self.check_health()
        created = self.create_session()
        if created is None:
            return self.checks
        sid, headers = created

        self.send(sid, headers, ESCALATE_CASE)
        self.send(sid, headers, ABSTAIN_CASE)
        self.send(sid, headers, GENERALIZE_CASE)

        response, ms, error = self._timed(
            "POST", f"/api/sessions/{sid}/messages", json={"message": PII_MESSAGE}, headers=headers
        )
        if response is not None and response.status_code == 200:
            body = response.json()
            count = int(body.get("redaction_count") or 0)
            pii_reply: dict[str, Any] | None = body
            self._record(
                Check(
                    "pii_redaction",
                    "PASS" if count > 0 else "FAIL",
                    "redaction_count > 0",
                    f"redaction_count={count}",
                    ms,
                    details={"action": body.get("action")},
                )
            )
        else:
            pii_reply = None
            self._record(
                Check(
                    "pii_redaction",
                    "FAIL",
                    "redaction_count > 0",
                    error or f"HTTP {response.status_code if response else '?'}",
                    ms,
                )
            )

        self.check_stream(sid, headers)

        message_id = (pii_reply or {}).get("message_id")
        if message_id:
            fb, ms, error = self._timed(
                "POST",
                f"/api/sessions/{sid}/feedback",
                json={"message_id": message_id, "rating": 1, "comment": "smoke test"},
                headers=headers,
            )
            self._record(
                Check(
                    "feedback",
                    "PASS" if fb is not None and fb.status_code == 201 else "FAIL",
                    "201",
                    error or str(fb.status_code if fb else ""),
                    ms,
                )
            )
        else:
            self._record(Check("feedback", "SKIP", "201", "no message_id to rate"))

        self.check_get(
            "context",
            f"/api/sessions/{sid}/context",
            headers,
            200,
            lambda r: "snapshot" in r.json(),
        )
        self.check_get("summary", f"/api/sessions/{sid}/summary?format=json", headers, 200)
        self.check_get("auth_missing_token", f"/api/sessions/{sid}/history", {}, 401)

        second = self.create_session()
        if second is not None:
            sid2, headers2 = second
            first = self.send(sid2, headers2, SOFT_ASK_CASE)
            if first is not None:
                self.send(sid2, headers2, FOLLOW_UP_CASE)

        response, ms, error = self._timed("DELETE", f"/api/sessions/{sid}", headers=headers)
        self._record(
            Check(
                "delete_session",
                "PASS" if response is not None and response.status_code == 204 else "FAIL",
                "204",
                error or str(response.status_code if response else ""),
                ms,
            )
        )
        self.check_get("after_delete_gone", f"/api/sessions/{sid}/history", headers, 410)
        return self.checks


def render_table(checks: list[Check]) -> str:
    rows = [("CHECK", "STATUS", "EXPECTED", "OBSERVED", "MS", "NOTE")]
    rows += [
        (c.name, c.status, c.expected, c.observed, f"{c.latency_ms:.0f}", c.note) for c in checks
    ]
    widths = [min(max(len(str(r[i])) for r in rows), 48) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(str(cell)[: widths[i]].ljust(widths[i]) for i, cell in enumerate(row))
        for row in rows
    )


def write_report(checks: list[Check], base_url: str, output: Path) -> dict[str, Any]:
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "passed": all(c.status != "FAIL" for c in checks),
        "counts": {s: sum(1 for c in checks if c.status == s) for s in ("PASS", "FAIL", "SKIP")},
        "checks": [asdict(c) for c in checks],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MedBridge end-to-end smoke test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=300.0, help="per-request timeout (s)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        checks = SmokeRunner(client).run()
    report = write_report(checks, args.base_url, args.output)
    print(render_table(checks))
    print(f"\n{report['counts']} -> {'PASS' if report['passed'] else 'FAIL'}; report {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
