"""Concurrency load test for a running MedBridge API (asyncio + httpx).

For each concurrency level (``--users 1,5,10``) every simulated user gets its
own session and sends ``--requests-per-user`` messages sequentially; all users
of a level run concurrently. The query mix covers the deterministic paths
(emergency fast-path, small talk) and the LLM paths (out-of-domain, general,
personal with context, clarification).

Per level: p50 / p95 / max latency of successful requests, throughput
(successful requests per second of wall time), error rate, and the number of
HTTP 429 responses.

Notes printed into the report:

* A single local Ollama server serialises LLM calls, so latency grows with
  concurrency (requests queue behind each other); the deterministic paths do not.
* The API rate limiter is per client IP: ``RATE_LIMIT_PER_MINUTE`` message
  requests per minute (0 disables it) and 10 session creations per minute.
  All simulated users share one IP, so start the server under test with a
  higher limit — ``python scripts/load_test.py --print-server-env`` prints the
  value needed for the chosen levels. Session creation is retried after
  ``Retry-After`` (that wait is not counted as request latency).

Usage::

    python scripts/load_test.py --print-server-env --users 1,5,10 --requests-per-user 3
    RATE_LIMIT_PER_MINUTE=60 uvicorn medbridge.main:app --port 8100   # server under test
    python scripts/load_test.py --base-url http://127.0.0.1:8100 --users 1,5,10 \
        --requests-per-user 3

Writes ``reports/performance/load_test.json`` and ``load_test.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "performance"
SESSION_CREATE_PER_MINUTE = 10  # fixed server-side limit (medbridge.privacy.middleware)

QUERY_MIX: tuple[tuple[str, str], ...] = (
    ("emergency", "My BP is 215/125 and I have crushing chest pain spreading to my arm."),
    ("small_talk", "hello"),
    ("general", "What lifestyle changes help lower blood pressure?"),
    ("out_of_domain", "What is the best treatment for athlete's foot?"),
    (
        "personal",
        "I'm 58, my BP was 148/94 this morning and I take amlodipine 5 mg. "
        "What target should I aim for?",
    ),
    ("clarification", "Should I take a higher dose of my blood pressure tablets?"),
)
OLLAMA_NOTE = (
    "A single local Ollama server serialises LLM calls (one at a time), so concurrent requests "
    "queue and per-request latency grows with the number of users; the emergency fast-path and "
    "small-talk paths make no LLM call and are not affected."
)
RATE_LIMIT_NOTE = (
    "The API rate limiter counts per client IP. All simulated users share one IP, so the "
    "server under test needs RATE_LIMIT_PER_MINUTE at least as large as the number of message "
    "requests sent within one minute (or 0 to disable). Session creation is limited to "
    f"{SESSION_CREATE_PER_MINUTE}/min per IP; the load test waits for Retry-After and does not "
    "count that wait as latency."
)


@dataclass
class RequestResult:
    level: int
    user: int
    kind: str
    status: int
    latency_ms: float
    action: str | None = None
    error: str | None = None


@dataclass
class LevelSummary:
    users: int
    requests: int
    ok: int
    errors: int
    rate_limited: int
    error_rate: float
    wall_s: float
    throughput_rps: float
    p50_ms: float | None
    p95_ms: float | None
    max_ms: float | None
    session_wait_s: float = 0.0
    by_kind: dict[str, dict[str, Any]] = field(default_factory=dict)
    actions: dict[str, int] = field(default_factory=dict)


def required_rate_limit(levels: list[int], requests_per_user: int) -> int:
    """Message requests of the largest level (an upper bound on requests per minute)."""
    return max(levels) * requests_per_user if levels else 0


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (``None`` for no data)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return round(ordered[min(rank, len(ordered)) - 1], 1)


async def create_session(
    client: httpx.AsyncClient,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    attempts: int = 8,
) -> tuple[str, dict[str, str], float]:
    """Create a session; on 429 wait ``Retry-After``. Returns (id, headers, waited seconds)."""
    waited = 0.0
    for _ in range(attempts):
        response = await client.post("/api/sessions")
        if response.status_code == 429:
            delay = float(response.headers.get("Retry-After", "6"))
            waited += delay
            await sleep(delay)
            continue
        response.raise_for_status()
        body = response.json()
        token = body.get("session_token")
        return body["session_id"], ({"Authorization": f"Bearer {token}"} if token else {}), waited
    raise RuntimeError("session creation kept being rate limited")


async def run_user(
    client: httpx.AsyncClient,
    level: int,
    user: int,
    sid: str,
    headers: dict[str, str],
    requests: int,
    think_time: float = 0.0,
) -> list[RequestResult]:
    results: list[RequestResult] = []
    for i in range(requests):
        kind, text = QUERY_MIX[(user + i) % len(QUERY_MIX)]
        started = time.perf_counter()
        try:
            response = await client.post(
                f"/api/sessions/{sid}/messages", json={"message": text}, headers=headers
            )
            latency = (time.perf_counter() - started) * 1000.0
            action = response.json().get("action") if response.status_code == 200 else None
            results.append(
                RequestResult(level, user, kind, response.status_code, round(latency, 1), action)
            )
        except httpx.HTTPError as exc:
            latency = (time.perf_counter() - started) * 1000.0
            results.append(
                RequestResult(
                    level, user, kind, 0, round(latency, 1), error=f"{type(exc).__name__}: {exc}"
                )
            )
        if think_time:
            await asyncio.sleep(think_time)
    return results


def summarise_level(
    users: int, results: list[RequestResult], wall_s: float, session_wait_s: float = 0.0
) -> LevelSummary:
    ok = [r for r in results if r.status == 200]
    limited = sum(1 for r in results if r.status == 429)
    errors = len(results) - len(ok) - limited
    latencies = [r.latency_ms for r in ok]
    by_kind: dict[str, dict[str, Any]] = {}
    for kind in dict.fromkeys(r.kind for r in results):
        lat = [r.latency_ms for r in ok if r.kind == kind]
        by_kind[kind] = {
            "n": sum(1 for r in results if r.kind == kind),
            "ok": len(lat),
            "p50_ms": percentile(lat, 50),
            "max_ms": max(lat) if lat else None,
        }
    actions: dict[str, int] = {}
    for r in ok:
        actions[str(r.action)] = actions.get(str(r.action), 0) + 1
    return LevelSummary(
        users=users,
        requests=len(results),
        ok=len(ok),
        errors=errors,
        rate_limited=limited,
        error_rate=round((errors + limited) / len(results) * 100.0, 2) if results else 0.0,
        wall_s=round(wall_s, 2),
        throughput_rps=round(len(ok) / wall_s, 3) if wall_s > 0 else 0.0,
        p50_ms=percentile(latencies, 50),
        p95_ms=percentile(latencies, 95),
        max_ms=max(latencies) if latencies else None,
        session_wait_s=round(session_wait_s, 1),
        by_kind=by_kind,
        actions=actions,
    )


async def run_level(
    client: httpx.AsyncClient,
    users: int,
    requests_per_user: int,
    think_time: float = 0.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[LevelSummary, list[RequestResult]]:
    sessions: list[tuple[str, dict[str, str]]] = []
    waited = 0.0
    for _ in range(users):  # sequential: the session limiter is per IP
        sid, headers, wait = await create_session(client, sleep)
        sessions.append((sid, headers))
        waited += wait
    started = time.perf_counter()
    per_user = await asyncio.gather(
        *[
            run_user(client, users, u, sid, headers, requests_per_user, think_time)
            for u, (sid, headers) in enumerate(sessions)
        ]
    )
    wall = time.perf_counter() - started
    results = [r for rows in per_user for r in rows]
    return summarise_level(users, results, wall, waited), results


def render_markdown(levels: list[LevelSummary], meta: dict[str, Any]) -> str:
    lines = [
        "# MedBridge Load Test",
        "",
        f"- **Generated:** {meta['generated_at']}",
        f"- **Base URL:** `{meta['base_url']}`",
        f"- **Requests per user:** {meta['requests_per_user']}; query mix: "
        + ", ".join(k for k, _ in QUERY_MIX),
        f"- **Server rate limit needed:** RATE_LIMIT_PER_MINUTE >= {meta['required_rate_limit']} "
        "(or 0)",
        "",
        f"> {OLLAMA_NOTE}",
        "",
        f"> {RATE_LIMIT_NOTE}",
        "",
        "| Users | Requests | OK | Errors | 429 | Error rate | p50 ms | p95 ms | max ms | "
        "Throughput (req/s) | Wall s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in levels:
        lines.append(
            f"| {s.users} | {s.requests} | {s.ok} | {s.errors} | {s.rate_limited} | "
            f"{s.error_rate}% | {s.p50_ms} | {s.p95_ms} | {s.max_ms} | "
            f"{s.throughput_rps} | {s.wall_s} |"
        )
    lines += ["", "## p50 latency by query kind (ms)", ""]
    kinds = list(dict.fromkeys(k for s in levels for k in s.by_kind))
    lines += [
        "| Users | " + " | ".join(kinds) + " |",
        "| ---: | " + " | ".join("---:" for _ in kinds) + " |",
    ]
    for s in levels:
        cells = [str(s.by_kind.get(k, {}).get("p50_ms")) for k in kinds]
        lines.append(f"| {s.users} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


async def run_load_test(
    client: httpx.AsyncClient,
    levels: list[int],
    requests_per_user: int,
    think_time: float = 0.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[LevelSummary], list[RequestResult]]:
    summaries: list[LevelSummary] = []
    rows: list[RequestResult] = []
    for users in levels:
        summary, results = await run_level(client, users, requests_per_user, think_time, sleep)
        summaries.append(summary)
        rows.extend(results)
        if progress is not None:
            progress(
                f"users={users}: ok={summary.ok}/{summary.requests} 429={summary.rate_limited}"
                f" p50={summary.p50_ms} ms p95={summary.p95_ms} ms "
                f"throughput={summary.throughput_rps} req/s"
            )
    return summaries, rows


def write_outputs(
    levels: list[LevelSummary], rows: list[RequestResult], meta: dict[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "load_test.json"
    md_path = output_dir / "load_test.md"
    json_path.write_text(
        json.dumps(
            {
                **meta,
                "notes": [OLLAMA_NOTE, RATE_LIMIT_NOTE],
                "levels": [asdict(s) for s in levels],
                "requests": [asdict(r) for r in rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(levels, meta), encoding="utf-8")
    return json_path, md_path


def parse_levels(text: str) -> list[int]:
    levels = [int(x) for x in text.split(",") if x.strip()]
    if not levels or any(n <= 0 for n in levels):
        raise argparse.ArgumentTypeError("--users must be positive integers, e.g. 1,5,10")
    return levels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MedBridge concurrency load test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=parse_levels, default=[1, 5, 10])
    parser.add_argument("--requests-per-user", type=int, default=3)
    parser.add_argument("--think-time", type=float, default=0.0, help="seconds between requests")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--print-server-env",
        action="store_true",
        help="print the RATE_LIMIT_PER_MINUTE the server needs and exit",
    )
    args = parser.parse_args(argv)
    needed = required_rate_limit(args.users, args.requests_per_user)
    if args.print_server_env:
        print(f"RATE_LIMIT_PER_MINUTE={needed}")
        print(f"# {RATE_LIMIT_NOTE}")
        return 0

    async def _run() -> tuple[list[LevelSummary], list[RequestResult]]:
        async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout) as client:
            return await run_load_test(
                client, args.users, args.requests_per_user, args.think_time, progress=print
            )

    summaries, rows = asyncio.run(_run())
    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "levels": args.users,
        "requests_per_user": args.requests_per_user,
        "required_rate_limit": needed,
        "query_mix": [k for k, _ in QUERY_MIX],
        "median_ok_latency_all_ms": statistics.median(
            [r.latency_ms for r in rows if r.status == 200]
        )
        if any(r.status == 200 for r in rows)
        else None,
    }
    json_path, md_path = write_outputs(summaries, rows, meta, args.output_dir)
    print(f"Wrote {json_path} and {md_path}")
    if any(s.rate_limited for s in summaries):
        print(
            f"WARNING: 429 responses seen; restart the server with RATE_LIMIT_PER_MINUTE>={needed}"
        )
    return 0 if all(s.errors == 0 for s in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
