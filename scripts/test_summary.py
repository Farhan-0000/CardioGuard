"""Run the test suite and write ``reports/test_summary.json`` for the report and README.

Runs pytest once with JUnit XML and coverage JSON output, then aggregates:

* totals (collected, passed, failed, errors, skipped, xfailed) and wall time;
* per-suite counts (unit, integration, privacy, adversarial, benchmark, evaluation,
  report, scripts) derived from the test file paths;
* line coverage overall and for the core packages (``medbridge/core``,
  ``medbridge/ai``, ``medbridge/state``, ``medbridge/privacy``, ``medbridge/retrieval``,
  ``medbridge/api``, ``evaluation``).

Usage:  python scripts/test_summary.py [--pytest-args "-m 'not live'"]
"""
from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "reports" / "tests"
SUMMARY = REPO / "reports" / "test_summary.json"
CORE_PACKAGES = [
    "medbridge/core", "medbridge/ai", "medbridge/state", "medbridge/privacy",
    "medbridge/retrieval", "medbridge/api", "medbridge/nlp", "medbridge/ingestion",
    "medbridge/features", "evaluation",
]


def suite_of(classname: str, file_attr: str) -> str:
    path = (file_attr or classname.replace(".", "/")).replace("\\", "/")
    for key, name in (
        ("medbridge/tests/unit", "unit"), ("medbridge/tests/integration", "integration"),
        ("tests/privacy", "privacy"), ("tests/adversarial", "adversarial"),
        ("tests/benchmark", "benchmark"), ("tests/evaluation", "evaluation"),
        ("tests/report", "report"), ("tests/scripts", "scripts"),
    ):
        if key in path or key.replace("/", ".") in classname:
            return name
    return "other"


def parse_junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    totals: Counter[str] = Counter()
    suites: dict[str, Counter[str]] = defaultdict(Counter)
    for case in root.iter("testcase"):
        suite = suite_of(case.get("classname", ""), case.get("file", ""))
        if case.find("failure") is not None:
            outcome = "failed"
        elif case.find("error") is not None:
            outcome = "errors"
        elif (skip := case.find("skipped")) is not None:
            outcome = "xfailed" if "xfail" in (skip.get("type", "") + skip.get("message", "")) else "skipped"
        else:
            outcome = "passed"
        totals[outcome] += 1
        totals["collected"] += 1
        suites[suite][outcome] += 1
        suites[suite]["collected"] += 1
    return {"totals": dict(totals), "suites": {k: dict(v) for k, v in sorted(suites.items())}}


def parse_coverage(path: Path) -> dict:
    data = json.loads(path.read_text())
    files = data.get("files", {})
    per_pkg: dict[str, list[int]] = {p: [0, 0] for p in CORE_PACKAGES}
    for fname, info in files.items():
        rel = fname.replace("\\", "/")
        summ = info.get("summary", {})
        for pkg in CORE_PACKAGES:
            if rel.startswith(pkg + "/") or f"/{pkg}/" in rel:
                per_pkg[pkg][0] += summ.get("covered_lines", 0)
                per_pkg[pkg][1] += summ.get("num_statements", 0)
    overall = data.get("totals", {})
    return {
        "overall_percent": round(overall.get("percent_covered", 0.0), 1),
        "covered_lines": overall.get("covered_lines"),
        "statements": overall.get("num_statements"),
        "packages": {
            pkg: {"percent": round(100 * c / s, 1) if s else None, "covered": c, "statements": s}
            for pkg, (c, s) in per_pkg.items()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pytest-args", default="", help="extra pytest arguments")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    junit, cov = OUT_DIR / "junit.xml", OUT_DIR / "coverage.json"
    cmd = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        f"--junitxml={junit}", "--cov=medbridge", "--cov=evaluation",
        f"--cov-report=json:{cov}", "--cov-report=term:skip-covered",
        *shlex.split(args.pytest_args),
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, check=False)
    elapsed = time.perf_counter() - start
    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(cmd[2:]),
        "exit_code": proc.returncode,
        "wall_time_s": round(elapsed, 1),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        **parse_junit(junit),
        "coverage": parse_coverage(cov) if cov.exists() else None,
        "note": "Default run excludes tests marked 'live'; postgres/qdrant/ollama-marked tests run "
                "only when those services are reachable.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("totals", "coverage") if summary.get(k)}, indent=1)[:2000])
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
