"""
JSON / JSONL helpers shared by the runner, judges and analysis.

Run records are appended one line per turn. A crash in the middle of a
multi-turn item leaves a *partial attempt*; every record therefore carries an
``attempt_id`` and readers keep, per ``(condition, item_id)``, only the latest
attempt that contains every turn. Judgment files are keyed by
``(condition, item_id, turn_index)`` and the latest line wins.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

TurnKey = tuple[str, int]  # (item_id, turn_index)


def sanitize_json(value: Any) -> Any:
    """Recursively replace NaN/inf floats with ``None`` so the output is strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [sanitize_json(v) for v in value]
    return value


def write_json(path: Path, data: Any) -> None:
    """Write ``data`` as pretty, strict JSON (NaN/inf → null), creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_json(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one record as a single JSON line (flushed immediately)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(sanitize_json(record), ensure_ascii=False, allow_nan=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL file, skipping blank or truncated lines (crash-safe)."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[evaluation.io] skipping malformed line {lineno} in {path}", file=sys.stderr
                )
                continue
            if isinstance(record, dict):
                yield record


def latest_complete_attempts(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep, per item, the most recent attempt that recorded every turn.

    Records must carry ``item_id``, ``attempt_id``, ``turn_index`` and ``n_turns``.
    The result is sorted by ``(item_id, turn_index)``.
    """
    attempts: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    order: dict[tuple[str, str], int] = {}
    for position, rec in enumerate(records):
        key = (str(rec["item_id"]), str(rec.get("attempt_id", "")))
        attempts.setdefault(key, {})[int(rec["turn_index"])] = rec
        order[key] = position
    best: dict[str, tuple[int, dict[int, dict[str, Any]]]] = {}
    for (item_id, attempt_id), turns in attempts.items():
        n_turns = int(next(iter(turns.values())).get("n_turns", len(turns)))
        if set(turns) != set(range(n_turns)):
            continue
        position = order[(item_id, attempt_id)]
        if item_id not in best or position > best[item_id][0]:
            best[item_id] = (position, turns)
    out: list[dict[str, Any]] = []
    for item_id in sorted(best):
        turns = best[item_id][1]
        out.extend(turns[i] for i in sorted(turns))
    return out


class OutputExistsError(RuntimeError):
    """An output file already holds results and neither resume nor fresh was requested."""


def guard_output(path: Path, *, resume: bool, fresh: bool, tag: str) -> Path | None:
    """Protect existing results before a (re)run.

    * empty / missing file, or ``resume`` → nothing to do;
    * ``fresh`` → the file is renamed to ``<name>.<tag>.bak`` (returned);
    * otherwise → ``OutputExistsError``.
    """
    if not path.exists() or path.stat().st_size == 0 or resume:
        return None
    if not fresh:
        raise OutputExistsError(
            f"{path} already has results; pass --resume to continue it or --fresh to start over "
            "(the old file is kept as a backup)"
        )
    backup = path.with_name(f"{path.name}.{tag}.bak")
    path.rename(backup)
    return backup


def run_file(runs_dir: Path, condition: str) -> Path:
    """Path of the per-condition run JSONL file."""
    return runs_dir / f"{condition}.jsonl"


def load_run_records(
    runs_dir: Path, conditions: Iterable[str] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Load the latest complete attempt of every item for each condition found in ``runs_dir``."""
    wanted = None if conditions is None else {c.upper() for c in conditions}
    out: dict[str, list[dict[str, Any]]] = {}
    if not runs_dir.exists():
        return out
    for path in sorted(runs_dir.glob("*.jsonl")):
        condition = path.stem
        if wanted is not None and condition.upper() not in wanted:
            continue
        out[condition] = latest_complete_attempts(iter_jsonl(path))
    return out


def judgment_file(judgments_dir: Path, judge: str, condition: str) -> Path:
    """Path of the per-judge, per-condition judgment JSONL file."""
    return judgments_dir / judge / f"{condition}.jsonl"


def load_judgments(path: Path) -> dict[TurnKey, dict[str, Any]]:
    """Load a judgment file keyed by ``(item_id, turn_index)``; later lines win."""
    out: dict[TurnKey, dict[str, Any]] = {}
    for rec in iter_jsonl(path):
        out[(str(rec["item_id"]), int(rec["turn_index"]))] = rec
    return out


def load_all_judgments(
    judgments_dir: Path, judge: str, conditions: Iterable[str] | None = None
) -> dict[str, dict[TurnKey, dict[str, Any]]]:
    """Load every condition's judgments for one judge (``llm`` or ``nli``)."""
    base = judgments_dir / judge
    wanted = None if conditions is None else {c.upper() for c in conditions}
    out: dict[str, dict[TurnKey, dict[str, Any]]] = {}
    if not base.exists():
        return out
    for path in sorted(base.glob("*.jsonl")):
        if wanted is not None and path.stem.upper() not in wanted:
            continue
        out[path.stem] = load_judgments(path)
    return out


def write_table(
    rows: list[dict[str, Any]], tables_dir: Path, name: str, title: str = ""
) -> list[Path]:
    """Write ``rows`` as ``<name>.csv`` and a GitHub-markdown ``<name>.md``."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    md_path, csv_path = tables_dir / f"{name}.md", tables_dir / f"{name}.csv"
    headers = list(rows[0]) if rows else ["(no data)"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"### {title}", ""] if title else []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |" for r in rows)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [md_path, csv_path]
