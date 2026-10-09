#!/usr/bin/env python
"""
Download the curated MedBridge knowledge-base documents listed in the manifest.

Reads ``data/corpus_manifest.json``, downloads every missing file from its
``download_url`` (browser User-Agent, retries with back-off), verifies the
SHA-256 recorded in the manifest and prints a manual-download list for anything
that could not be fetched automatically (publisher sites that block scripted
downloads). Works on Windows, macOS and Linux (pathlib + standard library only).

Usage (from the repository root, inside the project virtualenv):
    python scripts/download_corpus.py                 # fetch missing files, verify hashes
    python scripts/download_corpus.py --verify-only   # only check presence + hashes
    python scripts/download_corpus.py --only WHO_HTN_PHARM_2021 NICE_NG133 --force

Exit status: 0 when every document is present (hash drift on ``volatile``
sources is reported but tolerated), 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medbridge.ingestion.manifest import (  # noqa: E402
    CorpusDocument,
    ManifestError,
    load_manifest,
    sha256_file,
)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)
# Pages served instead of the document when a site demands a browser check.
_CHALLENGE_MARKERS = (
    b"Just a moment...", b"Preparing to download", b"Checking your browser",
    b"challenges.cloudflare.com", b"Attention Required!",
)


class DownloadError(Exception):
    """Raised when a document cannot be downloaded automatically."""


@dataclass
class Outcome:
    """Result for one manifest document."""

    guideline_id: str
    status: str  # ok | downloaded | drift | mismatch | manual
    detail: str = ""


def _looks_valid(content: bytes, source_format: str) -> bool:
    head = content[:4096]
    if any(marker in content[:20000] for marker in _CHALLENGE_MARKERS):
        return False
    if source_format == "pdf":
        return head.startswith(b"%PDF")
    return b"<html" in head.lower() or b"<!doctype html" in head.lower()


def fetch(url: str, *, timeout: float, retries: int) -> bytes:
    """GET ``url`` with a browser User-Agent and exponential back-off."""
    last_error = ""
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
        })
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return bytes(response.read())
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code in (401, 403, 404, 410):
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = str(exc)
        if attempt < retries:
            time.sleep(2 ** attempt)
    raise DownloadError(last_error or "download failed")


def download_document(doc: CorpusDocument, *, timeout: float, retries: int) -> None:
    """Download one document atomically to its manifest path."""
    if not doc.download_url:
        raise DownloadError("no automatic download URL (publisher blocks scripted access)")
    content = fetch(doc.download_url, timeout=timeout, retries=retries)
    if not _looks_valid(content, doc.source_format):
        raise DownloadError("server returned a browser-check page instead of the document")
    target = doc.resolve_path(REPO_ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(content)
    partial.replace(target)


def check_hash(doc: CorpusDocument) -> Outcome:
    """Compare the local file's SHA-256 with the manifest."""
    actual = sha256_file(doc.resolve_path(REPO_ROOT))
    if not doc.sha256 or actual == doc.sha256:
        return Outcome(doc.guideline_id, "ok")
    if doc.volatile:
        return Outcome(doc.guideline_id, "drift",
                       "publisher regenerates this file; ingestion will re-index it")
    return Outcome(doc.guideline_id, "mismatch", f"expected {doc.sha256[:12]}…, got {actual[:12]}…")


def process(doc: CorpusDocument, args: argparse.Namespace) -> Outcome:
    """Ensure one document is present and verified."""
    path = doc.resolve_path(REPO_ROOT)
    if path.exists() and not args.force:
        return check_hash(doc)
    if args.verify_only:
        return Outcome(doc.guideline_id, "manual", "file missing")
    try:
        download_document(doc, timeout=args.timeout, retries=args.retries)
    except DownloadError as exc:
        return Outcome(doc.guideline_id, "manual", str(exc))
    outcome = check_hash(doc)
    if outcome.status == "ok":
        outcome.status = "downloaded"
    return outcome


def build_parser() -> argparse.ArgumentParser:
    """CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "corpus_manifest.json"))
    parser.add_argument("--only", nargs="+", metavar="GUIDELINE_ID", default=None)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--verify-only", action="store_true", help="do not download anything")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit status."""
    args = build_parser().parse_args(argv)
    try:
        manifest = load_manifest(Path(args.manifest))
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    documents = manifest.documents
    if args.only:
        documents = [d for d in documents if d.guideline_id in set(args.only)]

    outcomes = []
    for doc in documents:
        outcome = process(doc, args)
        outcomes.append((doc, outcome))
        print(f"{outcome.status:10s} {doc.guideline_id:30s} {outcome.detail}")

    manual = [(d, o) for d, o in outcomes if o.status == "manual"]
    mismatched = [(d, o) for d, o in outcomes if o.status == "mismatch"]
    if manual:
        print("\nManual downloads required (save each file at the path shown, then re-run):")
        for doc, outcome in manual:
            print(f"  - {doc.guideline_id}: {doc.title}\n"
                  f"      source : {doc.download_url or doc.url}\n"
                  f"      save as: {doc.file}\n"
                  f"      reason : {outcome.detail}")
    if mismatched:
        print("\nSHA-256 mismatches (file differs from the indexed version):")
        for doc, outcome in mismatched:
            print(f"  - {doc.guideline_id}: {outcome.detail}")
    present = sum(1 for _, o in outcomes if o.status != "manual")
    print(f"\n{present}/{len(outcomes)} documents present; {len(manual)} manual; "
          f"{len(mismatched)} hash mismatches.")
    return 1 if manual or mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
