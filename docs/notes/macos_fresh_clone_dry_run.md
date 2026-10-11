# macOS fresh-clone dry run (11 October 2026)

A fresh `git clone` of the repository was set up by following `HOW_TO_RUN.md` Part B
(sections 5.3 and 5.8 to 5.11) and section 9.2, to check that the instructions work from
scratch. The clone lived in a folder with another name (`medbridge-freshclone`), so Docker
Compose created separate containers and volumes; the development containers were stopped
during the run and restarted afterwards with their data intact.

## Machine

| Item | Value |
|---|---|
| Hardware | Apple M5 (Mac17,2), 24 GB |
| OS | macOS 26.7.1 |
| Python | 3.12.13, found by `scripts/setup.sh` (`python3.12` on PATH; the development venv uses 3.11) |
| Docker | Engine 29.6.2 |
| Ollama | 0.35.0, `qwen3.5:4b` |
| Commit | `8301353` (setup at `5d725d8` after the first finding below) |

## Steps and results

| Step | Command | Result | Time |
|---|---|---|---|
| 5.3 Automatic setup | `scripts/setup.sh --pull-models` | venv with Python 3.12, all packages, PyTorch 2.14.1 on MPS, `.env` generated, model pulled; doctor flagged the corpus as missing (expected at this point) | 1 min 55 s |
| 5.8 Services | `docker compose up -d`, `curl .../readyz` | both services healthy, Qdrant "all shards are ready" | 11 s |
| 5.9 Schema | `python -m alembic upgrade head` | at head (`h2b8d4f6a1c3`) | a few seconds |
| 5.10 Corpus download | `python scripts/download_corpus.py` | 20 of 27 documents downloaded; 7 listed as manual (the 2 publisher PDFs and 5 PMC articles that returned a browser-check page) | 43 s |
| 5.10 Manual files | copied from the development checkout, standing in for the browser downloads | `--verify-only`: 27/27 present, 0 hash mismatches (volatile files show `drift`) | |
| 5.10 Indexing | `python -m medbridge.ingestion --recreate --no-manifest-update` | 27 documents, 2,748 chunks (development index: 2,747; volatile documents differ), 0 failed | 2 min 38 s |
| 5.11 API | `python -m uvicorn medbridge.main:app --host 127.0.0.1 --port 8000` | `/health` healthy after 37 s, every `models_loaded` entry true, Gate 3 (`nli_verifier`) loaded | 37 s |
| Doctor | `python -m medbridge.cli doctor` | 11 OK, 1 warning (port 8000 in use by the running API), 0 problems | |
| 9.2 Smoke test | `python -m medbridge.cli smoke --output ...` | 16/16 PASS (`reports/smoke_test_fresh_clone_macos.json`) | 1 min 44 s |
| 5.13 Launcher | `scripts/run.sh --dry-run` | prints the expected sequence (Docker, health wait, migrations, ingestion if empty, API) | |
| 8 Tests | `python -m pytest -q` | 3,474 passed, 1 skipped (after the fix below) | 52 s |

The `.env` written by `setup_env.py` matched the final configuration: `LOCAL_ONLY_MODE=true`,
`LLM_MODEL=qwen3.5:4b`, `GROUNDING_VERIFIER_ENABLED=true`, θ = 0.2, τ = 1.0.

## Findings and fixes

1. **Scripts not executable after cloning.** `scripts/setup.sh` failed with
   `permission denied`: the repository has `core.fileMode=false` (it started on Windows), so
   git stored every `scripts/*.sh` as 100644. Fixed in commit `5d725d8` (stored as 100755),
   and HOW_TO_RUN 5.2 now gives the `chmod +x scripts/*.sh` fallback for copies that lose the
   bit.
2. **PMC browser checks.** HOW_TO_RUN said 25 documents download automatically. On this run
   PubMed Central returned a browser-check page for 5 articles. The script detected it, saved
   nothing and listed each one with its URL and file name. HOW_TO_RUN 4.11 now says this can
   happen and explains how to save the pages in a browser. The script does not try to get
   around the check.
3. **Corpus tests assumed the development files.** Two tests failed on the fresh download:
   `test_files_match_hashes_when_downloaded` and `test_every_manifest_document_is_indexed`
   compared every file and chunk count with the committed manifest, but 10 volatile documents
   (the NICE NG136 PDF and nine web pages) came back in a newer version. The tests now
   require exact hashes and counts for the fixed documents and for volatile documents that
   have not drifted, and only presence for a drifted volatile document. On the development
   machine they remain exact.
4. **macOS indexing command.** Section 5.10 ran ingestion without `--no-manifest-update`,
   which would rewrite the committed manifest; it now matches the Windows section.
5. **Doctor with a foreign database volume (expected).** Before the development containers
   were stopped, the doctor in the clone reached the development PostgreSQL and reported a
   failed migration check, because the clone's new `POSTGRES_PASSWORD` differs. Its fix hint
   (restore the old password or reset with `docker compose down -v`) is correct; nothing to
   change.

## Not covered by this run

- **Windows.** Part A (RTX 4050 laptop) has not been run end to end; use the checklist in
  HOW_TO_RUN 4.15 and record the outcome there.
- **First-time downloads.** The Hugging Face cache (reranker, Gate 3 model, fastembed models),
  the Ollama model and the Docker images were already on this Mac, and pip reused cached
  wheels for part of the install (94 cached, 79 downloaded). A machine without them needs
  internet access and more time on the first run (HOW_TO_RUN 3.3 lists the sizes).
- **Manual downloads.** The two publisher PDFs and the five PMC pages were copied from the
  development checkout, not saved through a browser.
