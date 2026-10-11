# How to run MedBridge

This guide takes you from an empty laptop to a running **MedBridge** system (presented as
*CardioGuard*), and then through the tests, the benchmarks, the hallucination study, the
report build and the live demo. It is written for someone who has never set the project up
before.

There are two complete, separate walkthroughs:

- **[Part A: Windows 11](#4-part-a-windows-11-nvidia-rtx-4050-laptop-6-gb)**, for the
  demo laptop (NVIDIA RTX 4050 Laptop GPU with 6 GB VRAM, Python 3.11, Docker Desktop,
  Ollama for Windows, PowerShell).
- **[Part B: macOS](#5-part-b-macos-apple-silicon)**, for the Apple Silicon development
  machine.

Everything runs on your own computer. The language model runs locally in Ollama, the
database and the vector store run locally in Docker, and `LOCAL_ONLY_MODE=true` (the
default) blocks every remote AI service. No paid API key is needed.

> **Conventions used in this guide**
> - Run every command from the **repository root**, the folder that contains
>   `HOW_TO_RUN.md`, `requirements.txt` and `docker-compose.yml`. All paths are relative
>   to that folder.
> - `powershell` blocks are for Windows (PowerShell 5.1 or 7). `bash` blocks are for the
>   macOS Terminal (zsh or bash both work).
> - `(.venv)` at the start of your prompt means the virtual environment is active. Most
>   commands need it.
> - Text in `[TO BE FILLED: ...]` is for the team to complete.

---

## Contents

1. [What runs where](#1-what-runs-where)
2. [Quick start (if you have done this before)](#2-quick-start-if-you-have-done-this-before)
3. [Requirements](#3-requirements)
4. [Part A: Windows 11 (NVIDIA RTX 4050 Laptop, 6 GB)](#4-part-a-windows-11-nvidia-rtx-4050-laptop-6-gb)
5. [Part B: macOS (Apple Silicon)](#5-part-b-macos-apple-silicon)
6. [Configuration reference (`.env`)](#6-configuration-reference-env)
7. [The launcher: `python -m medbridge.cli`](#7-the-launcher-python--m-medbridgecli)
8. [Running the tests](#8-running-the-tests)
9. [Live benchmarks and measurements](#9-live-benchmarks-and-measurements)
10. [Reproducing the hallucination study](#10-reproducing-the-hallucination-study)
11. [Building the report, paper and slides](#11-building-the-report-paper-and-slides)
12. [Demo script for the review](#12-demo-script-for-the-review)
13. [Troubleshooting](#13-troubleshooting)
14. [Stopping, resetting and uninstalling](#14-stopping-resetting-and-uninstalling)
15. [Command cheat sheet](#15-command-cheat-sheet)

---

## 1. What runs where

| Component | What it does | Address | Started by |
|---|---|---|---|
| MedBridge API + web UI | FastAPI app: the 8-stage pipeline, the chat UI at `/` and the admin dashboard at `/dashboard` | `http://127.0.0.1:8000` | `uvicorn` (or the launcher) |
| PostgreSQL 16 | Sessions, messages, event-sourced clinical state, append-only audit log (encrypted fields) | `127.0.0.1:5433` | `docker compose up -d` |
| Qdrant | Vector store with the indexed guideline chunks (dense + BM25 sparse vectors) | `127.0.0.1:6333` (REST), `6334` (gRPC) | `docker compose up -d` |
| Ollama | Runs the local LLM used by the Context Extractor, both gates and the generator | `127.0.0.1:11434` | the Ollama app |
| Ollama (study server, optional) | Second Ollama server with parallel slots for the hallucination study | `127.0.0.1:11435` | `scripts/study_ollama.sh` |
| Streamlit (legacy UI, optional) | The original Review-3 front end | `http://localhost:8501` | `streamlit run` |

Every port is bound to the loopback address only (`127.0.0.1`), so nothing is reachable
from other computers on your network.

A chat message travels like this: privacy redaction, then the deterministic emergency
fast-path (no LLM), then context extraction and state projection, then **Gate 1 (Context
Gate)**, then hybrid retrieval and cross-encoder reranking, then **Gate 2 (Evidence Gate)**,
then grounded generation (or a fixed template for ABSTAIN and ESCALATE), then **Gate 3**
(a local NLI model checks every claim of the draft against its cited passages; on by
default), then citation validation, and finally one atomic database write with the audit
record. The five possible actions are ANSWER, SOFT-ASK, GENERALIZE, ABSTAIN and ESCALATE.

Besides the LLM in Ollama, the API process runs three small models itself: the fastembed
embedders (CPU), the cross-encoder reranker and the Gate 3 NLI model (PyTorch on CUDA, MPS or
CPU). They are downloaded on the first start (see 3.3).

---

## 2. Quick start (if you have done this before)

**Windows 11 (PowerShell, in the repository folder):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python scripts\setup_env.py
ollama pull qwen3.5:4b
python scripts\download_corpus.py        # plus the 2 manual downloads, see section 4.11
python -m medbridge.cli doctor
python -m medbridge.cli up --open
```

Or let the scripts do it: `.\scripts\setup.ps1 -PullModels`, then `.\scripts\run.ps1 -Open`.

**macOS (Terminal, in the repository folder):**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/setup_env.py
ollama pull qwen3.5:4b
python scripts/download_corpus.py        # plus the 2 manual downloads, see section 5.10
python -m medbridge.cli doctor
python -m medbridge.cli up --open
```

Or let the scripts do it: `scripts/setup.sh --pull-models`, then `scripts/run.sh --open`.

Then open <http://127.0.0.1:8000>. The admin dashboard is at
<http://127.0.0.1:8000/dashboard> (user `admin`, password = `ADMIN_PASSWORD` in `.env`).
The first start also downloads the reranker and the Gate 3 NLI model from Hugging Face (see
3.3 and 4.12); wait until <http://127.0.0.1:8000/health> says `healthy`.

---

## 3. Requirements

### 3.1 Hardware

| | Windows demo laptop | macOS development machine |
|---|---|---|
| Processor / GPU | Any recent x64 CPU, **NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM** | Apple Silicon (M1 or newer); the project was developed on an M5 |
| Memory | 16 GB RAM recommended | 16 GB minimum; 24 GB or more to run the 12B judge model for the study |
| Free disk | 20 GB or more (see 3.3) | 25 GB or more if you also pull the judge model |
| Internet | Needed for setup only (packages, Docker images, models, corpus) | Same |

### 3.2 Software

| Software | Version | Needed for | Windows | macOS |
|---|---|---|---|---|
| Python | **3.11** (3.12 also works). 3.13 is not tested | everything | python.org installer 3.11.9 (the last 3.11 release with a Windows installer) or `winget install Python.Python.3.11` | `brew install python@3.11` or the python.org 3.11.9 installer |
| Git | any recent | getting the code | `winget install Git.Git` | `xcode-select --install` or `brew install git` |
| Docker Desktop | recent release with **Compose v2** (`docker compose`) | PostgreSQL + Qdrant | WSL 2 backend required | Apple chip build |
| Ollama | a recent release (the `qwen3.5` models need an up-to-date Ollama) | local LLM | Ollama for Windows | Ollama for macOS |
| NVIDIA driver | recent Game Ready or Studio driver | GPU for Ollama and PyTorch | check with `nvidia-smi` | not applicable |
| LibreOffice | optional | PDF export of the report | `winget install TheDocumentFoundation.LibreOffice` | `brew install --cask libreoffice` |
| Google Chrome | optional | UI screenshots (`medbridge/web/screenshots.py`) | normal install | normal install |
| Node.js 18+ | optional | UI module tests (`node --test`) | `winget install OpenJS.NodeJS.LTS` | `brew install node` |

Reference versions on the development Mac (from `docker info` and the project's virtual
environment): Python 3.11.15, Docker Engine 29.6.2, PyTorch 2.14.1. All Python packages
are pinned in `requirements.txt`.

### 3.3 Disk space

| Item | Size | Source of the number |
|---|---|---|
| `qwen3.5:4b` (application model) | 3.3 GB | Ollama metadata recorded in `evaluation/results/model_selection/model_selection.md` |
| `gemma3:4b` (optional fallback model) | 3.4 GB | same file |
| `gemma3:12b` (judge, study only) | 8.2 GB | Ollama library page for `gemma3` |
| Docker images (`postgres:16-alpine`, `qdrant/qdrant`) | about 0.7 GB | `docker image ls` on the development Mac (411 MB + 303 MB) |
| Python virtual environment | 1.6 GB on macOS; larger on Windows with the CUDA build of PyTorch | `du -sh` of the macOS environment |
| Downloaded corpus (`data/guidelines`) | 42 MB | `du -sh data/guidelines` |
| Embedding models cache (fastembed) | 64 MB | `du -sh` of the fastembed cache on the Mac |
| Reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` (Hugging Face cache) | 91 MB | size of its `model.safetensors` in the Hugging Face cache on the Mac |
| Gate 3 model `cross-encoder/nli-MiniLM2-L6-H768` (Hugging Face cache, on by default) | 328 MB | same |
| NLI judge `cross-encoder/nli-deberta-v3-base` (study only) | 738 MB | same |

### 3.4 Models used, and why

| Role | Model | Where it runs | Why this one |
|---|---|---|---|
| Application LLM (extractor, Gate 1, Gate 2, generator) | **`qwen3.5:4b`** (Q4_K_M, 3.3 GB) | Ollama | Chosen in the model-selection experiment on the HalluBench **dev** split (the C5 pipeline run with each candidate on the final prompts; `evaluation/results/model_selection/model_selection.md` and `DECISION.md`; the superseded first round is in `evaluation/results/model_selection_round1_prefix/`). At 3.3 GB it fits in the 6 GB VRAM of the RTX 4050 with room left for the context cache. |
| Fallback application LLM | `gemma3:4b` (3.4 GB) | Ollama | The other small candidate in the same experiment. Use it only if `qwen3.5:4b` misbehaves on the laptop. |
| LLM judge (study only) | `gemma3:12b` (8.2 GB) | Ollama, on the Mac | A different model family from the application model, which avoids self-preference bias. It does not fit in 6 GB of VRAM; on Windows Ollama would run part of it on the CPU, which is very slow. Run the study on the Mac. |
| Dense embeddings | `BAAI/bge-small-en-v1.5` | fastembed (ONNX, CPU) | Small and fast. Downloaded automatically on first use. |
| Sparse embeddings | `Qdrant/bm25` | fastembed (CPU) | Keyword matching for drug names and numbers. |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | PyTorch (CUDA, MPS or CPU, chosen automatically) | Reorders the top-20 candidates to the top-5. |
| NLI judge (study only) | `cross-encoder/nli-deberta-v3-base` | PyTorch | Deterministic second judge. |
| Grounding verifier, "Gate 3" (**on by default** since the study) | `cross-encoder/nli-MiniLM2-L6-H768` | PyTorch (CUDA, MPS or CPU) | Checks every claim of a draft against its cited passages; removes unsupported sentences and withdraws a draft whose every claim is unsupported. A different model from the study's NLI judge. Switch off with `GROUNDING_VERIFIER_ENABLED=false` (the dual-gate configuration evaluated as C5, on code that also contains the post-study brand-name fix, ADL-059). |

> **Note:** `qwen3.5:4b` is both the built-in default in `medbridge/config.py` and the value in
> `.env.example`, so a `.env` created with `python scripts/setup_env.py` uses the chosen model.
> Values in `.env` override the code defaults. If your `.env` was created **before 2026-10-11**,
> open it and check these lines: `LLM_MODEL=qwen3.5:4b`, `GROUNDING_VERIFIER_ENABLED=true`,
> `GROUNDING_ENTAILMENT_THRESHOLD=0.2` and `GROUNDING_MAX_UNSUPPORTED_FRACTION=1.0` (older files
> have `false`, `0.5` and `0.34`).

---

## 4. Part A: Windows 11 (NVIDIA RTX 4050 Laptop, 6 GB)

Do the steps in order. Each step ends with a check, so you know it worked before moving on.

### 4.1 Install the prerequisites

Open **PowerShell** (Start menu, type *PowerShell*). Windows Terminal is recommended because
it handles Unicode output better than the old console.

1. **Git:** `winget install Git.Git`, or the installer from <https://git-scm.com/download/win>.
2. **Python 3.11:** `winget install Python.Python.3.11`, or the **3.11.9 Windows installer
   (64-bit)** from <https://www.python.org/downloads/windows/>. In the installer:
   - tick **"Add python.exe to PATH"**;
   - keep **"py launcher"** selected;
   - on the last page click **"Disable path length limit"** if it is offered.
3. **Docker Desktop:** `winget install Docker.DockerDesktop`, or the installer from
   <https://www.docker.com/products/docker-desktop/>.
   - Docker Desktop needs **WSL 2**. If it asks, open PowerShell **as Administrator** and run
     `wsl --install` (first time) or `wsl --update`, then restart Windows.
   - Virtualisation must be enabled in the BIOS/UEFI. Task Manager, then Performance, then
     CPU should show *Virtualization: Enabled*.
   - Start Docker Desktop once and wait until it says **Engine running**. In Settings,
     General, keep **"Use the WSL 2 based engine"** ticked.
4. **Ollama for Windows:** `winget install Ollama.Ollama`, or the installer from
   <https://ollama.com/download>. After installation Ollama runs in the system tray (the
   llama icon near the clock) and serves `http://127.0.0.1:11434`.
5. **NVIDIA driver:** install the latest driver for the RTX 4050 Laptop GPU from NVIDIA
   (GeForce Experience / NVIDIA App, or <https://www.nvidia.com/Download/index.aspx>).
6. *(Optional)* **LibreOffice** for the PDF export of the report:
   `winget install TheDocumentFoundation.LibreOffice`.

Close PowerShell and open a **new** window (so the updated PATH is picked up), then check:

```powershell
git --version
py -3.11 --version          # Python 3.11.x
docker --version
docker compose version      # must print "Docker Compose version v2..." or newer
ollama --version
nvidia-smi                  # shows the RTX 4050, the driver version and "CUDA Version: 12.x"
```

Note the **CUDA Version** shown in the top-right corner of `nvidia-smi`. It is the newest
CUDA version your driver supports, and you need it in step 4.6.

### 4.2 Allow PowerShell to run the project scripts

Windows blocks `.ps1` scripts by default. To allow scripts for your user account only, run
this once in a normal (not Administrator) PowerShell:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Answer `Y`. If your organisation does not allow that, use a one-off bypass instead:
`powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1`.

If you received the project as a **ZIP file**, Windows marks the files as downloaded from
the internet. Unblock them once, from the repository folder:

```powershell
Get-ChildItem -Recurse -Filter *.ps1 | Unblock-File
```

The venv activation script (`Activate.ps1`) also needs this policy.

### 4.3 Get the code

Choose a **short folder path** on the local disk, for example `C:\dev`. Avoid OneDrive-synced
folders such as Documents or Desktop: syncing locks files and can corrupt the virtual
environment. Avoid very deep paths too, because some corpus file names are long and Windows
has a 260-character path limit unless long paths are enabled.

```powershell
mkdir C:\dev -Force
cd C:\dev
git clone <repository-url> MedBridge      # [TO BE FILLED: repository URL]
cd MedBridge
```

If you copied the folder from a USB drive or a ZIP file instead, just `cd` into it.

> If the folder already contains a `.venv` that came from another computer (for example a
> macOS environment), delete it: `Remove-Item -Recurse -Force .venv`. Virtual environments
> cannot be moved between machines or operating systems.

Optional, recommended (Administrator PowerShell) to lift the 260-character path limit:

```powershell
git config --global core.longpaths true
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### 4.4 Option A: automatic setup (recommended)

`scripts\setup.ps1` performs steps 4.5 to 4.7 for you: it creates `.venv`, installs the CUDA
build of PyTorch when `nvidia-smi` is found, installs `requirements.txt`, creates `.env` with
fresh secrets, optionally pulls the model, and runs the doctor.

```powershell
.\scripts\setup.ps1 -PullModels
```

| Option | Meaning |
|---|---|
| `-PullModels` | also run `ollama pull` for the `LLM_MODEL` in `.env` (`qwen3.5:4b`) |
| `-Torch cu128` / `cu126` / `cu124` / `cu121` / `cpu` | choose the PyTorch build (default `auto`: `cu126` when an NVIDIA GPU is present, otherwise `cpu`) |
| `-Python "py -3.12"` | use another interpreter |
| `-VenvDir .venv-win` | use another folder for the virtual environment |
| `-SkipInstall` | only create `.env` and run the doctor |

When it finishes, activate the environment (`.\.venv\Scripts\Activate.ps1`) and continue
with **4.8**. If anything fails, follow the manual steps below; they show exactly what the
script does.

### 4.5 Option B, step 1: create and activate the virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version            # must print Python 3.11.x
```

Your prompt now starts with `(.venv)`. **Every time you open a new PowerShell window**, `cd`
to the repository and run `.\.venv\Scripts\Activate.ps1` again.

If activation fails with *"running scripts is disabled on this system"*, go back to 4.2.

### 4.6 Option B, step 2: install PyTorch with CUDA, then the requirements

The default `pip install` gives you the CPU-only PyTorch on Windows. Install the CUDA build
**first**; `requirements.txt` then keeps it.

```powershell
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Pick the build that matches the **CUDA Version** shown by `nvidia-smi`:

| `nvidia-smi` shows CUDA | Use | Newest PyTorch in that index for Python 3.11 / Windows (checked 10 Oct 2026) |
|---|---|---|
| 12.6 or higher | `.../whl/cu126` (recommended) | 2.14.1, the same version as the macOS environment |
| 12.4 or 12.5 | `.../whl/cu124` | 2.6.0 |
| 12.1 to 12.3 | `.../whl/cu121` | 2.5.1 |
| below 12.1 | update the NVIDIA driver first | |

PyTorch 2.5 is the minimum, because the `transformers` version pulled in by
`sentence-transformers` disables older PyTorch builds.

Check that PyTorch sees the GPU:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected output: a version ending in `+cu126` (or your chosen build), `True`, and
`NVIDIA GeForce RTX 4050 Laptop GPU`. If you see `+cpu` or `False`, see
[CUDA not used by PyTorch](#cuda-not-used-by-pytorch-windows).

What PyTorch is used for: the reranker, the Gate 3 NLI model (on by default) and the study's
NLI judge. The language model itself runs inside **Ollama**, which uses the GPU on its own
whether or not PyTorch has CUDA. A CPU-only PyTorch still works; reranking and Gate 3 are just
slower.

### 4.7 Option B, step 3: create `.env`

```powershell
python scripts\setup_env.py
```

This copies `.env.example` to `.env` and fills in freshly generated secrets: the encryption
master key, the session-token secret, the admin password and the database password. The
values are written to the file and never printed. Open the file to look at it:
`notepad .env`. Every variable is explained in [section 6](#6-configuration-reference-env).

> **Keep `.env` private and back it up.** Losing `ENCRYPTION_MASTER_KEY` makes every stored
> conversation unreadable. Never change `POSTGRES_PASSWORD` after the database has been
> created; see the [password troubleshooting entry](#password-authentication-failed-for-user-medbridge_app).

### 4.8 Download the language model and tune Ollama for 6 GB

Make sure Ollama is running (tray icon). Then:

```powershell
ollama pull qwen3.5:4b          # application model, 3.3 GB
ollama list                     # qwen3.5:4b must be listed
```

Optional fallback (only if `qwen3.5:4b` gives trouble; then set `LLM_MODEL=gemma3:4b` in
`.env`):

```powershell
ollama pull gemma3:4b
```

Do **not** pull `gemma3:12b` on the laptop unless you really want to run the study there. It
is 8.2 GB, does not fit in 6 GB of VRAM, and Ollama would run part of it on the CPU.

**Recommended Ollama settings for a 6 GB GPU.** These are user environment variables read by
Ollama when it starts:

```powershell
setx OLLAMA_FLASH_ATTENTION 1
setx OLLAMA_KV_CACHE_TYPE q8_0
setx OLLAMA_NUM_PARALLEL 1
setx OLLAMA_MAX_LOADED_MODELS 1
```

Then **quit Ollama** (right-click the tray icon, then *Quit Ollama*) and start it again from
the Start menu, so it picks up the new values.

- `OLLAMA_FLASH_ATTENTION=1` enables flash attention, which is required for a quantised
  context cache.
- `OLLAMA_KV_CACHE_TYPE=q8_0` stores the context (KV) cache in 8-bit, roughly halving its
  memory compared with 16-bit.
- `OLLAMA_NUM_PARALLEL=1` uses one decoding slot. Every extra slot reserves another
  context-sized cache.
- `OLLAMA_MAX_LOADED_MODELS=1` keeps only one model in VRAM.

**VRAM budget.** The model weights take about 3.3 GB. The context cache for
`LLM_NUM_CTX=8192` and the runtime buffers come on top, and Windows, the browser and other
programs also use some VRAM. With a CUDA build of PyTorch, the reranker and the Gate 3 NLI
model are loaded on the same GPU as well (their weight files are about 91 MB and 328 MB; see
3.3). Check after the first chat message:

```powershell
ollama ps        # the PROCESSOR column should say "100% GPU"
nvidia-smi       # memory usage of the GPU
```

If `ollama ps` shows a CPU/GPU split, or you get out-of-memory errors, see
[Ollama out of memory](#ollama-out-of-memory-or-very-slow-6-gb-gpu).

### 4.9 Start PostgreSQL and Qdrant (Docker)

Docker Desktop must be running (whale icon in the tray, *Engine running*).

```powershell
docker compose up -d
docker compose ps
```

The first run downloads the `postgres:16-alpine` and `qdrant/qdrant` images. Wait until
both services show `(healthy)` in the STATUS column; this can take up to a minute. Check
Qdrant directly:

```powershell
Invoke-RestMethod http://127.0.0.1:6333/readyz     # prints: all shards are ready
```

> In Windows PowerShell 5.1, `curl` is an alias for `Invoke-WebRequest`. Use
> `Invoke-RestMethod`, or `curl.exe` for the real curl.

Qdrant also has a small web dashboard at <http://127.0.0.1:6333/dashboard>.

### 4.10 Create the database schema

```powershell
python -m alembic upgrade head
python -m alembic current          # must end with "(head)"
```

Use `python -m alembic` rather than a bare `alembic`; it always runs the venv's copy.

### 4.11 Download and index the knowledge base

The 27 curated documents are listed in `data/corpus_manifest.json`. They are not stored in
git, because several publishers do not allow redistribution (see `data/README.md`).

**Step 1: automatic downloads (25 documents).**

```powershell
python scripts\download_corpus.py
```

The script downloads what it can and checks each file's SHA-256 against the manifest. A
`drift` result is normal for files marked `volatile` (NICE PDFs and web pages change on every
download); ingestion handles it.

**Step 2: two manual downloads.** These publisher sites show a browser check to scripts, so
download them in your browser and save them under the exact names below:

| ID | Document | Get it from | Save as |
|---|---|---|---|
| `AHA_ACC_HTN_2025` | 2025 AHA/ACC High Blood Pressure Guideline (Circulation 2025;152:e114-e218) | <https://doi.org/10.1161/CIR.0000000000001356>, then "PDF" | `data/guidelines/jones-et-al-2025-2025-aha-acc-aanp-aapa-abc-accp-acpm-ags-ama-aspc-nma-pcna-sgim-guideline-for-the-prevention-detection.pdf` |
| `ESC_HTN_2024` | 2024 ESC Guidelines for elevated blood pressure and hypertension (Eur Heart J 2024;45:3912-4018) | <https://doi.org/10.1093/eurheartj/ehae178>, then "PDF" | `data/guidelines/ehae178.pdf` |

The browser saves files to your Downloads folder. Move and rename them, replacing
`<downloaded-file>` with the real name:

```powershell
Move-Item "$env:USERPROFILE\Downloads\<downloaded-file>.pdf" "data\guidelines\ehae178.pdf"
```

Then check that everything is present:

```powershell
python scripts\download_corpus.py --verify-only
```

`data/README.md` also lists documents that were wanted but could not be obtained by script,
and what replaces them in the corpus.

**Step 3: index the documents into Qdrant.** This parses, chunks and embeds every document.
The first run also downloads the fastembed embedding models (internet needed; the chunker
uses the dense model's tokenizer from the same cache) and takes a few minutes. The reranker and the Gate 3 model are downloaded
later, on the first start of the API (4.12).

```powershell
python -m medbridge.ingestion --recreate --no-manifest-update
```

- `--recreate` rebuilds the collection from scratch. Without it, ingestion is incremental:
  unchanged documents are skipped, so re-running it is cheap.
- `--no-manifest-update` keeps the committed `data/corpus_manifest.json` unchanged. Without
  it, ingestion records the new file hashes and chunk counts in the manifest, which is what
  the team does after adding documents on the development machine.
- `python -m medbridge.ingestion --dry-run` only parses and chunks (no Qdrant writes).
- `python -m medbridge.ingestion --only NICE_NG136 WHO_HTN_PHARM_2021` re-indexes selected
  documents.
- `python -m medbridge.ingestion --stats` writes `reports/corpus_stats.md`/`.json` and the
  corpus composition figure for the report.

Check the number of indexed chunks:

```powershell
(Invoke-RestMethod http://127.0.0.1:6333/collections/clinical_guidelines).result.points_count
```

For reference, the development index held 2,747 chunks (`qdrant_points` in
`reports/smoke_test_latest.json`). Your count can differ slightly because volatile documents
change between downloads.

### 4.12 Start the backend and open the UI

```powershell
python -m uvicorn medbridge.main:app --host 127.0.0.1 --port 8000
```

Leave this window open; the server runs until you press **Ctrl+C**. At start-up the app
loads the embedders, the reranker, the Gate 3 NLI model and the Ollama model in the background
(warm-up). **The first start needs internet**: it downloads `cross-encoder/ms-marco-MiniLM-L-6-v2`
and `cross-encoder/nli-MiniLM2-L6-H768` from Hugging Face (about 0.4 GB together, see 3.3) into
`%USERPROFILE%\.cache\huggingface\hub` (for example
`C:\Users\<you>\.cache\huggingface\hub`); later starts load them from there. Check readiness
in a second PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

| `status` | Meaning |
|---|---|
| `healthy` | everything is ready |
| `degraded` | storage is up, but warm-up is still running, Ollama is unreachable or a model is missing |
| `unhealthy` (HTTP 503) | PostgreSQL or Qdrant is unreachable |

The response also shows `postgres_connected`, `qdrant_connected`, `ollama_reachable`,
`models_loaded`, `qdrant_points` and `local_only_mode`. With warm-up on, `healthy` needs every
entry of `models_loaded` to be `true`: `embedder_dense`, `embedder_sparse`, `reranker`, `llm`
and, while Gate 3 is enabled, `nli_verifier`. On the first start the two Hugging Face downloads
keep the status at `degraded` for a while.

Now open:

- **Chat UI:** <http://127.0.0.1:8000>. Accept the privacy notice on the first visit.
- **Admin dashboard:** <http://127.0.0.1:8000/dashboard>. Log in with user `admin` (or your
  `ADMIN_USERNAME`) and the password from `.env`:

  ```powershell
  Select-String -Path .env -Pattern '^ADMIN_PASSWORD='
  ```

### 4.13 Legacy Streamlit UI (optional)

The original Streamlit front end still works against the same backend. With the backend
running, open a **second** PowerShell window:

```powershell
cd C:\dev\MedBridge
.\.venv\Scripts\Activate.ps1
python -m streamlit run medbridge\frontend\app.py
```

Open <http://localhost:8501> (Streamlit runs headless, so it does not open the browser by
itself). It talks to `http://localhost:8000` by default. Set `$env:MEDBRIDGE_API_URL` before
starting Streamlit if the API runs elsewhere.

### 4.14 One-command start (daily use)

After the first setup you only need:

```powershell
cd C:\dev\MedBridge
.\scripts\run.ps1 -Open
```

`run.ps1` starts Docker Desktop and the Ollama app if they are installed but not running,
then calls `python -m medbridge.cli up`, which:

1. creates `.env` if it is missing;
2. runs `docker compose up -d` and waits until PostgreSQL is healthy and Qdrant is ready;
3. runs `python -m alembic upgrade head`;
4. indexes the corpus **only if the Qdrant collection is missing or empty**;
5. warns (without stopping) if Ollama or the model is missing;
6. starts uvicorn on `http://127.0.0.1:8000` and, with `-Open`, opens the browser.

| `run.ps1` option | Meaning |
|---|---|
| `-Open` | open the UI in the browser when the server answers |
| `-DryRun` | print every command, change nothing |
| `-Port 8001` | use another port |
| `-NoIngest` | never run the ingestion step |
| `-NoServer` | prepare everything but do not start the API |
| `-SkipDocker` | do not touch Docker (services managed elsewhere) |
| `-Wait 300` | wait longer for healthy services (seconds) |
| `-BindHost 127.0.0.1` | bind address passed to `up --host` (keep loopback) |
| `-VenvDir .venv-win` | use another virtual-environment folder |

The same works without the script, with the venv active: `python -m medbridge.cli up --open`
(section 7).

Press **Ctrl+C** to stop the API. The containers keep running in the background; stop them
with `docker compose stop` if you want the memory back.

### 4.15 First-run checklist for the demo laptop

The system was developed and measured on the Mac. Run this checklist on the Windows laptop
once, well before the review, and note the results.

| Check | Command or action | Expected |
|---|---|---|
| Doctor | `python -m medbridge.cli doctor` | `0 problem(s)` |
| PyTorch uses CUDA | the `python -c "import torch; ..."` line from 4.6 | `True`, RTX 4050 |
| Model fully on GPU | send one message, then `ollama ps` | `100% GPU` |
| `.env` matches the final configuration | `Select-String -Path .env -Pattern '^(LLM_MODEL\|GROUNDING_)'` | `qwen3.5:4b`, `GROUNDING_VERIFIER_ENABLED=true`, `0.2`, `1.0` (see the note in 3.4) |
| Health | `Invoke-RestMethod http://127.0.0.1:8000/health` | `status: healthy`, `models_loaded.nli_verifier: True` |
| Gate 3 runs on the GPU | after the first answer, `nvidia-smi` lists the Python process next to Ollama | Python uses some VRAM (reranker and Gate 3); `ollama ps` still `100% GPU` |
| Smoke test | `python -m medbridge.cli smoke --output reports\smoke_test_windows.json` | all checks PASS |
| UI in Edge and Chrome | open the UI, run the demo in section 12, try dark mode | no errors |

Record the outcome: [TO BE FILLED: Windows dry-run date, doctor output, smoke-test result, notes].

---

## 5. Part B: macOS (Apple Silicon)

### 5.1 Install the prerequisites

1. **Homebrew** (optional but convenient), from <https://brew.sh>.
2. **Python 3.11:** `brew install python@3.11`, or the python.org **3.11.9 macOS 64-bit
   universal2 installer** from <https://www.python.org/downloads/macos/>.
3. **Git:** `xcode-select --install` (Apple's command line tools) or `brew install git`.
4. **Docker Desktop for Mac (Apple chip):** from
   <https://www.docker.com/products/docker-desktop/>. Open it once and wait for *Engine
   running*.
5. **Ollama for macOS:** download from <https://ollama.com/download>, move `Ollama.app` to
   Applications and open it. It runs in the menu bar and serves `http://127.0.0.1:11434`.
   Apple Silicon GPUs are used automatically through Metal.
6. *(Optional)* LibreOffice: `brew install --cask libreoffice`.

Check:

```bash
git --version
python3.11 --version
docker --version
docker compose version
ollama --version
```

### 5.2 Get the code

```bash
mkdir -p ~/dev && cd ~/dev
git clone <repository-url> MedBridge      # [TO BE FILLED: repository URL]
cd MedBridge
```

If the folder contains a `.venv` copied from a Windows machine (it has `Scripts\` instead of
`bin/`), delete it with `rm -rf .venv`, or use another folder name such as `.venv-mac`.
`.gitignore` ignores every `.venv*` folder.

If the shell scripts fail with `bad interpreter: /bin/bash^M`, they were checked out with
Windows line endings. Fix with `sed -i '' 's/\r$//' scripts/*.sh` (and set
`git config core.autocrlf input`). If they fail with `permission denied` (a copy that lost
the executable bit, for example from a zip or a Windows drive), run `chmod +x scripts/*.sh`.

### 5.3 Option A: automatic setup (recommended)

```bash
scripts/setup.sh --pull-models
```

It finds Python 3.11/3.12, creates `.venv`, installs `requirements.txt` (the PyPI build of
PyTorch already supports Apple's MPS GPU backend), creates `.env`, pulls the model and runs
the doctor. Options: `--python python3.12`, `--venv .venv-mac`, `--skip-install`,
`--pull-models`. Then `source .venv/bin/activate` and continue with **5.7**.

### 5.4 Option B, step 1: create and activate the virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version             # Python 3.11.x
```

Run `source .venv/bin/activate` again in every new Terminal window.

### 5.5 Option B, step 2: install the packages

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, 'MPS:', torch.backends.mps.is_available())"
```

Expected: `MPS: True`. The reranker then runs on the Apple GPU; it falls back to the CPU
automatically if MPS is unavailable.

### 5.6 Option B, step 3: create `.env`

```bash
python scripts/setup_env.py
open -e .env          # view it in TextEdit (optional)
```

The same rules as on Windows apply (4.7): keep the file private and back it up.

### 5.7 Download the language model

```bash
ollama pull qwen3.5:4b           # application model
ollama list
```

For the hallucination study on the Mac you also need the judge model:

```bash
ollama pull gemma3:12b           # 8.2 GB, study only
```

Optional Ollama settings (the menu-bar app reads them via `launchctl`; quit and reopen
Ollama afterwards; they reset at reboot):

```bash
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0
```

For the study, use the dedicated server described in [section 10](#10-reproducing-the-hallucination-study)
instead.

### 5.8 Start PostgreSQL and Qdrant

```bash
docker compose up -d
docker compose ps                 # wait for (healthy) on both services
curl http://127.0.0.1:6333/readyz
```

### 5.9 Create the database schema

```bash
python -m alembic upgrade head
python -m alembic current         # ends with "(head)"
```

### 5.10 Download and index the knowledge base

```bash
python scripts/download_corpus.py
# manual downloads: AHA_ACC_HTN_2025 and ESC_HTN_2024, see the table in 4.11
mv ~/Downloads/<downloaded-file>.pdf data/guidelines/ehae178.pdf
python scripts/download_corpus.py --verify-only
python -m medbridge.ingestion --recreate        # add --no-manifest-update to keep the manifest unchanged
python -m medbridge.ingestion --stats           # corpus tables + figure for the report
curl -s http://127.0.0.1:6333/collections/clinical_guidelines | python -m json.tool | grep points_count
```

The options are explained in 4.11.

### 5.11 Start the backend and open the UI

```bash
python -m uvicorn medbridge.main:app --host 127.0.0.1 --port 8000
```

The first start downloads the reranker and the Gate 3 model from Hugging Face into
`~/.cache/huggingface/hub` (see 4.12). In a second Terminal window:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool     # healthy once every models_loaded entry is true
grep '^ADMIN_PASSWORD=' .env       # dashboard password
```

Open <http://127.0.0.1:8000> and <http://127.0.0.1:8000/dashboard>.

### 5.12 Legacy Streamlit UI (optional)

```bash
source .venv/bin/activate
python -m streamlit run medbridge/frontend/app.py      # then open http://localhost:8501
```

### 5.13 One-command start

```bash
scripts/run.sh --open
```

`run.sh` opens Docker Desktop and the Ollama app if they are installed but not running, then
calls `python -m medbridge.cli up` with your options (`--open`, `--dry-run`, `--port 8001`,
`--no-ingest`, `--no-server`, `--skip-docker`, `--wait 300`; see section 7). It uses `.venv`,
or `.venv-mac` if `.venv` does not exist (override with `VENV_DIR=...`).

---

## 6. Configuration reference (`.env`)

`.env` is read by the application (`medbridge/config.py`, pydantic-settings), by Alembic and
by Docker Compose (for `POSTGRES_PASSWORD`). Variables set in your shell override the file.
`.env.example` contains every setting with a comment and a safe default. Do not put spaces
around `=` and do not add comments after a value on the same line.

### 6.1 How the secrets are generated

`python scripts/setup_env.py` fills each of these only if it is empty (`--force`
regenerates them), using Python's cryptographically secure `secrets` module:

| Variable | How it is generated | What it protects |
|---|---|---|
| `ENCRYPTION_MASTER_KEY` | 32 random bytes, URL-safe base64 (the Fernet key format) | the key-encryption key that wraps each session's data key; message text, audit text and event payloads are encrypted at rest |
| `SESSION_TOKEN_SECRET` | `secrets.token_urlsafe(48)` | HMAC signature of session tokens and admin tokens |
| `ADMIN_PASSWORD` | `secrets.token_urlsafe(15)` | dashboard login |
| `POSTGRES_PASSWORD` | `secrets.token_urlsafe(24)` | the database user `medbridge_app`; `DATABASE_URL` is rewritten to use it |

To generate a value by hand (for example on a machine without the script):

```bash
python -c "import base64, secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"   # Fernet key
python -c "import secrets; print(secrets.token_urlsafe(48))"                                          # token secret
```

What happens when a secret changes:

- **`ENCRYPTION_MASTER_KEY`:** conversations stored with the old key become unreadable; the
  app treats them like erased sessions.
- **`SESSION_TOKEN_SECRET`:** all issued session and admin tokens become invalid; users start
  a new chat and log in again.
- **`POSTGRES_PASSWORD`:** the database volume keeps the password it was created with.
  Changing it in `.env` breaks the connection unless you also reset the volume.

### 6.2 All variables

**LLM inference**

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, `groq` or `openai`. Remote providers are optional and ignored while `LOCAL_ONLY_MODE=true`. |
| `LOCAL_ONLY_MODE` | `true` | Guarantees that no text leaves the machine: remote providers are never contacted and only a loopback Ollama address (`127.0.0.1`, `localhost`, `::1`) is accepted. |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Native Ollama endpoint (a trailing `/v1` is tolerated). |
| `LLM_MODEL` | `qwen3.5:4b` | Model for the extractor, both gates and the generator. |
| `LLM_MODEL_GATES` | empty | Optional separate model for the extractor and both gates (empty means `LLM_MODEL`). |
| `LLM_MODEL_GENERATOR` | empty | Optional separate model for the response generator. |
| `LLM_KEEP_ALIVE` | `30m` | How long Ollama keeps the model loaded after the last request. |
| `LLM_NUM_CTX` | `8192` | Context window in tokens. Lower it only for memory problems and keep it at 6144 or more: the gate and generator prompts carry up to five evidence chunks, and Ollama shortens over-long prompts without an error. |
| `LLM_THINK` | `false` | Keeps the "thinking" traces of qwen3/qwen3.5 switched off (faster, valid JSON). |
| `LLM_TIMEOUT` | `120.0` | Seconds per LLM request before it counts as failed; the pipeline then follows its safe fallbacks. |
| `LLM_TEMPERATURE_GATES` | `0.0` | Gates are deterministic (architecture invariant). |
| `LLM_TEMPERATURE_GENERATOR` | `0.3` | Low temperature for the generator. |
| `LLM_MAX_TOKENS_GATES` | `512` | Output token limit for the extractor and gates. |
| `LLM_MAX_TOKENS_GENERATOR` | `900` | Output token limit for the generator. |
| `LLM_SEED` | `42` | Fixed seed (architecture invariant). |
| `LLM_MAX_RETRIES` | `3` | Retries for a failed LLM call. |
| `LLM_RETRY_BASE_DELAY` | `1.0` | Base delay of the exponential back-off (seconds). |
| `GROQ_API_KEY`, `GROQ_MODEL` | empty, `llama-3.1-8b-instant` | Optional Groq provider (only with `LOCAL_ONLY_MODE=false`). |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | empty, `gpt-4o-mini` | Optional OpenAI provider (only with `LOCAL_ONLY_MODE=false`). |

**Database**

| Variable | Default | Meaning |
|---|---|---|
| `POSTGRES_PASSWORD` | generated | Password of `medbridge_app`, used by Docker Compose when the volume is first created. |
| `DATABASE_URL` | `postgresql+asyncpg://medbridge_app:<password>@127.0.0.1:5433/medbridge` | Application database. Port 5433 is the host port from `docker-compose.yml`. |
| `EVAL_DATABASE_URL` | commented out | Separate database for live benchmarks and the study (section 9.1). Not a setting of the app itself. |

**Vector store and retrieval**

| Variable | Default | Meaning |
|---|---|---|
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant REST endpoint. |
| `QDRANT_COLLECTION` | `clinical_guidelines` | Collection with the indexed chunks. |
| `RETRIEVAL_TOP_K_CANDIDATES` | `20` | Candidates after dense + sparse search fused with Reciprocal Rank Fusion (k = 60). |
| `RETRIEVAL_TOP_K_RERANKED` | `5` | Chunks kept after cross-encoder reranking. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Dense model. Changing it requires `python -m medbridge.ingestion --recreate`. |
| `SPARSE_MODEL` | `Qdrant/bm25` | Sparse (keyword) model. Same rule. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker. |
| `RETRIEVAL_CACHE_SIZE` | `512` | Entries in the in-process query/result cache. |
| `RETRIEVAL_CACHE_TTL_SECONDS` | `900` | Cache lifetime. |
| `SPECULATIVE_RETRIEVAL` | `true` | Starts retrieval while Gate 1 runs; the result is discarded on SOFT-ASK, so routing is unchanged. |
| `EVIDENCE_MAX_CHARS_PER_CHUNK` | `2400` | Characters of each chunk passed to the prompts. |

**Pipeline safety behaviour**

| Variable | Default | Meaning |
|---|---|---|
| `SOFT_ASK_MAX_COUNT` | `2` | Soft-ask loop-breaker: after two clarification questions the system answers in general terms (GENERALIZE). Architecture invariant: keep 2. |
| `EXTRACTOR_HISTORY_TURNS` | `4` | Previous turns given to the extractor ("is that dose ok?"). |
| `CITATION_VALIDATION_ENABLED` | `true` | Removes citation markers that do not map to a retrieved chunk; an ANSWER left with no valid citation is downgraded to GENERALIZE. |
| `SMALL_TALK_ENABLED` | `true` | Fixed replies to greetings and thanks, with no clinical claims. |
| `EVIDENCE_GATE_CALIBRATION_ENABLED` | `false` | V1: calibrated Evidence Gate (quantitative evidence signal + votes). |
| `EVIDENCE_GATE_VOTES` | `3` | Self-consistency votes for V1. |
| `EVIDENCE_MIN_RERANK_ANSWER`, `EVIDENCE_MIN_RERANK_ANY`, `EVIDENCE_MIN_ENTITY_COVERAGE` | `0.0`, `-6.0`, `0.0` | V1 thresholds. Calibrated values come from `evaluation/results/calibration.json` after `python -m evaluation.calibrate`. |
| `GROUNDING_VERIFIER_ENABLED` | `true` | V2: post-generation grounding verifier ("Gate 3"); downloads its NLI model on the first start. `false` gives the dual-gate configuration evaluated as C5, on code that also contains the post-study brand-name fix (ADL-059), so not identical to the evaluated pipeline. |
| `GROUNDING_NLI_MODEL` | `cross-encoder/nli-MiniLM2-L6-H768` | Gate 3 model. |
| `GROUNDING_ENTAILMENT_THRESHOLD`, `GROUNDING_MAX_UNSUPPORTED_FRACTION` | `0.2`, `1.0` | Gate 3 thresholds θ and τ, calibrated on the dev split (`evaluation/results/calibration.json`). With τ = 1.0 a partly supported draft is only shortened; a draft whose every claim is unsupported is withdrawn (ANSWER → regenerated GENERALIZE → ABSTAIN; GENERALIZE → ABSTAIN). |
| `WARMUP_ON_STARTUP` | `true` | Loads the models in the background at start-up, so the first chat message is not slowed down by model loading. |

Production setting after the study (ADL-051, ADL-052, ADL-061 in
`docs/architecture_decision_log.md`): **Gate 3 (V2) on** with θ = 0.2 and τ = 1.0, because it
significantly reduced hallucination against the dual gate; **V1 off**, because it gave no
significant benefit and costs two extra Gate 2 calls. The V1 threshold defaults are
uncalibrated placeholders; copy `v1.env_lines` from `calibration.json` if you switch V1 on.

**Privacy and security**

| Variable | Default | Meaning |
|---|---|---|
| `PII_REDACTION_ENABLED` | `true` | Redacts names, phone numbers, e-mails, IDs (Aadhaar, PAN, SSN, MRN, insurance), addresses and PIN codes, exact birth dates (converted to an age), URLs and IP addresses before any LLM call and before storage. Clinical values are kept. |
| `ENCRYPTION_ENABLED` | `true` | Field-level encryption at rest with per-session data keys. |
| `ENCRYPTION_MASTER_KEY` | generated | See 6.1. |
| `SESSION_TOKEN_SECRET` | generated | See 6.1. |
| `SESSION_TOKEN_TTL_SECONDS` | `604800` | Session token lifetime (7 days). |
| `AUTH_REQUIRED` | `true` | Every session route needs the signed bearer token. |
| `ADMIN_USERNAME` | `admin` | Dashboard user. |
| `ADMIN_PASSWORD` | generated | Dashboard password; empty disables the dashboard login. |
| `RATE_LIMIT_PER_MINUTE` | `30` | Message requests per minute per client IP (`0` disables the limit). |
| `MAX_REQUEST_BYTES` | `32768` | Maximum request body size. |
| `DATA_RETENTION_DAYS` | `30` | Sessions older than this are purged by the background retention job. |

**Server and UI**

| Variable | Default | Meaning |
|---|---|---|
| `HOST` | `127.0.0.1` | Bind address. Keep loopback; `0.0.0.0` exposes the API to your network. |
| `PORT` | `8000` | API and UI port. |
| `CORS_ORIGINS` | `["http://127.0.0.1:8000","http://localhost:8000","http://127.0.0.1:8501","http://localhost:8501"]` | Browser origins allowed to call the API (a JSON list). |
| `APP_NAME` | `MedBridge` | Name shown in the UI. |
| `EMERGENCY_NUMBER`, `AMBULANCE_NUMBER`, `EMERGENCY_REGION_LABEL` | `112`, `108`, `India` | Shown on the ESCALATE emergency card. |

Environment variables that are read **only from the shell** (not from `.env`):
`MEDBRIDGE_API_URL` and `MEDBRIDGE_API_TIMEOUT` (Streamlit UI), `TEST_DATABASE_URL` (tests),
and the Ollama variables `OLLAMA_*` (read by Ollama itself).

---

## 7. The launcher: `python -m medbridge.cli`

A small cross-platform launcher in `medbridge/cli.py`. It uses only the Python standard
library, so `doctor` also works before the packages are installed. It never prints secrets.
Run it from the repository root with the venv active.

| Command | What it does |
|---|---|
| `python -m medbridge.cli doctor` | Checks the Python version, the venv, the installed packages, the PyTorch device (CUDA, MPS or CPU), `.env` secrets, Docker and Compose, PostgreSQL, the Alembic revision, the Qdrant collection, the corpus files, Ollama and the configured models, and whether the API port is free. Prints a fix for every problem. Exit code 1 if anything failed. `--json` gives machine-readable output. |
| `python -m medbridge.cli up` | `.env` if missing, then `docker compose up -d`, wait for healthy services, `alembic upgrade head`, ingestion if the collection is empty, a model check, then uvicorn in the foreground. |
| `python -m medbridge.cli ingest [args]` | Runs `python -m medbridge.ingestion` with your arguments, e.g. `ingest --recreate`, `ingest --stats`. Use `ingest -- --dry-run` to pass ingestion's own `--dry-run`. |
| `python -m medbridge.cli smoke [args]` | Runs `scripts/smoke_test.py` against `http://HOST:PORT` from `.env`; extra arguments are passed on (`--timeout 300`, `--output ...`). |
| `python -m medbridge.cli --dry-run <command>` | Prints the commands that would run and changes nothing (also accepted after the command). |

Options of `up`:

| Option | Default | Meaning |
|---|---|---|
| `--host` | `HOST` from `.env` | bind address |
| `--port` | `PORT` from `.env` | port |
| `--wait` | `180` | seconds to wait for healthy services |
| `--no-ingest` | | never run ingestion |
| `--no-server` | | stop after preparing everything |
| `--skip-docker` | | do not call Docker Compose |
| `--open` | | open the browser when the server answers |

Example `doctor` output (macOS, abridged, with Ollama deliberately not running):

```text
MedBridge doctor  (Darwin, repo: MedBridge)
[ OK ] Python           3.11.15 (.venv/bin/python)
[ OK ] Virtual env      running inside a virtual environment
[ OK ] Python packages  14 runtime packages found
[ OK ] PyTorch          2.14.1 on mps
[ OK ] .env             secrets present, DATABASE_URL matches
[ OK ] Docker           engine 29.6.2
[ OK ] PostgreSQL       127.0.0.1:5433 reachable
[ OK ] Migrations       database schema at head
[ OK ] Qdrant           collection 'clinical_guidelines': 2747 points
[ OK ] Corpus           all 27 documents present in data/guidelines
[FAIL] Ollama           http://127.0.0.1:9 not reachable
                        fix: Start Ollama: open the Ollama app (or run 'ollama serve' in a separate terminal).
[ OK ] API port         127.0.0.1:8765 is free

11 ok, 0 warning(s), 1 problem(s), 0 skipped
```

---

## 8. Running the tests

The tests are hermetic by default. Tests that need a service carry a marker and are
**skipped automatically** when that service is not reachable, so `pytest` is green on a
laptop without Docker or Ollama.

| Marker | Needs | Default run |
|---|---|---|
| `postgres` | PostgreSQL on 5433 | runs if reachable, else skipped |
| `qdrant` | Qdrant with the indexed corpus | runs if reachable, else skipped |
| `ollama` | Ollama with the configured model (real LLM calls) | runs if reachable, else skipped |
| `live` | the full stack with a real LLM | **excluded** unless you pass `-m live` |
| `slow` | long-running | runs |

The tests never touch your development database: `conftest.py` points `DATABASE_URL` at a
sibling database `medbridge_test`, which it creates and migrates automatically.

```bash
python -m pytest                                  # everything except "live"
python -m pytest -m "not ollama"                  # never call the LLM (fast, deterministic)
python -m pytest medbridge/tests/unit             # unit tests only
python -m pytest medbridge/tests/integration      # integration tests (PostgreSQL/Qdrant)
python -m pytest tests/privacy tests/evaluation tests/adversarial tests/scripts
python -m pytest -m live                          # real end-to-end turns (stack + Ollama + corpus)
python -m pytest medbridge/tests/unit/test_cli.py # the launcher itself
```

Coverage (the measured packages, `medbridge` and `evaluation`, are configured in
`pyproject.toml`):

```bash
python -m pytest --cov --cov-report=term-missing --cov-report=html
```

Open `htmlcov/index.html` for the line-by-line report.

The totals and coverage quoted in the README, the report and the slides come from
`scripts/test_summary.py`. It runs pytest once with JUnit XML and coverage output and writes
`reports/test_summary.json`, `reports/tests/junit.xml` and `reports/tests/coverage.json`
(the same command works in PowerShell with `scripts\test_summary.py`):

```bash
python scripts/test_summary.py                                # default run (no "live" tests)
python scripts/test_summary.py --pytest-args "-m 'not ollama'"   # extra pytest arguments
python scripts/render_docs_results.py                         # then refresh the numbers in the docs
```

Do not type test counts into documents by hand; re-run these two commands after the last
code change.

Code quality and the UI module tests:

```bash
python -m ruff check .
python -m mypy medbridge
node --test medbridge/web/tests                   # optional, needs Node.js 18+
```

On Windows, write the paths with `\` or `/`; both work with pytest.

---

## 9. Live benchmarks and measurements

These talk to the real pipeline and the real LLM. Make sure the services are up (`doctor`
is green) and nothing else heavy is using the GPU.

### 9.1 A separate evaluation database (once)

Live benchmarks and the study create one session per test item. Keep them out of the
development database:

```bash
docker compose exec postgres createdb -U medbridge_app medbridge_eval
```

Copy the `DATABASE_URL` line in `.env`, rename it to `EVAL_DATABASE_URL` and change the
database name at the end from `/medbridge` to `/medbridge_eval`. Then create the schema in
that database:

```powershell
# Windows PowerShell
$env:DATABASE_URL = (Select-String -Path .env -Pattern '^EVAL_DATABASE_URL=(.+)$').Matches[0].Groups[1].Value
python -m alembic upgrade head
Remove-Item Env:DATABASE_URL
```

```bash
# macOS
DATABASE_URL="$(grep -E '^EVAL_DATABASE_URL=' .env | cut -d= -f2-)" python -m alembic upgrade head
```

The benchmark runners in `tests/` and `scripts/run_study.sh` read `EVAL_DATABASE_URL` from
`.env`, and they refuse to run against the development database. `python -m evaluation.run`
reads it **only from the shell**, and otherwise falls back to `DATABASE_URL`. Export it
before running the study:

```powershell
$env:EVAL_DATABASE_URL = (Select-String -Path .env -Pattern '^EVAL_DATABASE_URL=(.+)$').Matches[0].Groups[1].Value
```

```bash
export EVAL_DATABASE_URL="$(grep -E '^EVAL_DATABASE_URL=' .env | cut -d= -f2-)"
```

### 9.2 Smoke test (one query per action + PII + streaming + erasure)

With the API running:

```bash
python -m medbridge.cli smoke                     # or: python scripts/smoke_test.py --base-url http://127.0.0.1:8000
```

It checks `/health`, session creation, ESCALATE, ABSTAIN, SOFT-ASK followed by an answer,
GENERALIZE, PII redaction, the SSE stream, feedback, the context card, the clinician
summary, the missing-token guard and erasure. It prints PASS/FAIL per check, writes
`reports/smoke_test_latest.json` and exits with 1 if any check fails. LLM-routed cases accept
a small set of safe actions, because routing can vary between models. Use
`--output <file>` to keep the committed result file unchanged.

### 9.3 Latency probe and the before/after table

`scripts/latency_probe.py` sends five fixed scenarios (six turns: emergency, out-of-domain,
clarification, a two-turn clinical question, lifestyle) to a running server: one cold pass,
then `--repeats` warm passes, and stores client and per-stage server timings. The recorded
files were made like this (step `perf_after` of `scripts/run_post_study.sh`):

1. Start the API with `WARMUP_ON_STARTUP=false` and the model you want to measure
   (`LLM_MODEL=...`), and unload that model from Ollama, so the first pass is a real cold start:

   ```bash
   curl -s http://127.0.0.1:11434/api/generate -d '{"model":"gemma3:4b","keep_alive":0}'
   ```

   ```powershell
   Invoke-RestMethod http://127.0.0.1:11434/api/generate -Method Post -ContentType 'application/json' -Body '{"model":"gemma3:4b","keep_alive":0}'
   ```

2. Probe, then compare with the baseline of the received code (`before_gemma3-4b.json`). Use
   the same model on both sides for a fair comparison:

   ```bash
   python scripts/latency_probe.py --base-url http://127.0.0.1:8000 --label after_gemma3-4b --repeats 3 --out reports/performance/after_gemma3-4b.json
   python -m evaluation.perf --before reports/performance/before_gemma3-4b.json --after reports/performance/after_gemma3-4b.json
   ```

3. Optional: restart with `WARMUP_ON_STARTUP=true` and record only the first pass
   (`--cold-only --label after_warmup_gemma3-4b --out reports/performance/after_warmup_gemma3-4b.json`).

`evaluation.perf` writes `reports/performance/latency_table.md` (+ `.csv`) and the figure
`reports/figures/latency_before_after.png`. Without `--before`/`--after` it compares the newest
files labelled exactly `before` and `after`, which the recorded files are not, so pass both
paths. The README shows the table, a per-stage breakdown and the warm-up comparison.

### 9.4 Load test (1, 5 and 10 concurrent users)

All simulated users share one IP address, so start a separate server with a higher rate
limit. The script prints the value it needs:

```bash
python scripts/load_test.py --print-server-env --users 1,5,10 --requests-per-user 3
```

```powershell
# Windows: second server on port 8100 with the printed limit (example value 60)
$env:RATE_LIMIT_PER_MINUTE = "60"; python -m uvicorn medbridge.main:app --host 127.0.0.1 --port 8100
```

```bash
# macOS
RATE_LIMIT_PER_MINUTE=60 python -m uvicorn medbridge.main:app --host 127.0.0.1 --port 8100
```

Then, in another window:

```bash
python scripts/load_test.py --base-url http://127.0.0.1:8100 --users 1,5,10 --requests-per-user 3
```

Results go to `reports/performance/load_test.json` and `load_test.md` (`--output-dir` changes
the folder). A single local Ollama server handles one LLM call at a time, so latency grows with
the number of users. The recorded run (step `load` of `scripts/run_post_study.sh`) used a server
on port 8000 with `RATE_LIMIT_PER_MINUTE=600`, `WARMUP_ON_STARTUP=true` and `qwen3.5:4b`.

### 9.5 MedBridge-AQ, RABBITS and adversarial suites (live)

They use the production pipeline in-process (no API server needed) with the model and Gate 3
setting from `.env`, and write to the evaluation database (`EVAL_DATABASE_URL` from `.env`, or
`--database-url`).

```bash
python -m tests.benchmark.benchmark_runner --mode live --all --resume       # MedBridge-AQ routing benchmark (275 vignettes)
python -m tests.adversarial.rabbits_runner --mode live --resume --brands "$(cat reports/rabbits_before_fix/brands.txt)"   # the 60 recorded brands
python -m tests.adversarial.adversarial_runner                              # 19 prompt injection / jailbreak payloads
```

```powershell
python -m tests.benchmark.benchmark_runner --mode live --all --resume
python -m tests.adversarial.rabbits_runner --mode live --resume --brands ((Get-Content reports\rabbits_before_fix\brands.txt -Raw).Trim())
python -m tests.adversarial.adversarial_runner
```

- Without `--brands`, RABBITS runs all 189 brands of the lexicon (much longer); the recorded
  before/after runs used the 60 brands in `reports/rabbits_before_fix/brands.txt`.
- Add `--limit 12` for a quick trial. `--resume` continues an interrupted run from its
  checkpoint in `reports/benchmark_runs/`. **Delete the checkpoint before re-running on changed
  code** (for example `reports/benchmark_runs/rabbits_live.jsonl`), otherwise `--resume` reuses
  the old turns.
- Reports: `reports/medbridge_aq_benchmark_report.md`, `reports/rabbits_benchmark_report.md`
  and `reports/adversarial_live_report.md` (each with a `.json` next to it); a new run
  overwrites them. The runs from before the brand-name fix are kept in
  `reports/rabbits_before_fix/` and `reports/adversarial_before_fix/`. AQ exits with code 1 when
  it misses its acceptance targets; the report is still written.
- `--mode mock` (AQ, RABBITS) runs the hermetic harness self-test without an LLM. Its report
  says so, and it is **not** evidence of model behaviour.
- macOS/Linux shortcut: `scripts/run_live_benchmarks.sh` runs the three suites resumably
  (`STEPS="aq rabbits adversarial"`, `RABBITS_BRANDS=...`). It talks to the study Ollama server
  on port 11435 by default; set `OLLAMA_URL=http://127.0.0.1:11434` to use the normal one. It
  also sets `LLM_TIMEOUT=900`.

### 9.6 Other measurements

```bash
python scripts/pii_eval.py                        # PII redaction precision/recall (no services needed)
python -m evaluation.retrieval_eval               # Recall@k, MRR, nDCG: dense vs sparse vs hybrid vs rerank (Qdrant)
python -m evaluation.datasets.schema --dataset evaluation/datasets/hallubench_v1.json   # validate HalluBench
```

`scripts/llm_sanity_check.py` runs the four LLM calls once against a model. It imports
`medbridge` from a script, so it needs the repository root on `PYTHONPATH`:

```bash
PYTHONPATH=. python scripts/llm_sanity_check.py --model qwen3.5:4b
```

```powershell
$env:PYTHONPATH = "."; python scripts\llm_sanity_check.py --model qwen3.5:4b
```

### 9.7 All post-study measurements in one go

`scripts/run_post_study.sh` (bash) reproduces every measurement recorded after the study, in
this order of steps: `rabbits_after`, `adversarial_after`, `perf_after`, `load`, `smoke`,
`screenshots`, `annotation`, `perf_table` (and, only on request, `aq_after`). It starts and
stops its own API server on port 8000 with the settings each step needs, so **stop your own
server first**; server logs go to `evaluation/results/logs/uvicorn_post_*.log`.

> **Careful:** the default steps include `annotation`, which re-exports the human annotation
> sheet and overwrites `evaluation/annotation/human_annotation_sheet.xlsx`. Once the team has
> started labelling, always pass `STEPS` without `annotation`. The `rabbits_after` step also
> deletes the RABBITS checkpoint and overwrites `reports/rabbits_benchmark_report.*`.

```bash
scripts/run_post_study.sh                                  # all default steps
STEPS="perf_after perf_table" scripts/run_post_study.sh    # only some steps
PY=.venv/bin/python scripts/run_post_study.sh              # if your venv is not .venv-mac
```

Environment overrides: `PY`, `STEPS`, `APP_OLLAMA` (default `http://127.0.0.1:11434`),
`RABBITS_BRANDS_FILE` (default `reports/rabbits_before_fix/brands.txt`) and
`EVAL_DATABASE_URL` (read from `.env` when unset). The recorded run took 1 h 45 min on the
Apple M5 (16:09 to 17:54 IST; step times in `evaluation/results/logs/post_study_20261011T160919.log`);
its screenshot step failed and was re-run separately (`post_study_shots_20261011T175711.log`).

**Windows.** The bash drivers (`run_post_study.sh`, `run_live_benchmarks.sh`,
`run_study.sh`, `run_study_extras.sh`, `study_ollama.sh`) need Git Bash or WSL, and
`run_post_study.sh` and `run_study_extras.sh` also use `pkill`/`pgrep`/`nohup`, which Git Bash
may not provide. They were written and run on macOS only. On Windows, run the steps directly in
PowerShell instead (venv active, a server started as in 4.12 where noted):

| Step | PowerShell |
|---|---|
| RABBITS and adversarial | the commands of 9.5 (no server needed) |
| Latency probe | 9.3, with a server started by `$env:WARMUP_ON_STARTUP="false"; $env:LLM_MODEL="gemma3:4b"; python -m uvicorn medbridge.main:app --host 127.0.0.1 --port 8000` and the model unloaded |
| Load test | 9.4 |
| Smoke test | `python scripts\smoke_test.py --base-url http://127.0.0.1:8000 --output reports\smoke_test_latest.json` |
| Screenshots | `python -m medbridge.web.screenshots --base-url http://127.0.0.1:8000 --out report\screenshots\raw\desktop_light --admin-user admin --admin-password <ADMIN_PASSWORD> --viewport desktop --theme light` (needs Google Chrome; see 11) |
| Annotation sheet | `python -m evaluation.human_annotation --export --split test` (already done; overwrites the sheet, so never run it after labelling has started) |
| Latency table | `python -m evaluation.perf --before reports\performance\before_gemma3-4b.json --after reports\performance\after_gemma3-4b.json` |

---

## 10. Reproducing the hallucination study

The study asks whether the Dual Answerability Gate reduces hallucinated and unsafe answers
compared with plain LLM and RAG baselines (conditions C0 to C5, plus the variants V1, V2 and
V12). The full method is in `evaluation/README.md`. Run the study **on the Mac**: the
judge model needs more memory than the 6 GB laptop GPU has.

### 10.1 Prerequisites

1. Services up, corpus indexed (`python -m medbridge.cli doctor` is green).
2. Models: `ollama pull qwen3.5:4b` and `ollama pull gemma3:12b`.
3. The evaluation database from 9.1, with `EVAL_DATABASE_URL` exported in the shell.
4. *(Recommended)* A dedicated Ollama server with parallel decoding slots, in its own
   Terminal window. It shares the model store with the normal Ollama app but listens on port
   11435, so the app's server on 11434 is untouched:

   ```bash
   scripts/study_ollama.sh
   ```

   It sets `OLLAMA_HOST=127.0.0.1:11435`, `OLLAMA_NUM_PARALLEL=3`,
   `OLLAMA_MAX_LOADED_MODELS=2`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0` and
   `OLLAMA_KEEP_ALIVE=60m` (each can be overridden from the environment). Point the runner
   and the judge at it with `--ollama-url http://127.0.0.1:11435`, and keep `--workers` no
   higher than `OLLAMA_NUM_PARALLEL` (extra requests only queue). `scripts/run_study.sh` uses
   4 workers by default, as in the recorded test runs, so either start the server with
   `OLLAMA_NUM_PARALLEL=4 scripts/study_ollama.sh` or run the script with `WORKERS=3`.

   Windows equivalent, only if you must run it there (one slot on a 6 GB GPU):

   ```powershell
   $env:OLLAMA_HOST = "127.0.0.1:11435"; $env:OLLAMA_NUM_PARALLEL = "1"; $env:OLLAMA_FLASH_ATTENTION = "1"; $env:OLLAMA_KV_CACHE_TYPE = "q8_0"; ollama serve
   ```

### 10.2 Step 0: model selection (already done; re-run only to reproduce)

```bash
python -m evaluation.model_selection --models qwen3.5:4b,gemma3:4b,qwen3:8b --split dev --ollama-url http://127.0.0.1:11435
```

It writes per-model runs, `summary.json` and `model_selection.md` to
`evaluation/results/model_selection/`, and the figure `reports/figures/model_selection.png`.
`--skip-run` only re-summarises existing runs.

### 10.3 The protocol (the order is enforced in code)

V1, V2 and V12 refuse to run on the test split without `evaluation/results/calibration.json`,
and `analyze` refuses headline numbers if more than 5 % of a condition's turns are degraded.

```bash
# 1. C5 on the DEV split (the only data used for tuning)
python -m evaluation.run --conditions C5 --split dev --model qwen3.5:4b --ollama-url http://127.0.0.1:11435 --workers 3

# 2. judge DEV (LLM judge and NLI judge on the same claims)
python -m evaluation.judge --judge both --conditions C5 --split dev --judge-model gemma3:12b --ollama-url http://127.0.0.1:11435 --workers 3

# 3. calibrate on DEV -> evaluation/results/calibration.json
python -m evaluation.calibrate

# 4. every condition on TEST (V1/V2/V12 read their thresholds from calibration.json)
python -m evaluation.run --conditions all,V12 --split test --resume --model qwen3.5:4b --ollama-url http://127.0.0.1:11435 --workers 3
python -m evaluation.run --conditions all,V12 --split test --retry-degraded --model qwen3.5:4b --ollama-url http://127.0.0.1:11435

# 5. judge TEST
python -m evaluation.judge --judge both --split test --resume --judge-model gemma3:12b --ollama-url http://127.0.0.1:11435 --workers 3

# 6. analyse TEST: statistics, tables, figures, error analysis
python -m evaluation.analyze
```

Notes:

- `all` means C0, C1, C2, C3, C4, C5, V1 and V2. `all,V12` adds V12. `C1G` (RAG with the
  production prompt) is optional: `--conditions all,C1G`.
- Every step is **resumable**. Interrupt with Ctrl+C and re-run the same command with
  `--resume`. `evaluation.run ... --limit 10` does a quick trial run.
- Identical LLM requests are memoised in `evaluation/cache/` (one cache file per set of
  model digests), so paired conditions share calls. Use `--no-cache` for a clean latency
  run.
- On Windows, type each command on one line (as above).

**One-shot script (macOS).** `scripts/run_study.sh` runs the same steps (dev run, dev judge,
calibration with `--part all`, test run, retry of degraded turns, test judge, analysis),
resumably:

```bash
scripts/run_study.sh
STEPS="analyze" scripts/run_study.sh                         # only some steps
MODEL=qwen3.5:4b JUDGE=gemma3:12b WORKERS=3 CONDITIONS=C0,C1,C2,C3,C4,C5,V1,V2,V12 scripts/run_study.sh
PY=.venv/bin/python scripts/run_study.sh                      # if your venv is not .venv-mac
```

Its defaults: `PY=.venv-mac/bin/python` (or `python` when that does not exist),
`MODEL=qwen3.5:4b`, `JUDGE=gemma3:12b`, `OLLAMA_URL=http://127.0.0.1:11435`, `WORKERS=4`,
`CONDITIONS=C0,C1,C2,C3,C4,C5,V1,V2`, `STEPS="dev_run dev_judge calibrate test_run test_retry
test_judge analyze"`, and `LLM_TIMEOUT=900` / `JUDGE_TIMEOUT=900` seconds (long, because parallel
workers share one GPU). It reads `EVAL_DATABASE_URL` from `.env` when the shell does not set it.

The follow-up conditions C1G and V12 were added by `scripts/run_study_extras.sh`: it waits until
no `run_study.sh` is running, then runs `test_run test_retry test_judge` for
`EXTRA_CONDITIONS` (default `C1G,V12`) and re-runs `analyze`.

**Windows.** The `.sh` drivers need Git Bash or WSL and were only run on macOS (in Git Bash you
would also set `PY=.venv/Scripts/python.exe`). The supported way on Windows is to type the
`python -m evaluation...` commands of the protocol above in PowerShell, after setting the
evaluation database and the long timeout for the session:

```powershell
$env:EVAL_DATABASE_URL = (Select-String -Path .env -Pattern '^EVAL_DATABASE_URL=(.+)$').Matches[0].Groups[1].Value
$env:LLM_TIMEOUT = "900"
python -m evaluation.run --conditions C1G,V12 --split test --resume --model qwen3.5:4b --ollama-url http://127.0.0.1:11435 --workers 1
```

The study's judge model does not fit the 6 GB GPU, so plan to run the study on the Mac.

### 10.4 Supporting steps

```bash
python -m evaluation.human_annotation --export --split test   # 50-item blinded sheet (xlsx + csv); already exported to evaluation/annotation/
python -m evaluation.human_annotation --score       # after the team fills the A1/A2 columns
python -m evaluation.retrieval_eval                 # retrieval metrics + figure
python -m evaluation.perf --before reports/performance/before_gemma3-4b.json --after reports/performance/after_gemma3-4b.json   # latency table + figure (9.3)
python scripts/render_docs_results.py               # refresh the result blocks in README.md from the result files
```

The sheet is already exported (`evaluation/annotation/human_annotation_sheet.xlsx`, with
instructions on its first tab; the key that links rows to conditions is
`evaluation/annotation/annotation_key.csv`, keep it away from the annotators). The labelling is
a team task: [TO BE FILLED: annotators, date, completed sheet]. Do not re-export after labelling
has started: the export writes to the same files and would overwrite the filled sheet.

### 10.5 Outputs and runtime

| Path | Content |
|---|---|
| `evaluation/results/runs/<COND>.jsonl`, `manifest_<run_id>.json` | one line per turn; the manifest records git commit, dataset hash, models and Ollama digests, hardware, seeds, start and end time |
| `evaluation/results/judgments/` | LLM and NLI judgments per condition |
| `evaluation/results/calibration.json` | dev-split thresholds |
| `evaluation/results/summary.json`, `tables/`, `error_analysis.md` | metrics, confidence intervals, McNemar / Cochran's Q tests, error analysis |
| `reports/figures/*.png` / `*.svg` | figures (300 dpi PNG + SVG) |

**Measured runtime on the Apple M5** (from the run and judge manifests): the test-split runs
of all ten conditions spanned 2026-10-10 06:28 to 15:32 UTC, about 9 hours of wall-clock
including one crash and resume (4 workers; manifests in `evaluation/results/runs/`); the
test-split LLM judge needed two resumed runs, the second alone 9 h 27 min with 8 workers
(`evaluation/results/judgments/manifest_judge_20261011T014507.json`); on the dev split the LLM
judge took 13 min and the NLI judge 7 min, and each final model-selection run took 10 to
17 minutes. The study was not run on the Windows laptop. To list the durations of all
finished runs:

```bash
python -c "import json,glob,datetime as d; [print(f, d.datetime.fromisoformat(m['finished_at'])-d.datetime.fromisoformat(m['started_at'])) for f in sorted(glob.glob('evaluation/results/**/manifest_*.json', recursive=True)) for m in [json.load(open(f))] if m.get('finished_at') and m.get('started_at')]"
```

Plan the time as items x turns x conditions x time per turn, plus judge time. Memoisation
removes repeated requests across conditions.

---

## 11. Building the report, paper and slides

The report is generated by a script, so every number, table and figure comes from the
result files. Install LibreOffice (section 3.2) for the PDF export.

```bash
python report/build_report.py              # -> report/MedBridge_Final_Report.docx (+ report/build/report_build_log.json)
python report/build_report.py --pdf        # + PDF through LibreOffice, with the TOC and lists refreshed
python report/build_report.py --strict     # exit 1 on unknown citation keys
```

Results that do not exist yet appear in the document as marked *pending* items; rebuild after
the study to fill them. `report/README.md` explains the content files and the build log.

The table of contents, list of figures and list of tables are Word fields. In Word, press
**Ctrl+A** then **F9** to update them; in LibreOffice use *Tools, Update, Update All*. The
`--pdf` option does this automatically.

Manual PDF export with LibreOffice:

```bash
# macOS
soffice --headless --convert-to pdf --outdir report report/MedBridge_Final_Report.docx
```

```powershell
# Windows
& "C:\Program Files\LibreOffice\program\soffice.exe" --headless --convert-to pdf --outdir report report\MedBridge_Final_Report.docx
```

Diagrams: each script in `report/diagrams/` regenerates its figure in `report/figures/`, for
example `python report/diagrams/architecture_pipeline.py`.

UI screenshots for the report (Playwright driving an installed **Google Chrome**; the app
must run against the real backend. The script sends many requests, so the recorded run used a
server with `RATE_LIMIT_PER_MINUTE=120` and paused 65 s between combinations. `--base-url`
defaults to the development mock server on port 8100, whose screenshots must never be used as
results, so always pass it):

```bash
python -m medbridge.web.screenshots --base-url http://127.0.0.1:8000 --out report/screenshots/raw/desktop_light --admin-user admin --admin-password <ADMIN_PASSWORD> --viewport desktop --theme light
```

`--viewport desktop|mobile` and `--theme light|dark` select one combination (without them every
combination is taken); `--only chat|dashboard` limits the scenes. The recorded set is in
`report/screenshots/raw/<viewport>_<theme>/`; the report and the README use copies of the
desktop-light shots under `report/screenshots/` (`welcome.png`, `answer_citations.png`,
`soft_ask.png`, `escalate.png`, `abstain.png`, `why_panel.png`, `bp_trend.png`,
`dashboard.png`) plus one dark-mode shot, `dark_mode.png`.

IEEE paper and presentation (same rules: every number comes from the result files):

```bash
python report/paper/build_paper.py           # -> report/paper/MedBridge_IEEE_Paper.docx (--pdf, --paper a4, --anonymous, --strict)
python report/paper/build_latex.py           # -> report/paper/latex/main.tex + references.bib + figures/ (compile on Overleaf or locally)
python report/build_presentation.py          # -> report/MedBridge_Review_Presentation.pptx (--pdf, --png DIR, --strict)
```

On Windows write the paths with `\`. `report/paper/README.md` and
`report/presentation/README.md` describe the options. Venue recommendations are in
`report/paper/VENUES.md`.

---

## 12. Demo script for the review

The full 10-minute panel script with likely viva questions is `docs/DEMO_SCRIPT.md`. This
section gives the exact inputs and what each one shows.

### 12.1 Before the panel (15 minutes ahead)

1. Plug in the charger and set Windows to *Best performance*. A laptop GPU on battery is much
   slower.
2. Close games, video calls and other GPU-heavy programs.
3. `.\scripts\run.ps1 -Open` (macOS: `scripts/run.sh --open`).
4. Wait until <http://127.0.0.1:8000/health> says `"status":"healthy"` (warm-up finished).
5. Send one throw-away message (for example "hi there!", then a short clinical question) so
   the model is loaded in VRAM, then start a **New chat**.
6. In a second tab, log in to <http://127.0.0.1:8000/dashboard>.
7. Optional: `python -m medbridge.cli smoke --output reports\smoke_test_demo.json` for a
   last end-to-end check.

### 12.2 Demo sequence

The emergency, small-talk, PII and template steps are deterministic. Steps that go through
the gates depend on the LLM, and a model can route a borderline question differently (for
example GENERALIZE instead of ANSWER). Every alternative listed is still a safe action;
explain why when it happens. Start a **New chat** where the table says so, because the
patient context carries over within a chat.

| # | What to show | Type exactly | Expected | Point out |
|---|---|---|---|---|
| 1 | Welcome screen | (no input) | starter cards, *Local-only* badge, disclaimer with 112 | the privacy notice on the first visit; nothing leaves the laptop |
| 2 | Small talk | `hi there!` | deterministic greeting, no clinical claims | small talk is answered from a template |
| 3 | ESCALATE (fast-path) | `My BP is 210/125 and I have crushing chest pain right now.` | red emergency card (112 / 108), almost instant | the deterministic emergency classifier runs before any LLM call |
| 4 | ESCALATE, colloquial Hinglish | `seene mein bahut dard ho raha hai` | emergency card | colloquial and Hinglish red-flag phrases |
| 5 | Negation (no false alarm), new chat | `I have no chest pain, but my BP was 150/95 this morning.` | **not** escalated; goes through the gates (ANSWER, GENERALIZE or SOFT-ASK) | negation handling ("no chest pain") |
| 6 | SOFT-ASK, new chat | `Should I take a higher dose of my BP medicine?` | SOFT-ASK: asks for the medicine, dose and readings (GENERALIZE is also acceptable) | Gate 1 asks before it retrieves |
| 7 | Follow-up gets answered | `I take amlodipine 5 mg once a day and my readings this week were around 150/95. I am 58.` | ANSWER (or GENERALIZE) with `[n]` citations; Gate 3 may shorten it or remove its citations, which turns it into GENERALIZE | the context card now shows the BP reading and amlodipine (event-sourced state); the Why panel shows what Gate 3 removed |
| 8 | Citations | click a `[1]` chip | source drawer: guideline, organisation, year, section, page, excerpt | grounding is visible and checkable |
| 9 | Why this answer? | switch the top bar to **Clinician**, open "Why this answer?" on the last reply | Gate 1 decision and rationale, top-5 retrieval with scores, Gate 2 decision, stage timeline | the Dual Answerability Gate made visible |
| 10 | GENERALIZE, new chat | `What dietary changes help lower blood pressure in general?` | GENERALIZE (or ANSWER) with citations | population-level guidance, no personal advice |
| 11 | ABSTAIN, out of domain | `What caused the French Revolution?` | ABSTAIN template | ABSTAIN bypasses generation: no LLM text, no hallucination possible |
| 12 | Fabricated drug, new chat | `Is Cardiozepam 20 mg safe to take with my Telma 40?` | ABSTAIN or GENERALIZE; it must **not** give a dose for "Cardiozepam" | "Cardiozepam" does not exist; Telma is an Indian brand of telmisartan, and the drug normaliser maps brand names to generics |
| 13 | PII redaction, new chat | `My name is John Smith and my phone number is +91 98765 43210. Is 128/82 a normal blood pressure?` | an answer plus the chip **PII redacted (2)** | name and phone are removed before any LLM call or storage; the BP value is kept |
| 14 | Loop-breaker, new chat | `Is my medicine dose okay?`, then `not sure`, then `I'd rather not say` | SOFT-ASK, SOFT-ASK, then GENERALIZE | after two clarification questions the system stops asking (threshold 2) |
| 15 | BP trend | open **My health context** (pulse icon), **Add reading**: 128/82, 136/86, 142/91, 151/94 | chart with ACC/AHA category bands | manual readings use the same event pipeline |
| 16 | Clinician summary | in the context panel: **Clinician summary** | downloads a cited Markdown summary | hand-over to a doctor |
| 17 | Message actions | copy, regenerate, thumbs up/down with a comment | feedback stored | feedback appears on the dashboard |
| 18 | Dashboard | the dashboard tab, refresh | action distribution, gate statistics, latency charts, feedback, recent audit entries (redacted) | audit rows are append-only |
| 19 | Right to erasure | context panel: **Clear my data**, confirm | the conversation is erased | crypto-shredding: the session key is destroyed; audit rows stay but are unreadable |

Backup inputs if a step does not route as hoped:

- ANSWER with citations: `What lifestyle changes help lower blood pressure?` (the ANSWER scene
  of the report screenshots; its citations survived Gate 3 there). A personal reading such as
  `My blood pressure was 145/92 this morning. Is that high?` is **not** a safe ANSWER demo:
  since Gate 3 is on, it ended as a GENERALIZE with one remaining sentence and no citations
  (ADL-052). It is a good live example of Gate 3 at work, though.
- SOFT-ASK: `Should I take a higher dose of my BP medicine?` (the SOFT-ASK scene of the report
  screenshots)
- ESCALATE: `my face feels droopy on one side and i can't lift my left arm`

### 12.3 If something goes wrong during the demo

| Symptom | Quick action |
|---|---|
| Every answer is a generic fallback | Ollama stopped or the model is unloaded: check `ollama ps`, start the Ollama app, wait about 10 seconds |
| First answer takes very long | warm-up or model loading; wait, and next time do step 12.1-5 |
| `429 Too Many Requests` | the rate limit (30 messages per minute); wait a minute |
| Page looks broken after an update | hard refresh with **Ctrl+F5** |
| Session error after a restart | click **New chat** |

---

## 13. Troubleshooting

Start every investigation with:

```bash
python -m medbridge.cli doctor
```

It names the failing component and prints the fix. The entries below cover the common cases
in more detail. Entries marked (Windows) or (macOS) apply to that system only.

### Port already in use (5433, 6333 or 8000)

Symptoms: `docker compose up -d` fails with *"port is already allocated"* or *"bind: address
already in use"*, or uvicorn fails with *"[Errno 10048]"* (Windows) or *"[Errno 48]"* (macOS).

Find the program that uses the port:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object OwningProcess
Get-Process -Id <OwningProcess>
```

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

- **8000:** often MedBridge itself is already running (open <http://127.0.0.1:8000>).
  Otherwise stop the other program, or start MedBridge on another port:
  `python -m medbridge.cli up --port 8001` and open `http://127.0.0.1:8001`.
- **5433:** another PostgreSQL, or an old MedBridge container from another folder
  (`docker ps`). Stop it. The host port is fixed in `docker-compose.yml`; if you really must
  change it, change it there **and** in `DATABASE_URL`.
- **6333/6334:** another Qdrant container. `docker ps`, then `docker stop <name>`.
- **11434:** this is Ollama's own port. Only one Ollama server can use it; a second
  `ollama serve` fails while the app is running. That is fine.

### Docker is not running

Symptoms: *"error during connect ... dockerDesktopLinuxEngine"* (Windows) or *"Cannot connect
to the Docker daemon at unix:///.../docker.sock"* (macOS).

Start Docker Desktop and wait for *Engine running* (`run.ps1` and `run.sh` try to start it for
you). Check with `docker info`.

### WSL 2 problems (Windows)

Symptoms: Docker Desktop says *"WSL 2 installation is incomplete"*, *"WSL update required"*
or does not start.

1. Administrator PowerShell: `wsl --update`, then `wsl --shutdown`, then restart Docker
   Desktop.
2. First-time setup: `wsl --install`, then restart Windows.
3. Make sure virtualisation is enabled in the BIOS/UEFI (Task Manager, Performance, CPU).
4. In Docker Desktop, Settings, General: *Use the WSL 2 based engine* must be ticked.

### Ollama out of memory or very slow (6 GB GPU)

Symptoms: *"model requires more system memory"*, *"CUDA error: out of memory"*, very slow
replies, or `ollama ps` showing a CPU/GPU split instead of `100% GPU`. Try these in order:

1. Use the chosen small model: `LLM_MODEL=qwen3.5:4b` in `.env` (not `qwen3:8b` or a 12B
   model), then restart the API.
2. Apply the 6 GB settings from 4.8 (`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`,
   `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`), then quit and restart Ollama.
3. Close other GPU users (games, video calls, browser tabs with heavy graphics) and plug in
   the charger.
4. Keep PyTorch (the reranker) off the GPU so all VRAM goes to Ollama. Start the API with
   CUDA hidden:

   ```powershell
   $env:CUDA_VISIBLE_DEVICES = "-1"; python -m medbridge.cli up
   ```

   The reranker and the Gate 3 model then run on the CPU (slower, but it frees VRAM).
5. Reduce the context window: `LLM_NUM_CTX=6144` in `.env`. Do not go lower: the prompts need
   several thousand tokens, and Ollama shortens over-long prompts silently.
6. Use the fallback model `gemma3:4b` (`ollama pull gemma3:4b`, `LLM_MODEL=gemma3:4b`).
   Smaller quantisations or `qwen3.5:2b` would also fit, but they were **not** evaluated in
   the model-selection experiment. Use them only as a last resort and say so if you present
   with them.

### Model not found

Symptoms: the doctor reports *"model(s) not pulled"*, the log shows *"model 'qwen3.5:4b' not
found"*, or `/health` is `degraded` with a model listed as `false`.

```bash
ollama pull qwen3.5:4b
ollama list           # the name must match LLM_MODEL in .env exactly, including the tag after ":"
```

If `ollama pull` says the model needs a newer version of Ollama, update Ollama and try again.

### Qdrant is empty (answers have no citations)

Symptoms: `/health` shows `qdrant_points: 0` or `null`, the doctor says the collection is
empty or missing, and most answers are ABSTAIN or GENERALIZE without citations.

```bash
python scripts/download_corpus.py --verify-only        # are the files there?
python -m medbridge.ingestion --recreate --no-manifest-update
```

`docker compose down -v` (or `scripts/reset.*`) deletes the index; the launcher re-indexes
automatically on the next `up`.

### CUDA not used by PyTorch (Windows)

Check: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

- The version ends in `+cpu`: the CPU build was installed (usually because
  `requirements.txt` was installed first). Fix:

  ```powershell
  python -m pip uninstall -y torch
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
  ```

- `+cu126` but `False`: the driver is too old for that build. Compare with the CUDA Version
  in `nvidia-smi`, update the driver or use `cu124`/`cu121` (table in 4.6).
- `nvidia-smi` is not found: install the NVIDIA driver.

Remember that the LLM runs in Ollama, which uses the GPU independently of PyTorch. A CPU
PyTorch only makes the reranker slower.

### MPS issues (macOS)

- `torch.backends.mps.is_available()` is `False`: you are probably running an x86 Python
  under Rosetta. Check with `python -c "import platform; print(platform.machine())"`; it must
  print `arm64`. Install an arm64 Python (Homebrew on Apple Silicon or the python.org
  universal2 installer) and recreate `.venv`.
- An error about an operator that is not implemented for MPS: set
  `export PYTORCH_ENABLE_MPS_FALLBACK=1` before starting the API. For the study's NLI judge
  and the calibration you can force the CPU with `--nli-device cpu` or `--verifier-device cpu`.

### PowerShell execution policy (Windows)

Symptom: *"... cannot be loaded because running scripts is disabled on this system"*.

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
Get-ChildItem -Recurse -Filter *.ps1 | Unblock-File       # only if the project came as a ZIP
```

Or run a script once with `powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1`.

### Long path errors (Windows)

Symptoms: `pip install` or `git clone` fails with *"No such file or directory"* or
*"Filename too long"* although the file exists. Move the project to a short path such as
`C:\dev\MedBridge` and enable long paths (4.3), then recreate `.venv`.

### Antivirus slowing things down (Windows)

Real-time scanning can make `pip install`, the first model load and Docker very slow. If your
institution allows it, add an exclusion for the project folder and the Ollama model folder
(`%USERPROFILE%\.ollama`) in Windows Security, *Virus & threat protection settings*,
*Exclusions*. Do not disable the antivirus.

### The first request is slow

At start-up the app warms up the embedders, the reranker, the Gate 3 NLI model and the Ollama
model in the background. Until warm-up finishes `/health` reports `degraded`, and a message sent
in that window waits for the models to load. Wait for `healthy` before the demo. The very first
run on a machine is slower still, because the embedding models, the reranker and the Gate 3
model are downloaded. The out-of-domain ABSTAIN path takes several seconds even when warm,
because Gate 2's prompt is long (see the latency tables in the README).

### Answers are very short, start mid-sentence or come back as GENERALIZE without citations

That is Gate 3 (on by default): it removes every sentence its NLI model judges unsupported
by the cited passages, and an ANSWER left without a valid citation becomes GENERALIZE. When
every claim is unsupported the draft is withdrawn (regenerated as GENERALIZE, then ABSTAIN).
Open "Why this answer?" in Clinician mode to see what Gate 3 did. This is the measured
trade-off of the study (less hallucination, shorter answers; ADL-052). To compare with the
evaluated dual-gate configuration, set `GROUNDING_VERIFIER_ENABLED=false` in `.env` and restart
the API (the code still contains the post-study brand-name fix, ADL-059, so this is close to,
not identical with, the evaluated C5).

### Model downloads on the first run (Hugging Face)

The first ingestion downloads `BAAI/bge-small-en-v1.5` and `Qdrant/bm25` (fastembed). The
first API start downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker) and, because
Gate 3 is on by default, `cross-encoder/nli-MiniLM2-L6-H768` from Hugging Face (weight files
of about 91 MB and 328 MB, see 3.3). The study also downloads `cross-encoder/nli-deberta-v3-base`.

- Internet access is needed once. Behind a proxy, set `HTTPS_PROXY` before starting.
- Caches: Hugging Face models in `~/.cache/huggingface/hub` (Windows:
  `%USERPROFILE%\.cache\huggingface\hub`, for example
  `C:\Users\<you>\.cache\huggingface\hub`). To keep them elsewhere (for example on a drive
  with more space), set `HF_HOME` before the first start:
  `setx HF_HOME "D:\hf-cache"` (new PowerShell window afterwards). fastembed uses the system
  temporary folder unless `FASTEMBED_CACHE_PATH` is set. Windows may clean the temporary
  folder, which causes a re-download; set a permanent folder with
  `setx FASTEMBED_CACHE_PATH "%USERPROFILE%\.cache\fastembed"`.
- For an offline demo, run everything once with internet the day before (until `/health`
  shows `nli_verifier: true`), then set `HF_HUB_OFFLINE=1` in the shell before starting the API
  (`$env:HF_HUB_OFFLINE = "1"` in PowerShell).

### Firewall prompts (Windows)

Windows Defender Firewall may ask whether Python, Docker Desktop or Ollama may communicate on
networks. MedBridge only uses the loopback address (`127.0.0.1`), which the firewall does not
block, so you can allow **private networks only** or cancel the prompt.

### `password authentication failed for user "medbridge_app"`

The PostgreSQL volume was created with a different password than the one now in `.env`
(for example after `python scripts/setup_env.py --force`, or after copying `.env` from another
machine). Either:

- restore the old `POSTGRES_PASSWORD` and `DATABASE_URL` values in `.env`; or
- reset the database volume, which **deletes all stored conversations**:
  `docker compose down -v`, then `python -m medbridge.cli up`.

### `ModuleNotFoundError: No module named 'medbridge'` (or another package)

- Run commands from the repository root, with the venv active (`(.venv)` in the prompt).
- `No module named 'fastapi'` (or similar): `python -m pip install -r requirements.txt`.
- For `scripts/llm_sanity_check.py`, set `PYTHONPATH` to the repository root (9.6).

### `UnicodeEncodeError` in the console (Windows)

Use Windows Terminal, or set UTF-8 mode for the session: `$env:PYTHONUTF8 = "1"`.

### Streamlit says it cannot reach the backend

The API must be running on port 8000 first. If it runs elsewhere, set `$env:MEDBRIDGE_API_URL`
(Windows) or `export MEDBRIDGE_API_URL=...` (macOS) before `streamlit run`.

### `401` / `403` / "session expired" in the UI

Session tokens are signed with `SESSION_TOKEN_SECRET` and expire after 7 days. After a secret
change or a database reset, old chats in the sidebar cannot be opened. Start a **New chat**.

### `429 Too Many Requests`

`RATE_LIMIT_PER_MINUTE` (default 30 messages per minute per client) was exceeded. Wait a
minute. For load tests use a separate server with a higher limit (9.4).

---

## 14. Stopping, resetting and uninstalling

**Stop:**

| What | How | Data kept? |
|---|---|---|
| API | Ctrl+C in its window | yes |
| Containers | `docker compose stop` (start again with `docker compose start` or `up -d`) | yes |
| Containers, removed | `docker compose down` | yes (volumes stay) |
| Ollama | tray or menu-bar icon, *Quit Ollama* | yes |

**Reset (delete local data).** The reset scripts ask for confirmation (type `yes`) unless you
pass `-Yes` / `--yes`, and support a dry run:

```powershell
.\scripts\reset.ps1 -DryRun              # show what would be deleted
.\scripts\reset.ps1                      # containers + volumes: ALL conversations, audit logs, the eval/test databases and the Qdrant index
.\scripts\reset.ps1 -Venv -Models -EnvFile   # also the venv, the Ollama models and .env
.\scripts\reset.ps1 -All -Yes            # everything, no question
```

```bash
scripts/reset.sh --dry-run
scripts/reset.sh
scripts/reset.sh --venv --models --env
scripts/reset.sh --all --yes
```

The reset never touches the source code, the downloaded corpus (`data/guidelines`),
`evaluation/results` or `reports/`. Afterwards run the setup script again, then the run script
(which re-indexes the corpus).

The same steps by hand:

```bash
docker compose down -v           # deletes the PostgreSQL and Qdrant volumes
ollama rm qwen3.5:4b             # and gemma3:4b / gemma3:12b if pulled
```

Delete the virtual environment with `Remove-Item -Recurse -Force .venv` (Windows, after
`deactivate`) or `rm -rf .venv` (macOS).

**Uninstall** Docker Desktop, Ollama and Python through *Settings, Apps* (Windows) or by moving
the apps to the Bin (macOS). Optional caches to delete: `%USERPROFILE%\.ollama` or `~/.ollama`
(models), `%USERPROFILE%\.cache\huggingface` or `~/.cache/huggingface` (Hugging Face models).

---

## 15. Command cheat sheet

| Task | Windows (PowerShell) | macOS (Terminal) |
|---|---|---|
| Activate the venv | `.\.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |
| First-time setup | `.\scripts\setup.ps1 -PullModels` | `scripts/setup.sh --pull-models` |
| Start everything | `.\scripts\run.ps1 -Open` | `scripts/run.sh --open` |
| Check the machine | `python -m medbridge.cli doctor` | `python -m medbridge.cli doctor` |
| Preview without changes | `.\scripts\run.ps1 -DryRun` | `scripts/run.sh --dry-run` |
| Services only | `docker compose up -d` | `docker compose up -d` |
| Migrations | `python -m alembic upgrade head` | `python -m alembic upgrade head` |
| Download the corpus | `python scripts\download_corpus.py` | `python scripts/download_corpus.py` |
| Re-index | `python -m medbridge.ingestion --recreate` | `python -m medbridge.ingestion --recreate` |
| API only | `python -m uvicorn medbridge.main:app --host 127.0.0.1 --port 8000` | same |
| Health | `Invoke-RestMethod http://127.0.0.1:8000/health` | `curl -s http://127.0.0.1:8000/health` |
| Smoke test | `python -m medbridge.cli smoke` | `python -m medbridge.cli smoke` |
| Tests | `python -m pytest` | `python -m pytest` |
| Test summary for the docs | `python scripts\test_summary.py` | `python scripts/test_summary.py` |
| Refresh result numbers in the docs | `python scripts\render_docs_results.py` | `python scripts/render_docs_results.py` |
| Live RABBITS (60 brands) | see 9.5 | see 9.5 |
| All post-study measurements | the PowerShell table in 9.7 | `scripts/run_post_study.sh` |
| Streamlit (legacy) | `python -m streamlit run medbridge\frontend\app.py` | `python -m streamlit run medbridge/frontend/app.py` |
| Build the report | `python report\build_report.py --pdf` | `python report/build_report.py --pdf` |
| Stop the containers | `docker compose stop` | `docker compose stop` |
| Reset | `.\scripts\reset.ps1` | `scripts/reset.sh` |
