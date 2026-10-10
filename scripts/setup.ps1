<#
.SYNOPSIS
  MedBridge one-time setup for Windows 11 (PowerShell 5.1 or 7).

.DESCRIPTION
  Creates the virtual environment, installs PyTorch (CUDA build when an NVIDIA GPU is
  present), installs requirements.txt, creates .env with fresh secrets, optionally pulls
  the Ollama model(s) and runs "python -m medbridge.cli doctor".
  Safe to re-run: an existing venv is reused and existing .env values are kept.

.EXAMPLE
  .\scripts\setup.ps1
.EXAMPLE
  .\scripts\setup.ps1 -PullModels
.EXAMPLE
  .\scripts\setup.ps1 -Torch cu124        # older NVIDIA driver (CUDA 12.4)
.EXAMPLE
  .\scripts\setup.ps1 -Torch cpu          # no GPU / skip the CUDA download

.NOTES
  If PowerShell refuses to run scripts, run once:
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  or start this script with:
    powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
#>
[CmdletBinding()]
param(
    # Python launcher command, e.g. "py -3.11" or a full path to python.exe
    [string]$Python = "",
    [string]$VenvDir = ".venv",
    # auto = cu126 when nvidia-smi exists, else cpu (cu126 = same torch version as the macOS setup)
    [ValidateSet("auto", "cpu", "cu121", "cu124", "cu126", "cu128")]
    [string]$Torch = "auto",
    [switch]$PullModels,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Native {
    # Runs a native program and stops on a non-zero exit code. Native stderr output
    # (pip/ollama progress and warnings) must not abort Windows PowerShell 5.1, so the
    # error preference is relaxed while the program runs.
    param([string]$Exe, [string[]]$Arguments)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @Arguments
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) {
        throw "Command failed (exit $code): $Exe $($Arguments -join ' ')"
    }
}

function Test-PythonVersion([string]$Exe, [string[]]$PrefixArgs) {
    $code = "import sys; sys.exit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) else 1)"
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @PrefixArgs -c $code *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $previous
    }
}

# ---------------------------------------------------------------------------
# 1. Find Python 3.11 / 3.12
# ---------------------------------------------------------------------------
$PyExe = $null
$PyPrefix = @()
if ($Python -ne "") {
    $parts = $Python -split " "
    $PyExe = $parts[0]
    if ($parts.Length -gt 1) { $PyPrefix = @($parts[1..($parts.Length - 1)]) }
} else {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($ver in @("-3.11", "-3.12")) {
            if (Test-PythonVersion "py" @($ver)) { $PyExe = "py"; $PyPrefix = @($ver); break }
        }
    }
    if (-not $PyExe -and (Get-Command python -ErrorAction SilentlyContinue)) {
        if (Test-PythonVersion "python" @()) { $PyExe = "python"; $PyPrefix = @() }
    }
}
if (-not $PyExe) {
    throw "Python 3.11 not found. Install it from https://www.python.org/downloads/windows/ (tick 'Add python.exe to PATH'), open a NEW PowerShell window and re-run."
}
if (-not (Test-PythonVersion $PyExe $PyPrefix)) {
    throw "'$PyExe $($PyPrefix -join ' ')' is not Python 3.11/3.12."
}

$VenvPy = Join-Path $RepoRoot (Join-Path $VenvDir "Scripts\python.exe")

if (-not $SkipInstall) {
    # -----------------------------------------------------------------------
    # 2. Virtual environment
    # -----------------------------------------------------------------------
    $VenvPath = Join-Path $RepoRoot $VenvDir
    if ((Test-Path $VenvPath) -and -not (Test-Path $VenvPy)) {
        throw "$VenvDir exists but has no Scripts\python.exe (probably a macOS venv copied with the project). Delete the folder or use -VenvDir .venv-win."
    }
    if (-not (Test-Path $VenvPy)) {
        Write-Step "Creating the virtual environment $VenvDir"
        Invoke-Native $PyExe ($PyPrefix + @("-m", "venv", $VenvDir))
    } else {
        Write-Step "Reusing the virtual environment $VenvDir"
    }
    Invoke-Native $VenvPy @("--version")

    Write-Step "Upgrading pip"
    Invoke-Native $VenvPy @("-m", "pip", "install", "--upgrade", "pip")

    # -----------------------------------------------------------------------
    # 3. PyTorch (CUDA build first, so requirements.txt keeps it)
    # -----------------------------------------------------------------------
    if ($Torch -eq "auto") {
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $Torch = "cu126" } else { $Torch = "cpu" }
    }
    if ($Torch -ne "cpu") {
        Write-Step "Installing PyTorch with CUDA ($Torch) - a large download"
        Invoke-Native $VenvPy @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/$Torch")
    } else {
        Write-Step "PyTorch: CPU build (installed with requirements.txt)"
    }

    # -----------------------------------------------------------------------
    # 4. Requirements
    # -----------------------------------------------------------------------
    Write-Step "Installing the Python packages (requirements.txt)"
    Invoke-Native $VenvPy @("-m", "pip", "install", "-r", "requirements.txt")

    Write-Step "Checking PyTorch"
    Invoke-Native $VenvPy @("-c", "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')")
}

if (-not (Test-Path $VenvPy)) {
    throw "No virtual environment at $VenvDir. Run .\scripts\setup.ps1 without -SkipInstall."
}

# ---------------------------------------------------------------------------
# 5. .env with generated secrets
# ---------------------------------------------------------------------------
Write-Step "Creating/completing .env (secrets are generated, never printed)"
Invoke-Native $VenvPy @("scripts\setup_env.py")

# ---------------------------------------------------------------------------
# 6. Optional: Ollama models
# ---------------------------------------------------------------------------
if ($PullModels) {
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        $models = @()
        foreach ($line in Get-Content ".env") {
            if ($line -match '^LLM_MODEL(_GATES|_GENERATOR)?=(.+)$') {
                $value = $Matches[2].Trim().Trim('"')
                if ($value -ne "") { $models += $value }
            }
        }
        if ($models.Count -eq 0) { $models = @("qwen3.5:4b") }
        foreach ($model in ($models | Select-Object -Unique)) {
            Write-Step "ollama pull $model"
            Invoke-Native "ollama" @("pull", $model)
        }
    } else {
        Write-Warning "'ollama' not found. Install Ollama for Windows from https://ollama.com/download, then run: ollama pull qwen3.5:4b"
    }
}

# ---------------------------------------------------------------------------
# 7. Doctor
# ---------------------------------------------------------------------------
Write-Step "Checking the machine (python -m medbridge.cli doctor)"
& $VenvPy -m medbridge.cli doctor

Write-Host ""
Write-Host "Setup finished. Next steps (see HOW_TO_RUN.md):"
Write-Host "  1. $VenvDir\Scripts\Activate.ps1"
Write-Host "  2. ollama pull qwen3.5:4b                    (if not done: .\scripts\setup.ps1 -PullModels)"
Write-Host "  3. python scripts\download_corpus.py         (+ manual downloads listed in data\README.md)"
Write-Host "  4. .\scripts\run.ps1                         (Docker services, migrations, ingestion, API)"
Write-Host "     then open http://127.0.0.1:8000"
