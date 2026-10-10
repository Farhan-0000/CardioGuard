<#
.SYNOPSIS
  Start MedBridge on Windows with one command.

.DESCRIPTION
  Starts Docker Desktop and the Ollama app when they are installed but not running, then
  runs "python -m medbridge.cli up": docker compose up -d, wait until healthy,
  alembic upgrade head, ingestion when the Qdrant collection is empty, and the API on
  http://127.0.0.1:8000 (Ctrl+C stops the API; the containers keep running).

.EXAMPLE
  .\scripts\run.ps1
.EXAMPLE
  .\scripts\run.ps1 -Open
.EXAMPLE
  .\scripts\run.ps1 -DryRun
.EXAMPLE
  .\scripts\run.ps1 -Port 8001 -NoIngest
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$Open,
    [switch]$NoIngest,
    [switch]$NoServer,
    [switch]$SkipDocker,
    [int]$Port = 0,
    [string]$BindHost = "",
    [int]$Wait = 0,
    [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Test-DockerEngine {
    # True when "docker info" succeeds. Native stderr must not abort PowerShell 5.1.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & docker info *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $previous
    }
}

$VenvPy = Join-Path $RepoRoot (Join-Path $VenvDir "Scripts\python.exe")
if (-not (Test-Path $VenvPy)) {
    Write-Host "ERROR: no virtual environment at $VenvDir. Run .\scripts\setup.ps1 first." -ForegroundColor Red
    exit 1
}

if (-not $DryRun -and -not $SkipDocker) {
    if ((Get-Command docker -ErrorAction SilentlyContinue) -and -not (Test-DockerEngine)) {
        $dockerDesktop = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
        if (Test-Path $dockerDesktop) {
            Write-Host "==> Starting Docker Desktop (waiting up to 120 s for the engine)"
            Start-Process $dockerDesktop
            for ($i = 0; $i -lt 60; $i++) {
                Start-Sleep -Seconds 2
                if (Test-DockerEngine) { break }
            }
        }
    }
}

if (-not $DryRun) {
    # Ollama: start the tray app when no Ollama process is running.
    if (-not (Get-Process -Name "ollama", "ollama app" -ErrorAction SilentlyContinue)) {
        $ollamaApp = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
        if (Test-Path $ollamaApp) {
            Write-Host "==> Starting the Ollama app"
            Start-Process $ollamaApp
            Start-Sleep -Seconds 3
        }
    }
}

$cliArgs = @("-m", "medbridge.cli")
if ($DryRun) { $cliArgs += "--dry-run" }
$cliArgs += "up"
if ($Open) { $cliArgs += "--open" }
if ($NoIngest) { $cliArgs += "--no-ingest" }
if ($NoServer) { $cliArgs += "--no-server" }
if ($SkipDocker) { $cliArgs += "--skip-docker" }
if ($Port -gt 0) { $cliArgs += @("--port", "$Port") }
if ($BindHost -ne "") { $cliArgs += @("--host", $BindHost) }
if ($Wait -gt 0) { $cliArgs += @("--wait", "$Wait") }

$ErrorActionPreference = "Continue"
& $VenvPy @cliArgs
exit $LASTEXITCODE
