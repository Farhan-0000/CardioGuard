<#
.SYNOPSIS
  Reset / clean up a local MedBridge installation (Windows).

.DESCRIPTION
  Always: stops the containers and DELETES the database and Qdrant volumes
  (docker compose down -v): all conversations, audit logs, the evaluation and test
  databases and the vector index are lost.
  Optional: -Venv deletes the virtual environment, -Models removes the Ollama models,
  -EnvFile deletes .env (new secrets on the next setup), -All does all three.
  Never touched: source code, data\guidelines (downloaded corpus), evaluation\results,
  reports\. After a reset run .\scripts\setup.ps1, then .\scripts\run.ps1.

.EXAMPLE
  .\scripts\reset.ps1
.EXAMPLE
  .\scripts\reset.ps1 -All -Yes
.EXAMPLE
  .\scripts\reset.ps1 -Models -DryRun
#>
[CmdletBinding()]
param(
    [switch]$Venv,
    [switch]$Models,
    [switch]$EnvFile,
    [switch]$All,
    [switch]$Yes,
    [switch]$DryRun,
    [string]$VenvDir = ".venv",
    [string[]]$ModelNames = @("qwen3.5:4b", "gemma3:4b", "gemma3:12b")
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ($All) { $Venv = $true; $Models = $true; $EnvFile = $true }

function Invoke-NativeQuiet {
    # Runs a native program; a non-zero exit code is reported as a warning, never fatal.
    param([string]$Exe, [string[]]$Arguments)
    if ($DryRun) {
        Write-Host "[dry-run] $Exe $($Arguments -join ' ')"
        return
    }
    Write-Host "> $Exe $($Arguments -join ' ')"
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @Arguments
        if ($LASTEXITCODE -ne 0) { Write-Warning "'$Exe $($Arguments -join ' ')' exited with $LASTEXITCODE" }
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Remove-PathStep([string]$Path, [string]$Label) {
    if (-not (Test-Path $Path)) { return }
    if ($DryRun) {
        Write-Host "[dry-run] delete $Label"
    } else {
        Write-Host "> delete $Label"
        Remove-Item -Recurse -Force $Path
    }
}

Write-Host "This will permanently delete:"
Write-Host "  - the MedBridge containers and their volumes (ALL conversations, audit logs and the Qdrant index)"
if ($Venv) { Write-Host "  - the virtual environment $VenvDir (deactivate it first if it is active)" }
if ($Models) { Write-Host "  - the Ollama models: $($ModelNames -join ', ')" }
if ($EnvFile) { Write-Host "  - .env (all generated secrets)" }

if (-not $Yes -and -not $DryRun) {
    $answer = Read-Host 'Type "yes" to continue'
    if ($answer -ne "yes") {
        Write-Host "Cancelled."
        exit 1
    }
}

if (Get-Command docker -ErrorAction SilentlyContinue) {
    Invoke-NativeQuiet "docker" @("compose", "down", "-v", "--remove-orphans")
} else {
    Write-Host "docker not found - skipping the containers."
}

# Incremental-ingestion bookkeeping belongs to the deleted Qdrant volume.
Remove-PathStep (Join-Path $RepoRoot "data\.ingest_state.json") "data\.ingest_state.json"

if ($Venv) {
    Remove-PathStep (Join-Path $RepoRoot $VenvDir) $VenvDir
}

if ($Models) {
    if (Get-Command ollama -ErrorAction SilentlyContinue) {
        foreach ($model in $ModelNames) {
            Invoke-NativeQuiet "ollama" @("rm", $model)
        }
    } else {
        Write-Host "ollama not found - skipping the models."
    }
}

if ($EnvFile) {
    Remove-PathStep (Join-Path $RepoRoot ".env") ".env"
}

Write-Host "Reset complete. Next: .\scripts\setup.ps1, then .\scripts\run.ps1"
