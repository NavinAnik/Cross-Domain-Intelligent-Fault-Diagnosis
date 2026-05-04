# Run eval from repo root so "experiments" imports work (cd src + venv python).
# Paste only the command line, not the PS prompt.
#   .\scripts\run_eval_spectrogram_checkpoint.ps1
#   .\scripts\run_eval_spectrogram_checkpoint.ps1 -RunId my_run -DatasetPath ..\data\cwru_spectrograms_png

param(
    [string] $RunId = "cwru_finetune_honestval_v2",
    [string] $DatasetPath = "..\data\cwru_spectrograms_png"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Src = Join-Path $RepoRoot "src"
$Py = Join-Path $RepoRoot "venv\Scripts\python.exe"

if (-not (Test-Path $Py)) {
    Write-Error "Python not found at $Py. Use repo venv or edit path."
}
if (-not (Test-Path $Src)) {
    Write-Error "Missing folder: $Src"
}

Push-Location $Src
try {
    & $Py -m experiments.eval_spectrogram_checkpoint --run_id $RunId --dataset_path $DatasetPath
} finally {
    Pop-Location
}
