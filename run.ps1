
$ErrorActionPreference = "Stop"

$Python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Error "Python not found at $Python"
    exit 1
}

New-Item -ItemType Directory -Force -Path models, results | Out-Null

Write-Host "`n========================================"
Write-Host "1. Training DDQN"
Write-Host "========================================"

& $Python train.py --episodes 1000 --seed 0 `
    --out models/ddqn.pt --log results/train_log.csv

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n========================================"
Write-Host "2. Training DDQN without latency"
Write-Host "========================================"

& $Python train.py --episodes 1000 --seed 0 --ablate-latency `
    --tag "DDQN no-latency (ablation)" `
    --out models/ddqn_nolat.pt `
    --log results/train_log_nolat.csv

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n========================================"
Write-Host "3. Evaluating both models"
Write-Host "========================================"

& $Python eval.py --ckpt models/ddqn.pt --ckpt models/ddqn_nolat.pt --episodes 20

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n========================================"
Write-Host "4. Plotting results"
Write-Host "========================================"

& $Python plot_results.py

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nAll commands completed successfully!" -ForegroundColor Green
