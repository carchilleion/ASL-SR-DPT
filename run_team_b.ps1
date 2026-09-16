# ==============================================================================
# ASL-SR-DPT Pilot Benchmark - PowerShell Launcher for TEAM B
# Strictly locks thread allocation for reproducible single-threaded timing.
# ==============================================================================

$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "Launching ASL-SR-DPT Pilot Benchmark Application for TEAM B" -ForegroundColor Green
Write-Host "Thread settings: OMP=1, OPENBLAS=1, MKL=1, NUMEXPR=1" -ForegroundColor Yellow
Write-Host "Workload: 5 BSD68 Images (test006-test010) x 3 Noise x 5 Trials x 3 Solvers = 225" -ForegroundColor Yellow
Write-Host "==============================================================================" -ForegroundColor Cyan

python -m streamlit run app.py -- --team "Team B"
