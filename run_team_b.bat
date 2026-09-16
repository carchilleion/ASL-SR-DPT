@echo off
REM ==============================================================================
REM ASL-SR-DPT Pilot Benchmark - Launcher for TEAM B
REM Strictly locks thread allocation for reproducible single-threaded timing.
REM ==============================================================================

set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1

echo ==============================================================================
echo Launching ASL-SR-DPT Pilot Benchmark Application for TEAM B
echo Thread settings: OMP=1, OPENBLAS=1, MKL=1, NUMEXPR=1
echo Workload: 5 BSD68 Images (test006-test010) x 3 Noise x 5 Trials x 3 Solvers = 225
echo ==============================================================================

python -m streamlit run app.py -- --team "Team B"
pause
