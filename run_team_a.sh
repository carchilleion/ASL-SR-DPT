#!/usr/bin/env bash
# ==============================================================================
# ASL-SR-DPT Pilot Benchmark - Launcher for TEAM A (Khevin)
# Strictly locks thread allocation for reproducible single-threaded timing.
# ==============================================================================

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=============================================================================="
echo "Launching ASL-SR-DPT Pilot Benchmark Application for TEAM A (Khevin)"
echo "Thread settings: OMP=1, OPENBLAS=1, MKL=1, NUMEXPR=1"
echo "Workload: 5 BSD68 Images (test001-test005) x 3 Noise x 5 Trials x 3 Solvers = 225"
echo "=============================================================================="

python3 -m streamlit run app.py -- --team "Team A"
