@echo off
echo Starting ASL-SR-DPT Final BSD68 Benchmark Standalone Application...
cd /d "%~dp0"
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set MAX_CPU_WORKERS=15
streamlit run final_benchmark_app.py --server.port 8502 --server.headless false
pause
