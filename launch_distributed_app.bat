@echo off
echo Starting ASL-SR-DPT 3-PC Distributed Benchmark Application...
cd /d "%~dp0"
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set MKL_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
streamlit run distributed_benchmark_app.py --server.port 8504 --server.headless false
pause
