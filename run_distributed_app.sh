#!/bin/bash
echo "Starting ASL-SR-DPT 3-PC Distributed Benchmark Application..."
cd "$(dirname "$0")"

# Pin single-thread linear algebra libraries
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Activate virtual environment if present
if [ -d "venv" ]; then
    source venv/bin/activate
fi

python3 -m streamlit run distributed_benchmark_app.py --server.port 8504 --server.headless false
