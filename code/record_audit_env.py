import os
import sys
import hashlib
import json
import platform
import psutil
import numpy as np
import scipy as sp
import PIL

files_to_hash = [
    'code/hybrid_sparse_solver_v7_fixed.py',
    'code/hybrid_sparse_solver_v7_optimized.py',
    'code/sensing.py',
    'code/reconstruction.py',
    'code/metrics.py',
    'code/benchmark_runner.py',
    'configs/final_config.json'
]

hashes = {}
for f in files_to_hash:
    if os.path.exists(f):
        with open(f, 'rb') as fp:
            hashes[f] = hashlib.sha256(fp.read()).hexdigest()
    else:
        hashes[f] = 'FILE_NOT_FOUND'

env_info = {
    'audit_label': 'FINAL_BASELINE_CONFORMANCE_AUDIT',
    'timestamp': '2026-09-08T17:25:30',
    'implementation_labels': {
        'V7_FROZEN': 'code/hybrid_sparse_solver_v7_fixed.py:HybridSparseSolverV7',
        'V7_OPT_FROZEN': 'code/hybrid_sparse_solver_v7_optimized.py:HybridSparseSolverV7',
        'OMP_CURRENT': 'code/hybrid_sparse_solver_v7_fixed.py:run_omp',
        'LASSO_CURRENT': 'code/hybrid_sparse_solver_v7_fixed.py:run_lasso_admm',
        'A6_CURRENT': 'code/sensing.py:generate_dc_preserving_measurement + hybrid_sparse_solver_v7_fixed.py'
    },
    'file_sha256_hashes': hashes,
    'python_version': sys.version,
    'numpy_version': np.__version__,
    'scipy_version': sp.__version__,
    'pillow_version': PIL.__version__,
    'os': platform.platform(),
    'system': platform.system(),
    'release': platform.release(),
    'machine': platform.machine(),
    'processor': platform.processor(),
    'cpu_count_logical': os.cpu_count(),
    'ram_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
    'ram_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
    'vm_configuration': 'Native host (non-virtualized bare metal)',
    'threads': {
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', 'unset'),
        'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS', 'unset'),
        'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS', 'unset'),
        'NUMEXPR_NUM_THREADS': os.environ.get('NUMEXPR_NUM_THREADS', 'unset'),
        'VECLIB_MAXIMUM_THREADS': os.environ.get('VECLIB_MAXIMUM_THREADS', 'unset')
    }
}

os.makedirs('results/final_audit', exist_ok=True)
out_path = 'results/final_audit/environment.json'
with open(out_path, 'w') as fp:
    json.dump(env_info, fp, indent=2)
print('Wrote results/final_audit/environment.json')
print(json.dumps(env_info, indent=2))
