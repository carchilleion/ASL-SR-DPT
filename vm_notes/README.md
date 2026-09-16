# ASL-SR-DPT Benchmark Execution Environment Notes

## System Configuration
- **Operating System**: Windows 11 (build-compliant)
- **Runtime**: Python 3.12.9 64-bit
- **Linear Algebra Acceleration**: OpenBLAS / MKL underlying NumPy 2.4.3 / SciPy 1.15.3
- **Thread Policy**: Single-thread benchmarking (`threads: 1`) to ensure fair, deterministic comparability across algorithms.

## Execution Model
1. **Separation of Setup vs. Solve Timing**:
   - For **ASL-SR-DPT**, the Moore-Penrose pseudoinverse $P_{\text{init}} = \text{pinv}(A)$ is computed once per trial and recorded as `setup_time`.
   - For **LASSO-ADMM**, the Cholesky factor $L = \text{chol}(A^T A + \rho I)$ is factorized once per trial and recorded as `setup_time`.
   - For **OMP**, no precomputed factor is used (`setup_time = 0.0`).
   - `solve_time` strictly measures the iterative algorithm loop per patch via `time.perf_counter()`.
   - Patch extraction, 2D DCT, 2D IDCT, 2D Hamming aggregation, and metric calculations are strictly excluded from `solve_time`.

2. **Sensing Matrix Reuse Protocol (Step 15)**:
   - A single Gaussian random sensing matrix $A$ (or $A_{\text{ac}}$ for DC-preserving sensing) is generated per randomized trial.
   - The matrix is reused across all 68 images and all overlapping patches within the trial.

3. **Performance Extrapolation Note**:
   - While benchmark runs on this host reflect single-core x86_64 performance, the relative speedup and iteration-count advantages directly translate to resource-constrained embedded/mobile processors.
