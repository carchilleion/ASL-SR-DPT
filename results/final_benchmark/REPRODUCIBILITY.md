# ASL-SR-DPT Final Benchmark Reproducibility Specification

**Repository:** ASL-SR-DPT (*Accelerated Single-Loop Sparse Recovery with Dynamic Parameter Tuning for Mobile Image Denoising*)  
**Release:** Final Candidate Freeze (`A6_FINAL_FROZEN`)  
**Date:** September 8, 2026  
**Artifact Path:** `results/final_benchmark/REPRODUCIBILITY.md`  

---

## 1. Execution Commands

### 1.1 Pilot Verification Execution (Phase 12)
```bash
python code/final_benchmark_runner.py --mode pilot --workers 8
```
- Evaluates: 2 images (`test001`, `test002`), 2 noise levels ($\sigma \in \{15, 25\}$), 2 trials ($1, 2$), 4 solvers.
- Total evaluations: 32 full-image reconstructions.
- Status: **PASSED (32/32 completed, 0 duplicates, resume verified)**.

### 1.2 Full Production Benchmark Execution (Phase 7 / 14)
```bash
python code/final_benchmark_runner.py --mode full --confirm-full --workers 8 --resume
```
- Evaluates: 68 BSD68 images (`test001`–`test068`), 3 noise levels ($\sigma \in \{15, 25, 50\}$), 50 trials ($1$–$50$), 4 solvers.
- Total evaluations: $68 \times 3 \times 50 \times 4 = 40,800$ full-image reconstructions.
- Safety Guard: Will not execute without explicit `--confirm-full` authorization flag.

---

## 2. Certified Execution Environment

| Environment Component | Specification / Version | Verification Digest / Output |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Enterprise (Build 26100) | `platform.platform()` |
| **CPU Architecture** | AMD64 Family 25 Model 104 Stepping 1 | 16 logical CPUs / 8 physical cores |
| **System Memory (RAM)** | 15.71 GB Physical RAM | Verified available |
| **Python Runtime** | Python 3.12.9 (64-bit) | MSC v.1942 64 bit |
| **NumPy Version** | 2.4.3 | Pure BLAS single-thread bound |
| **SciPy Version** | 1.15.3 | Cholesky & linear algebra routines |
| **Pillow (PIL) Version** | 12.1.1 | 8-bit image I/O |
| **Scikit-Image Version** | 0.25.2 | SSIM evaluation (`data_range=1.0`) |
| **Matplotlib Version** | 3.10.1 | Figure rendering (`Agg` non-interactive backend) |

### Thread Binding Protocol
```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```
Ensures every worker process executes in strict single-threaded isolation, preventing BLAS thread contention and guaranteeing monotonic timing comparability.

---

## 3. Cryptographic Code & Configuration Hashes

| Frozen Module Path | Role | SHA-256 Checksum |
| :--- | :--- | :--- |
| `results/final_release/A6_FINAL/hybrid_sparse_solver_v7_optimized.py` | ASL-SR-DPT A6 Final Solver | `e2c74fa3f89555f08f392eec5db36d3bd62656ad9b3f6aea52394fa9f7cb5d2e` |
| `results/final_release/A6_FINAL/hybrid_sparse_solver_v7_fixed.py` | Canonical OMP & LASSO Baselines | `c6ccd84eeb8f00f389bc7c6234e354df4fac735739a5cabfcc00530357d7eeb1` |
| `results/final_release/A6_FINAL/sensing.py` | Compressive Sensing Operators | `b145f8a644d385cc23e28450ea743473a2af81e3b465c999c753c7f4ac948a54` |
| `results/final_release/A6_FINAL/reconstruction.py` | DCT/IDCT & Hamming Aggregation | `87f706f6486c6bf0f91ab3dc9fd7c744f977ca7baab72ddb0aa09e4a2cbc5a12` |
| `results/final_release/A6_FINAL/metrics.py` | Quantitative Metric Evaluation | `eaf8806c3c6ba80122f74b08ca51dc270195e867ced4dd96f40650320baa7324` |
| `results/final_release/A6_FINAL/final_config.json` | Master Configuration File | `d3619add2942fe04ea5eefccfd7c5570b2eae6a9156eec823acf0522b3c4346f` |
| `results/final_release/A6_FINAL/benchmark_runner.py` | Production Benchmark Driver | `d8f9bbe68b2af6155f10e95da47294f65cb5c80c98dadfb5cfeb1ed3f1099d0c` |

---

## 4. Deterministic Seed Generation Equations

- **Base Epoch Seed:** $S_{\text{base}} = 20260908$
- **Trial Sensing Matrix Seed:**  
  $$S_{\text{sensing}}(\text{trial}) = S_{\text{base}} + \text{trial}, \quad \text{trial} \in \{1, \dots, 50\}$$
  *One sensing matrix generated per trial, shared across all 68 images and all noise levels within that trial.*
- **AWGN Noise Realization Seed:**  
  $$S_{\text{noise}}(\text{image}, \sigma, \text{trial}) = S_{\text{base}} + \text{trial} \times 100000 + \lfloor\sigma\rfloor \times 1000 + \text{image\_id}$$
  *Bitwise identical across all 4 solvers compared in each condition.*

---

## 5. Resume Checkpoint & Observation Accounting

- **Raw Observation Destination:** `results/final_benchmark/raw/benchmark_raw_results.csv`
- **Composite Primary Key:** `(image_id, trial, noise_sigma, solver)`
- **Deduplication Policy:** If an entry with matching composite key exists in the raw CSV, it is parsed, verified for completeness, and skipped.
- **Data Integrity Guards:** Pre-write assertion verifies $0 < \text{PSNR} < 100$, $-1 \le \text{SSIM} \le 1$, $\text{MSE} \ge 0$, $\text{solve\_time} > 0$, and non-negative measurement residual.
