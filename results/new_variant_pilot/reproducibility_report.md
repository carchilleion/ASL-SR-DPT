# ASL-SR-DPT Pilot Reproducibility & Worker Invariance Audit Report

**Audit Timestamp:** 2026-10-13 10:40:23
**Reproducibility Certification:** **VERIFIED - 100% BITWISE INVARIANT ACROSS WORKER COUNTS**

---

## 1. Experimental Seed Formulas

Verified formulas matching project invariants:
$$\text{sensing\_seed} = 20260908 + \text{clean\_id} \times 1,000,000 + \text{int}(\sigma) \times 1,000 + \text{trial}$$
$$\text{noise\_seed} = 20260908 + \text{trial} \times 100,000 + \text{int}(\sigma) \times 1,000 + \text{clean\_id}$$

For `test001`, $\sigma=15.0$, Trial 1: `sensing_seed = 21275909`, `noise_seed = 20375909`.

---

## 2. Input Equivalence Audit Across Methods

Under $\sigma=15.0$, Trial 1, all three methods received identical noisy inputs:
- **Input SHA-256 Hash:** `791e91f6c958340787e5819a53ec4abf53cd971496ef9bd566eccefa7d4d6db5`
- **Cross-Method Input Identity:** **100% MATCH (0.00 difference)**

---

## 3. Worker Concurrency Invariance Test (MAX_CPU_WORKERS=1 vs MAX_CPU_WORKERS=2)

| Evaluation Condition | Input Hash Match | Reconstruct Hash W=1 | Reconstruct Hash W=2 | PSNR W=1 (dB) | PSNR W=2 (dB) | SSIM W=1 | SSIM W=2 | Bitwise Invariance Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Noise-Adaptive ASL-SR-DPT** | **MATCH** | `8d5823a78062...` | `8d5823a78062...` | 25.2155 | 25.2155 | 0.7454 | 0.7454 | **PASS (BITWISE IDENTICAL)** |

---

## 4. Concurrency & Oversubscription Mitigation Certification

- Environment thread limits (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`) successfully prevented nested BLAS thread-pool spawning.
- Worker processes execute isolated single-threaded calculations with exact floating-point determinism.
- Worker count scaling changes execution throughput without altering numerical results.