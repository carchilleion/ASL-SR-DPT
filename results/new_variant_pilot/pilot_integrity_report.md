# ASL-SR-DPT Stage 1: New-Variant Pilot Integrity Report

**Execution Timestamp:** 2026-10-13T14:31:25.918048
**Audit Status:** PASSED - PERFECT INTEGRITY

---

## 1. Expected vs. Actual Workload Audit

| Metric Dimension | Expected Count | Actual Count | Pass/Fail Status |
| :--- | :---: | :---: | :---: |
| **Total Completed Evaluations** | **750** | **750** | PASS |
| **Unique Composite Keys** | **750** | **750** | PASS |
| **Duplicate Key Count** | **0** | **0** | PASS |
| **Missing Evaluations** | **0** | **0** | PASS |

### 1.1 Workload by Method (Expected: 150 each)

| Method | Expected | Actual | Pass/Fail |
| :--- | :---: | :---: | :---: |
| Frozen ASL-SR-DPT | 150 | 150 | PASS |
| Fixed Candidate ASL-SR-DPT | 150 | 150 | PASS |
| Noise-Adaptive ASL-SR-DPT | 150 | 150 | PASS |
| OMP | 150 | 150 | PASS |
| LASSO-ADMM | 150 | 150 | PASS |

### 1.2 Workload by Noise Level (Expected: 250 each)

| Noise Level (sigma) | Expected | Actual | Pass/Fail |
| :--- | :---: | :---: | :---: |
| sigma = 15 | 250 | 250 | PASS |
| sigma = 25 | 250 | 250 | PASS |
| sigma = 50 | 250 | 250 | PASS |

### 1.3 Workload by Image (Expected: 75 each)

| Image ID | Expected | Actual | Pass/Fail |
| :--- | :---: | :---: | :---: |
| test001 | 75 | 75 | PASS |
| test002 | 75 | 75 | PASS |
| test003 | 75 | 75 | PASS |
| test004 | 75 | 75 | PASS |
| test005 | 75 | 75 | PASS |
| test006 | 75 | 75 | PASS |
| test007 | 75 | 75 | PASS |
| test008 | 75 | 75 | PASS |
| test009 | 75 | 75 | PASS |
| test010 | 75 | 75 | PASS |

---

## 2. Solver Performance Summary Table

| Method | Noise (sigma) | Evaluations | PSNR Mean (dB) | PSNR Std | SSIM Mean | SSIM Std | MSE Mean | Mean Runtime (s) | Mean Meas. Residual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frozen ASL-SR-DPT** | ALL | 150 | 23.93 | 2.24 | 0.6052 | 0.1157 | 0.004559 | 174.77 | 0.7591 |
| **Frozen ASL-SR-DPT** | 15.0 | 50 | 24.39 | 2.43 | 0.6246 | 0.1253 | 0.004183 | 192.50 | 0.5155 |
| **Frozen ASL-SR-DPT** | 25.0 | 50 | 24.22 | 2.29 | 0.6190 | 0.1221 | 0.004297 | 171.27 | 0.6840 |
| **Frozen ASL-SR-DPT** | 50.0 | 50 | 23.19 | 1.78 | 0.5720 | 0.0917 | 0.005196 | 160.53 | 1.0778 |
| **Fixed Candidate ASL-SR-DPT** | ALL | 150 | 24.24 | 3.87 | 0.5701 | 0.2009 | 0.005475 | 121.23 | 0.2182 |
| **Fixed Candidate ASL-SR-DPT** | 15.0 | 50 | 27.99 | 1.94 | 0.7783 | 0.0236 | 0.001749 | 155.11 | 0.1813 |
| **Fixed Candidate ASL-SR-DPT** | 25.0 | 50 | 25.39 | 1.17 | 0.6232 | 0.0384 | 0.002999 | 111.25 | 0.2236 |
| **Fixed Candidate ASL-SR-DPT** | 50.0 | 50 | 19.34 | 0.30 | 0.3088 | 0.0632 | 0.011675 | 97.34 | 0.2497 |
| **Noise-Adaptive ASL-SR-DPT** | ALL | 150 | 25.00 | 3.21 | 0.6121 | 0.1654 | 0.004082 | 133.10 | 0.3488 |
| **Noise-Adaptive ASL-SR-DPT** | 15.0 | 50 | 27.99 | 1.94 | 0.7783 | 0.0236 | 0.001749 | 155.13 | 0.1813 |
| **Noise-Adaptive ASL-SR-DPT** | 25.0 | 50 | 25.87 | 1.50 | 0.6635 | 0.0252 | 0.002747 | 130.30 | 0.3061 |
| **Noise-Adaptive ASL-SR-DPT** | 50.0 | 50 | 21.15 | 0.61 | 0.3946 | 0.0535 | 0.007751 | 113.87 | 0.5591 |
| **OMP** | ALL | 150 | 21.85 | 3.49 | 0.4466 | 0.1749 | 0.008889 | 206.98 | 0.0000 |
| **OMP** | 15.0 | 50 | 25.76 | 0.94 | 0.6377 | 0.0621 | 0.002717 | 233.81 | 0.0000 |
| **OMP** | 25.0 | 50 | 22.35 | 0.48 | 0.4582 | 0.0773 | 0.005849 | 196.26 | 0.0000 |
| **OMP** | 50.0 | 50 | 17.43 | 0.24 | 0.2439 | 0.0619 | 0.018102 | 190.87 | 0.0000 |
| **LASSO-ADMM** | ALL | 150 | 20.81 | 3.74 | 0.4876 | 0.1802 | 0.011673 | 205.30 | 0.0858 |
| **LASSO-ADMM** | 15.0 | 50 | 23.85 | 3.43 | 0.6836 | 0.0593 | 0.005735 | 233.86 | 0.0798 |
| **LASSO-ADMM** | 25.0 | 50 | 21.21 | 2.59 | 0.5026 | 0.0775 | 0.009497 | 194.01 | 0.0851 |
| **LASSO-ADMM** | 50.0 | 50 | 17.38 | 1.60 | 0.2767 | 0.0665 | 0.019787 | 188.02 | 0.0926 |

---

## 3. ASL Nonconvex Solver Diagnostic Analysis

| ASL Variant | Noise | Mean Iterations | Mean Active Ratio | Low Retention | Mid Retention | High Retention | Final Sigma |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Frozen ASL-SR-DPT | 15 | 92.7 | 0.9997 | 31.1% | 3.6% | 1.2% | 0.0100 |
| Frozen ASL-SR-DPT | 25 | 98.0 | 0.9998 | 31.2% | 5.5% | 2.9% | 0.0100 |
| Frozen ASL-SR-DPT | 50 | 111.9 | 0.9999 | 42.6% | 61.6% | 273.7% | 0.0100 |
| Fixed Candidate ASL-SR-DPT | 15 | 107.3 | 1.0000 | 76.1% | 71.8% | 174.8% | 0.1000 |
| Fixed Candidate ASL-SR-DPT | 25 | 139.7 | 1.0000 | 84.7% | 175.4% | 681.9% | 0.1000 |
| Fixed Candidate ASL-SR-DPT | 50 | 149.8 | 1.0000 | 134.7% | 809.3% | 4153.0% | 0.1000 |
| Noise-Adaptive ASL-SR-DPT | 15 | 107.3 | 1.0000 | 76.1% | 71.8% | 174.8% | 0.1000 |
| Noise-Adaptive ASL-SR-DPT | 25 | 115.1 | 1.0000 | 79.0% | 128.8% | 458.9% | 0.1000 |
| Noise-Adaptive ASL-SR-DPT | 50 | 124.7 | 1.0000 | 107.5% | 512.6% | 2584.5% | 0.1000 |

---

## 4. Hardware & Worker Concurrency Audit

- **Requested Worker Setting:** [np.int64(14), np.int64(12)]
- **Actual Logical Workers Allocated:** [np.int64(14), np.int64(12)]
- **Available System Logical Cores:** 16
- **Oversubscription Prevention:** Strict single-threading enforced per worker (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`).

---

## 5. Seed Formula Verification & Reproducibility

Verified invariants:
$$\text{sensing\_seed} = 20260908 + \text{clean\_id} \times 1,000,000 + \text{int}(\sigma) \times 1,000 + \text{trial}$$
$$\text{noise\_seed} = 20260908 + \text{trial} \times 100,000 + \text{int}(\sigma) \times 1,000 + \text{clean\_id}$$
- Every condition was uniquely seeded and is 100% reproducible.
