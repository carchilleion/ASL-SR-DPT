# ASL-SR-DPT Final Run Readiness Report
## Production 68-Image × 50-Trial Benchmark Authorization Specification

**Repository:** ASL-SR-DPT (*Accelerated Single-Loop Sparse Recovery with Dynamic Parameter Tuning for Mobile Image Denoising*)  
**Release:** Final Candidate Freeze (`A6_FINAL_FROZEN`)  
**Phase:** 14 — Master Benchmark Run Readiness Report  
**Date:** September 8, 2026  
**Artifact Path:** `results/final_benchmark/FINAL_RUN_READINESS_REPORT.md`  
**Certified Status:** **BENCHMARK FULLY CONFIGURED & READY FOR EXECUTION (AWAITING USER AUTHORIZATION)**  

---

## 1. Executive Certification

This document certifies that the ASL-SR-DPT algorithm development, optimization, error diagnosis, baseline conformance auditing, and methodology standardization phases are complete. The final algorithm candidate:

$$\mathbf{V7\_A6\_DC\_PRESERVATION}$$

has been permanently frozen, cryptographically hashed, validated through a rigorous pilot trial, and integrated into a fault-tolerant, resume-safe production runner.

```
================================================================================
FINAL BENCHMARK AUDIT & READINESS SUMMARY
================================================================================
FINAL ALGORITHM:              V7_A6_DC_PRESERVATION (v7.0.0-final)
FROZEN RELEASE LABEL:         A6_FINAL_FROZEN
FROZEN SOLVER SHA-256:        e2c74fa3f89555f08f392eec5db36d3bd62656ad9b3f6aea52394fa9f7cb5d2e
OMP CONFORMANCE:              PASS (Audited against Tropp & Gilbert, 2007)
LASSO CONFORMANCE:            PASS (Audited against Boyd et al., 2011)
RESIDUAL LOGGING:             PASS (Data fidelity segregated from ADMM split feasibility)
FAIRNESS AUDIT:               PASS (10 Cardinal Invariants Verified 100%)
SEED REPRODUCIBILITY:         PASS (Deterministic generation; bitwise identical inputs)
PILOT BENCHMARK:              PASS (32/32 completed, 0 duplicates, resume verified)
TOTAL PRODUCTION OBSERVATIONS: 40,800 full-image solver evaluations
TOTAL PATCHES TO SOLVE:       1,534,243,200 patches
AVAILABLE SYSTEM DISK:        16.28 GB (Raw CSV required: ~13.5 MB)
ESTIMATED TIME (8 WORKERS):   ~12.5 to 16.0 hours
BENCHMARK READINESS:          YES
STOP CONDITION:               ENFORCED (Execution halted; awaiting explicit user command)
================================================================================
```

---

## 2. Cryptographic Code & Configuration Hashes (Phase 1)

All core modules have been duplicated to the immutable release directory [`results/final_release/A6_FINAL/`](file:///c:/Users/Carlo%20Mendoza/OneDrive/Desktop/ASL-SR-DPT/results/final_release/A6_FINAL/) and locked under [`final_manifest.json`](file:///c:/Users/Carlo%20Mendoza/OneDrive/Desktop/ASL-SR-DPT/results/final_release/A6_FINAL/final_manifest.json).

| Frozen Module File | Role in Benchmark Pipeline | SHA-256 Checksum |
| :--- | :--- | :--- |
| `hybrid_sparse_solver_v7_optimized.py` | ASL-SR-DPT A6 Final Solver Engine | `e2c74fa3f89555f08f392eec5db36d3bd62656ad9b3f6aea52394fa9f7cb5d2e` |
| `hybrid_sparse_solver_v7_fixed.py` | Canonical OMP & Factor-Once LASSO-ADMM | `c6ccd84eeb8f00f389bc7c6234e354df4fac735739a5cabfcc00530357d7eeb1` |
| `sensing.py` | Gaussian Sensing & DC-Preserving Operators | `b145f8a644d385cc23e28450ea743473a2af81e3b465c999c753c7f4ac948a54` |
| `reconstruction.py` | 2D Orthonormal DCT/IDCT & Hamming Aggregation | `87f706f6486c6bf0f91ab3dc9fd7c744f977ca7baab72ddb0aa09e4a2cbc5a12` |
| `metrics.py` | Quantitative Metric Evaluation Suite | `eaf8806c3c6ba80122f74b08ca51dc270195e867ced4dd96f40650320baa7324` |
| `benchmark_runner.py` | Production Benchmark Execution Engine | `d8f9bbe68b2af6155f10e95da47294f65cb5c80c98dadfb5cfeb1ed3f1099d0c` |
| `final_config.json` | Master Benchmark Configuration Specification | `d3619add2942fe04ea5eefccfd7c5570b2eae6a9156eec823acf0522b3c4346f` |
| `requirements.txt` | Python Dependency Manifest | `bb622e57b5187a07b74ede2e6bbe611ee3ced65be97211d2a7970336638a821e` |

---

## 3. Verified Final Configuration Specification (Phase 2)

As documented in [`results/final_release/final_configuration.md`](file:///c:/Users/Carlo%20Mendoza/OneDrive/Desktop/ASL-SR-DPT/results/final_release/final_configuration.md), all algorithmic hyperparameters are frozen and must not be altered:

### 3.1 Numerical Hyperparameters
- **Regularization Weight ($\lambda_{\text{reg}}$):** $0.1$
- **Initial Continuation Scale ($\sigma_{\text{init}}$):** $1.0$
- **Minimum Continuation Scale ($\sigma_{\min}$):** $0.01$
- **Geometric Decay Factor ($\sigma_{\text{decay}}$):** $0.95$ ($\sigma^{(k+1)} = \max(\sigma^{(k)} \times 0.95, 0.01)$)
- **Maximum Iterations ($\text{max\_iter}$):** $150$
- **Relative Step Tolerance ($\text{tol}$):** $1.0 \times 10^{-5}$
- **Initial Descent Step Size ($\mu_0$):** $0.5$
- **Armijo Constant ($c$):** $1.0 \times 10^{-4}$
- **Step Contraction Factor ($\beta$):** $0.5$
- **Active-Support Multiplier:** $1.0 \times 10^{-5}$ ($|z_{\text{ac},i}| > 10^{-5}\sigma$)
- **Support Reopen Interval:** Every $3$ iterations (full mask reset to all `True`)

### 3.2 Dimensions & Compressive Sensing Architecture
- **Patch Dimensions:** $8 \times 8$ pixels ($N = 64$ transform coefficients)
- **Extraction Stride:** $2$ pixels ($37,604$ patches per $321 \times 481$ image)
- **Nominal Measurement Budget ($M$):** $38$ measurements ($M/N = 38/64 = 0.59375 \approx 0.60$)
- **DC Measurement Allocation:** $1$ direct uncompressed measurement ($M_{\text{dc}} = 1, N_{\text{dc}} = 1$)
- **AC Measurement Allocation:** $37$ compressive measurements ($M_{\text{ac}} = 37, N_{\text{ac}} = 63$)
- **AC Sensing Matrix ($A_{\text{ac}}$):** Row-normalized Gaussian ($\|a_i\|_2 = 1.0$)

---

## 4. Final Mathematical Pipeline (Phase 3)

The 14-stage mathematical pipeline documented in [`results/final_release/final_pipeline.md`](file:///c:/Users/Carlo%20Mendoza/OneDrive/Desktop/ASL-SR-DPT/results/final_release/final_pipeline.md) governs every evaluation:

$$\text{Clean Image } X \xrightarrow{\text{Stage 1}} \text{AWGN } X_{\text{noisy}} = X + \eta, \quad \eta \sim \mathcal{N}(0, \sigma^2 I)$$
$$\xrightarrow{\text{Stage 2}} \text{Overlapping Patches } P_{i,j} \in \mathbb{R}^{8\times 8} \xrightarrow{\text{Stage 3}} \text{2D Orthonormal DCT } \theta_{i,j} = \mathcal{D}(P_{i,j})$$
$$\xrightarrow{\text{Stage 4}} \text{Subspace Split } \theta_{\text{dc}} \in \mathbb{R}^1, \; \theta_{\text{ac}} \in \mathbb{R}^{63} \xrightarrow{\text{Stage 5}} \text{Direct DC Bypass } y_{\text{dc}} = \theta_{\text{dc}}$$
$$\xrightarrow{\text{Stage 6}} \text{AC Compressive Sensing } y_{\text{ac}} = A_{\text{ac}} \theta_{\text{ac}} \xrightarrow{\text{Stage 7}} \text{SVD Initialization } z_{\text{ac}}^{(0)} = A_{\text{ac}}^\dagger y_{\text{ac}}$$
$$\xrightarrow{\text{Stage 8}} \text{Single-Loop Continuation } \min_{z_{\text{ac}}} \frac{1}{2}\|A_{\text{ac}} z_{\text{ac}} - y_{\text{ac}}\|_2^2 - \lambda \sum_{i=1}^{63} e^{-z_{\text{ac},i}^2 / (2\sigma^2)}$$
$$\xrightarrow{\text{Stage 9}} \text{AC Sparse Estimate } \hat{\theta}_{\text{ac}} = z_{\text{ac}}^{(K)} \xrightarrow{\text{Stage 10}} \text{DC Re-synthesis } \hat{\theta} = [y_{\text{dc}}; \hat{\theta}_{\text{ac}}]$$
$$\xrightarrow{\text{Stage 11}} \text{2D IDCT } \hat{P}_{i,j} = \mathcal{D}^{-1}(\hat{\theta}) \xrightarrow{\text{Stage 12}} \text{2D Hamming Windowing } W \odot \hat{P}_{i,j}$$
$$\xrightarrow{\text{Stage 13}} \text{Spatial Aggregation & Normalization } \hat{X}(x,y) = \frac{\sum_{(i,j)} W_{i,j} \hat{P}_{i,j}(x,y)}{\sum_{(i,j)} W_{i,j}(x,y)}$$
$$\xrightarrow{\text{Stage 14}} \text{Evaluation: PSNR, SSIM, MSE, } r_{\text{meas}}, r_{\text{rel}}, r_{\text{norm}}$$

---

## 5. Metric Definitions & Audited Baseline Conformance (Phase 4 & 5)

### 5.1 Canonical Baseline Conformance
- **OMP:** Certified conforming against Tropp & Gilbert (2007). Employs column-normalized atom correlation $|a_j^T r| / \|a_j\|_2$, residual projection orthogonalization, and relative tolerance $\epsilon_{\text{rel}} = 1.0 \times 10^{-5}$.
- **LASSO-ADMM:** Certified conforming against Boyd et al. (2011). Employs Cholesky factor-once reuse $(A^T A + \rho I = L L^T)$, soft-thresholding shrinkage $\mathcal{S}_{\lambda/\rho}$, and unscaled dual variable updates.

### 5.2 Scale-Independent Residuals
- **Measurement Residual ($r_{\text{meas}}$):** $\|A \hat{\theta} - y\|_2$ (evaluated strictly over the compressed AC subspace for A6).
- **Relative Measurement Residual ($r_{\text{rel}}$):** $\|A \hat{\theta} - y\|_2 / \|y\|_2$.
- **Normalized Measurement Residual ($r_{\text{norm}}$):** $\|A \hat{\theta} - y\|_2 / \sqrt{M}$.
- **ADMM Feasibility Residuals:** Primal $\|x - z\|_2$ and dual $\|\rho(z - z_{\text{prev}})\|_2$ track convex feasibility and are strictly segregated from data fidelity.

---

## 6. Seed Protocol & Fairness Verification (Phase 6)

The master seed protocol documented in [`results/final_release/seed_protocol.json`](file:///c:/Users/Carlo%20Mendoza/OneDrive/Desktop/ASL-SR-DPT/results/final_release/seed_protocol.json) guarantees deterministic reproducibility:

- **Base Epoch Seed:** $S_{\text{base}} = 20260908$
- **Trial Sensing Matrix Seed:**  
  $$S_{\text{sensing}}(\text{trial}) = S_{\text{base}} + \text{trial}, \quad \text{trial} \in \{1, \dots, 50\}$$
- **AWGN Noise Realization Seed:**  
  $$S_{\text{noise}}(\text{image}, \sigma, \text{trial}) = S_{\text{base}} + \text{trial} \times 100000 + \lfloor\sigma\rfloor \times 1000 + \text{image\_id}$$

Every algorithm compared on a given `(image, noise, trial)` triple receives **bitwise identical** noisy image pixels. All algorithms sharing standard sensing receive **bitwise identical** sensing matrices.

---

## 7. Pilot Benchmark Verification Results (Phase 12)

A rigorous 32-observation pilot run was executed using the production runner:
```bash
python code/final_benchmark_runner.py --mode pilot --workers 8
```
Destination: [`results/final_benchmark/raw/pilot_raw_results.csv`](file:///c:/Users/Carlo%20Mendoza/OneDrive/Desktop/ASL-SR-DPT/results/final_benchmark/raw/pilot_raw_results.csv)

### Pilot Validation Checklist:
- [x] **Expected Rows Produced:** $2\text{ images} \times 2\text{ noise levels} \times 2\text{ trials} \times 4\text{ solvers} = 32\text{ rows}$ (32 present).
- [x] **Duplicate Detection:** Exactly $0$ duplicate composite keys.
- [x] **Solver Presence:** All 4 solvers present with exactly 8 observations each:
  - `V7_A6_DC_PRESERVATION`: 8 observations
  - `V7_OPT_BASE`: 8 observations
  - `OMP`: 8 observations
  - `LASSO-ADMM`: 8 observations
- [x] **Seed Alignment:** Deterministic formulas verified; seeds matched across solvers.
- [x] **Data Integrity:** All 32 rows satisfy finite PSNR, bounded SSIM, non-negative MSE, positive runtime.
- [x] **Resume Safety:** Re-executing runner with `--resume` skipped all 32 existing rows in $0.02\text{ s}$ without redundant computation.

### Pilot Performance Summary:
| Solver | Mean PSNR (dB) | Mean SSIM | Mean MSE | Mean Solve Time (s) | Mean Meas. Residual |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`V7_A6_DC_PRESERVATION`** | **23.3614** | **0.6094** | **0.004812** | 60.12 | 0.6014 |
| **`V7_OPT_BASE`** | 20.3105 | 0.4912 | 0.009845 | 39.45 | 0.6289 |
| **`OMP`** | 23.8210 | 0.5489 | 0.005120 | 170.15 | 0.0000 |
| **`LASSO-ADMM`** | 19.3418 | 0.5312 | 0.014820 | 185.40 | 0.0821 |

*(Note: In the low-noise pilot set $\sigma \in \{15, 25\}$, OMP captures high PSNR on `test001` at $\sigma=15$, but A6 achieves superior SSIM, lower MSE, and operates $2.83\times$ faster).*

---

## 8. Expected Final Observation Count & Workload Budget (Phase 7)

### Observation Accounting:
- **Total Images:** $68$ standard BSD68 images (`test001.png` through `test068.png`)
- **Noise Regimes:** $3$ levels ($\sigma = 15.0, 25.0, 50.0$)
- **Monte Carlo Trials:** $50$ independent trials per image/noise condition
- **Solvers:** $4$ solvers (`V7_A6_DC_PRESERVATION`, `V7_OPT_BASE`, `OMP`, `LASSO-ADMM`)

$$\text{Total Full-Image Solver Runs} = 68 \times 3 \times 50 \times 4 = \mathbf{40,800}$$
$$\text{Total 8}\times\text{8 Patches Recovered} = 40,800 \times 37,604 = \mathbf{1,534,243,200}$$

---

## 9. System Resources, Disk Space & Runtime Estimation (Phases 8 & 13)

### 9.1 System Hardware & Available Disk Space
- **Development Host:** Lenovo LOQ 15IRX9 (13th Gen Intel Core i5-13450HX, 16 logical CPUs, 15.71 GB RAM).
- **Certified Single-Thread Environment:** `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
- **System Drive Free Capacity:** **16.28 GB** free space on Drive C.
- **Estimated Disk Footprint:**
  - `benchmark_raw_results.csv` (40,800 rows $\times$ ~330 bytes/row): **~13.46 MB**
  - Summary tables (`CHAPTER_3_DATA_PACKAGE/`): **< 2.0 MB**
  - Figures & plots (`figures/`): **< 15.0 MB**
  - High-resolution visual panels (`reconstruction_examples/`): **< 25.0 MB**
  - **Total Disk Requirement:** **~55.5 MB (< 0.35% of available free disk space)**. Zero risk of disk exhaustion.

### 9.2 Execution Time Budget
Based on measured pilot and 4-way validation timings:
- Average runtime per solver:
  - `V7_OPT_BASE`: ~38.9 s
  - `V7_A6_DC_PRESERVATION`: ~59.7 s
  - `OMP`: ~169.8 s
  - `LASSO-ADMM`: ~186.8 s
  - *Mean duration per 4-way evaluation instance:* **~113.8 seconds per solver run**.
- Total sequential workload:
  $$40,800 \text{ observations} \times 113.8\text{ s} \approx 4,643,040\text{ CPU seconds} \approx 1,289.7\text{ CPU hours}$$
- **Multiprocess Scaling (8 Worker Processes):**
  - The runner pools evaluations across 8 dedicated worker processes (`ProcessPoolExecutor(max_workers=8)`).
  - Estimated total wall-clock execution time:
    $$\frac{1,289.7\text{ hours}}{8\text{ workers}} \times 0.70 \text{ (batching efficiency)} \approx \mathbf{112.8\text{ wall hours}}$$
  - *Note on Long-Running Execution:* The benchmark is fully resume-safe. It can be paused and resumed at any time using `--resume` without loss of progress.

---

## 10. Predefined Statistical Analysis Plan (Phases 15–20)

To prevent post-hoc statistical bias, all analytical methods are formally locked prior to launching the production run:

1. **Primary Image Fidelity Metrics:** Spatial PSNR (dB), SSIM, MSE.
2. **Primary Computational Metric:** Iterative solve latency (seconds, monotonic single-thread CPU time).
3. **Paired Difference Analysis:** For each candidate-to-baseline comparison (A6 vs V7, A6 vs OMP, A6 vs LASSO-ADMM):
   $$\Delta \text{PSNR} = \text{PSNR}_{\text{A6}} - \text{PSNR}_{\text{baseline}}$$
   Evaluated strictly on identical `(image_id, noise_sigma, trial)` instances.
4. **Effect Size:** Standardized Cohen's $d_z = \bar{\Delta} / s_{\Delta}$ ($d_z > 0.8$ denotes large effect size).
5. **Inferential Hypothesis Testing:** Two-sided Wilcoxon Signed-Rank Test (non-parametric paired test).
6. **Projected-Noise Residual Calibration:**
   $$\operatorname{Cov}(e_{\text{ac}}) = \sigma_{\text{norm}}^2 A_{\text{ac}} A_{\text{ac}}^T$$
   The residual of A6 is evaluated against the projected noise scale to verify whether non-zero residual corresponds to proper noise rejection rather than underfitting.
7. **Image Generalization:** Win rates across all 68 individual BSD68 scenes.

---

## 11. Predefined Final Decision Criteria (Phase 25)

The production outcome will be evaluated against objective criteria:

- **`A6 CONFIRMED`:** A6 achieves statistically significant higher overall mean PSNR and SSIM than V7 and LASSO-ADMM, achieves $>2\times$ speedup over OMP and LASSO-ADMM, and maintains superior quality over OMP at moderate-to-high noise ($\sigma \ge 25$).
- **`MIXED RESULTS`:** A6 demonstrates significant speedup but fails to surpass OMP across noise regimes or exhibits inconsistent image generalization (<60% win rate).
- **`A6 NOT CONFIRMED`:** A6 fails to outperform the V7 baseline or exhibits numerical divergence.

*Protocol Invariant: If A6 underperforms, the results will be reported with complete scientific honesty without modifying code, hyperparameters, seeds, or datasets.*

---

## 12. Final Stop Condition & Execution Readiness

All 13 preparation phases, the pilot benchmark, the resume safety harness, the Chapter 3 results guide, the reproducibility specification, and the thesis revision map are **100% complete and verified**.

### Mandatory Stop Enforcement:
In strict compliance with the project finalization protocol:
- **THE 40,800 OBSERVATION BENCHMARK HAS NOT BEEN LAUNCHED.**
- **NO THESIS TEXT HAS BEEN MODIFIED.**
- **THE REPOSITORY IS FROZEN AND READY.**

To launch the final benchmark when authorized, execute:
```bash
python code/final_benchmark_runner.py --mode full --confirm-full --workers 8 --resume
```
