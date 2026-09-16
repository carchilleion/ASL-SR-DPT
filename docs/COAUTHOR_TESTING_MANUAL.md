# ASL-SR-DPT Co-Author Beginner Testing Manual
**Exact Operational Instructions for Co-Authors and Research Operators**

---

## 1. Purpose & Goals
This manual provides exact operational instructions for testing the **ASL-SR-DPT** sparse recovery algorithm against **OMP** and **LASSO-ADMM** on the **BSD68** benchmark dataset.
- **Rule 1**: Follow this manual from top to bottom. Do not improvise benchmark settings.
- **Rule 2**: Do not alter solver mathematics or change parameters to artificially improve performance.
- **Rule 3**: All empirical results, including cases where ASL-SR-DPT performs below a baseline, must be recorded transparently (Scientific Honesty in Reporting).

---

## 2. Core Terminology & Glossary

| Term | Meaning | Operational Formula / Interpretation |
| :--- | :--- | :--- |
| **PSNR** | Peak Signal-to-Noise Ratio | $10 \log_{10}(1 / \text{MSE})$ (dB). Higher is better. |
| **SSIM** | Structural Similarity Index | Structural fidelity in range $[0, 1]$. Higher is better. |
| **MSE** | Mean Squared Error | Average squared pixel difference. Lower is better. |
| **Iterative Solve Time** | Time inside solver loop | Strictly measured using `time.perf_counter()`. Excludes setup, DCT, and aggregation. |
| **Setup Time** | Precomputation duration | Time to compute $P_{\text{init}} = A^\dagger$ or Cholesky factor $L = \text{chol}(A^T A + \rho I)$. Reported separately. |
| **$R_{\text{active}}$** | Active-support ratio | Fraction of DCT coefficients actively participating in gradient updates: $\frac{1}{k_{\max} N} \sum_{i=1}^{k_{\max}} \|z^{(i)}_S\|_0$. |
| **Failed Line Searches** | Failed backtracks | Count of Armijo search failures where $\mu \leftarrow 0.5\mu$, $z$ is frozen, and $\sigma$ does **not** decay. |
| **Sigma Continuation** | Parameter schedule | $\sigma \leftarrow \max(0.95\sigma, \sigma_{\min})$ occurring **only** after an accepted step. |

---

## 3. Environment & Single-Thread Enforcement

To guarantee fair, reproducible timing without background thread scheduler contention, all linear algebra libraries are locked to single-threaded execution (`threads: 1`):

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```
*(Note: `benchmark_runner.py` sets these variables automatically upon startup).*

---

## 4. What the Co-Author Should Run, in Order

| Order | Action | Command | Purpose |
| :---: | :--- | :--- | :--- |
| **1** | Run Pytest Suite | `python -m pytest` | 16 automated tests verifying DCT, gradients, line-search, metrics |
| **2** | Run E1 Verification | `python benchmark_runner.py --experiment E1 --images 1 --trials 1` | 15 sanity checks verifying BSD68, dimensions, pseudoinverse, gradients |
| **3** | Run Performance Pilot | `python benchmark_runner.py --experiment E2 --images 1 --trials 1 --pilot` | Measures patch throughput and projects 50-trial full benchmark duration |
| **4** | Standard Sensing (E2) | `python benchmark_runner.py --experiment E2 --images 1 --trials 1 --noise 15 --sensing standard` | Smoke test on standard random sensing ($M=38, N=64$) |
| **5** | DC-Preserving (E3) | `python benchmark_runner.py --experiment E3 --images 1 --trials 1 --noise 15 --sensing dc` | Smoke test on DC-preserving model ($M_{\text{AC}}=37$, uncompressed DC) |
| **6** | Midpoint Ablation (E4) | `python benchmark_runner.py --experiment E4 --images 5 --trials 1 --noise 15` | Compares Midpoint ON vs. Midpoint OFF under identical conditions |
| **7** | Continuation Ablation (E5) | `python benchmark_runner.py --experiment E5 --images 5 --trials 1 --noise 15` | Evaluates $\sigma_{\text{decay}} \in \{0.90, 0.95, 0.98\}$ |
| **8** | Regularization Grid (E6) | `python benchmark_runner.py --experiment E6 --images 5 --trials 1 --noise 15` | Evaluates $\lambda \in \{0.01, 0.05, 0.1, 0.2, 0.5\}$ |
| **9** | Pairwise OMP (E7) | `python benchmark_runner.py --experiment E7 --images 68 --trials 50` | Matched pairwise benchmark: ASL-SR-DPT vs OMP |
| **10** | Pairwise LASSO (E8) | `python benchmark_runner.py --experiment E8 --images 68 --trials 50` | Matched pairwise benchmark: ASL-SR-DPT vs LASSO-ADMM |
| **11** | Audit Raw Observations | Automated in `summary.py` | Audits for NaN/Inf, duplicate keys, negative runtimes; logs to `logs/failed_runs.csv` |
| **12** | Checkpoint Resume | Add `--resume` to any command | Safely resumes interrupted runs without recomputing completed observations |

---

## 5. Fair Comparison Guarantees (The 5 Constants)

For every matched comparison across ASL-SR-DPT, OMP, and LASSO-ADMM:
1. **Same clean reference image**
2. **Same noisy image** (single AWGN draw with deterministic `noise_seed`)
3. **Same sensing matrix $A$** (generated once per trial with `sensing_seed` and reused across all images/patches)
4. **Same patch coordinates** (extracted at `patch_size=8`, `stride=2`)
5. **Same measurement vector $y$** ($y = A \theta$)
6. **Same 2D Hamming reconstruction aggregation** ($\text{Hamming}(8) \otimes \text{Hamming}(8)$)

---

## 6. Raw CSV Schema Reference

Every observation is written with the following exact columns:
```text
trial,image_id,noise_sigma,sensing_mode,solver,psnr,ssim,mse,solve_time,setup_time,iterations,final_sigma,residual,active_support_ratio,failed_line_searches,accepted_steps,seed,code_version,patch_count,run_id,experiment
```

---

## 7. Required Chapter 3 Evidence Mapping

| Chapter 3 Section | Required Evidence | Produced By |
| :--- | :--- | :--- |
| **3.1 Mathematical Verification** | Analytical vs numerical gradient check ($<10^{-5}$), finite values | `validation.py` (E1) |
| **3.2 Armijo & Continuation** | Step size $\mu$ trajectory, $\sigma$ decay on accepted steps, failed search freezing | `HybridSparseSolverV7` diagnostics |
| **3.3 Active Support** | $R_{\text{active}}$, active count history, reopening every 3 iterations | `HybridSparseSolverV7` diagnostics |
| **3.4 SVD Initialization** | Minimum-norm initialization via $P_{\text{init}} = A^\dagger$ | `setup_time` and `iterations` |
| **3.5 Spatial Reconstruction** | Reconstructed BSD68 panels, full spatial PSNR, SSIM, MSE | `reconstruction.py`, `metrics.py` |
| **3.6 Ablations** | Standard vs DC sensing (E2 vs E3), Midpoint ON vs OFF (E4), $\sigma$ decay (E5), $\lambda$ (E6) | Experiments E2, E3, E4, E5, E6 |
| **3.7 OMP Comparison** | Matched PSNR, SSIM, runtime speedup, latency reduction | Experiment E7 |
| **3.8 LASSO Comparison** | Matched PSNR, SSIM, runtime speedup, latency reduction | Experiment E8 |
| **3.9 Trade-off Summary** | Mean $\pm$ std tables, speedup factors, active-support ratio reduction | `summary.py` (`results/summaries/`) |

---

## 8. Handover Checklist for Lead Author

Before delivering experimental evidence to the lead author, verify:
- [ ] All 68 BSD68 images processed in final runs.
- [ ] Noise levels $\sigma \in \{15, 25, 50\}$ completed.
- [ ] All raw CSV files intact in `results/raw/`.
- [ ] Summary CSV files generated in `results/summaries/`.
- [ ] No NaN or Inf in any observation row.
- [ ] Setup times and solve times recorded separately.
- [ ] Representative reconstruction images saved in `results/reconstructions/`.
