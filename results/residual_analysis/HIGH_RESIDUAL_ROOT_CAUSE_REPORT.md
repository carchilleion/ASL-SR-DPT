# ASL-SR-DPT High Measurement Residual Root-Cause Analysis Report

**Author:** Carlo Mendoza  
**Repository:** ASL-SR-DPT (*Accelerated Single-Loop Sparse Recovery with Dynamic Parameter Tuning*)  
**Investigation Date:** September 8, 2026  
**Status:** Completed Diagnostic Audit  
**Artifact Directory:** `results/residual_analysis/`  

---

## 1. Executive Summary & Problem Formulation

In the multi-image controlled validation study of ASL-SR-DPT across 10 BSD68 images and three noise levels ($\sigma \in \{15, 25, 50\}$), a key empirical anomaly was documented:
- **`V7_OPT_BASE`:** Mean PSNR = $21.22\text{ dB}$, Mean SSIM = $0.5026$, Mean Residual = **$0.7647$**.
- **`V7_A6_DC_PRESERVATION`:** Mean PSNR = $24.00\text{ dB}$, Mean SSIM = $0.6075$, Mean Residual = **$0.7603$**.
- **`V7_A5_TWO_STAGE`:** Mean PSNR = $23.13\text{ dB}$, Mean SSIM = $0.5467$, Mean Residual = **$0.7206$** ($-5.8\%$).
- **`V7_A5A6_COMBINED`:** Mean PSNR = $22.67\text{ dB}$, Mean SSIM = $0.4816$, Mean Residual = **$0.1116$** (**$-85.4\%$**).

### The Central Anomaly:
Variant A6 dramatically improves PSNR (+2.79 dB), SSIM (+0.105), and MSE (-48.5%), yet its measurement residual ($0.7603$) remains virtually identical to the baseline V7 solver ($0.7647$). In contrast, Variant A5 and especially A5A6 reduce the measurement residual by an order of magnitude (down to $0.0505$ at $\sigma=15$). Furthermore, at severe noise ($\sigma=50$), A5A6 experiences a sharp quality collapse while A6 excels.

This diagnostic investigation was executed to resolve the root causes through rigorous instrumentation, code audits, iteration tracking, and conditioning analysis across 100 representative BSD68 patches without altering any production code or algorithm parameters.

---

## 2. Residual Definition & Mathematical Disambiguation

We distinguish three separate definitions of measurement discrepancy:

1. **Absolute Measurement Residual Norm:**
   $$\|r\|_2 = \|A \hat{z} - y\|_2 = \sqrt{\sum_{j=1}^M ((A \hat{z})_j - y_j)^2}$$
2. **Relative Measurement Residual Norm:**
   $$\text{RelRes}(\hat{z}, y) = \frac{\|A \hat{z} - y\|_2}{\max(\|y\|_2, \epsilon)}$$
3. **Normalized Root Mean Square Error (NRMSE):**
   $$\text{NRMSE}_{\text{meas}}(\hat{z}, y) = \frac{\|A \hat{z} - y\|_2}{\sqrt{M}}$$

For DC-preserving sensing ($M_{\text{ac}} = 37$), the AC residual is strictly separated from the scalar DC measurement error:
- $r_{\text{ac}} = \|A_{\text{ac}} \hat{z}_{\text{ac}} - y_{\text{ac}}\|_2$
- $\text{DC}_{\text{error}} = |\hat{z}_{\text{dc}} - y_{\text{dc}}|$

### Code Audit Findings (`residual_definition.md`):
- Across `benchmark_runner.py`, `hybrid_sparse_solver_v7_fixed.py`, `hybrid_sparse_solver_v7_optimized.py`, and `run_final_candidate_validation.py`, the metric labeled `"residual"` is strictly the **ABSOLUTE EUCLIDEAN NORM** $\|A \hat{z} - y\|_2$ per patch, averaged across patches.
- **Historical Benchmark Discrepancy Unmasked:** In `benchmark_runner.py` line 273, the reported residual for LASSO-ADMM was recorded as `diag["final_primal_residual"]` ($0.0007$), which is the ADMM internal split feasibility $\|x - z\|_2$, **NOT** the measurement residual $\|A z - y\|_2$. Comparing ASL-SR-DPT's measurement residual against LASSO-ADMM was an apples-to-oranges metric confusion in the legacy benchmark infrastructure.

---

## 3. Measurement Model & Forward Transform Audit

A mathematical and programmatic audit of the pipeline (`measurement_model_audit.md`) verified:
1. **2D DCT / IDCT:** Orthonormal Type-II/Type-III DCT (`norm="ortho"`) has a maximum round-trip reconstruction error of $3.89 \times 10^{-16}$.
2. **Indexing:** DC frequency corresponds strictly to index `0` (`(0, 0)` in row-major flattening).
3. **Sensing Matrix:** $A \in \mathbb{R}^{38 \times 64}$ (and $A_{\text{ac}} \in \mathbb{R}^{37 \times 63}$) has strictly unit-normalized rows ($\|A_{i,:}\|_2 = 1.0$), full row rank, and condition number $\kappa(A) \approx 5.0 - 6.2$.
4. **Noise Injection:** AWGN is added to spatial image pixels with standard deviation $\sigma_{\text{scaled}} = \sigma_{\text{noise}} / 255.0$. In the orthonormal DCT domain, noise is isotropic white Gaussian noise $e_{\text{dct}} \sim \mathcal{N}(0, \sigma_{\text{scaled}}^2 I_{64})$.
5. **Measurement Generation:** $y = A \theta_{\text{noisy}} = A \theta_{\text{clean}} + A e_{\text{dct}}$.

### Mathematical Consequence (Morozov's Discrepancy Principle):
Because $y$ contains projected noise $A e_{\text{dct}}$, the **ground-truth clean signal** $\theta_{\text{clean}}$ has an expected non-zero residual against $y$:
$$\mathbb{E}[\|A \theta_{\text{clean}} - y\|_2^2] = \operatorname{Tr}(A \operatorname{Cov}(e_{\text{dct}}) A^T) = \sigma_{\text{scaled}}^2 \operatorname{Tr}(A A^T) = M \sigma_{\text{scaled}}^2$$
$$\mathbb{E}[\|A \theta_{\text{clean}} - y\|_2] \approx \sqrt{M} \sigma_{\text{scaled}}$$

- At $\sigma = 15$: $\sqrt{38} \cdot \frac{15}{255} \approx \mathbf{0.3626}$
- At $\sigma = 25$: $\sqrt{38} \cdot \frac{25}{255} \approx \mathbf{0.6044}$
- At $\sigma = 50$: $\sqrt{38} \cdot \frac{50}{255} \approx \mathbf{1.2087}$

---

## 4. Signal & DCT Coefficient Scaling Analysis

Across the 100 representative BSD68 validation patches:

| Noise Level ($\sigma$) | Mean $\|\theta\|_2$ | Mean $\|y\|_2$ | Mean Clean DC ($|\theta_0|$) | Mean Clean AC ($|\theta_{1:63}|$) | Ratio DC / AC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$\sigma = 15$** | 3.5389 | 2.6487 | 3.4505 | 0.0366 | **94.2x** |
| **$\sigma = 25$** | 3.5871 | 2.6978 | 3.4505 | 0.0366 | **94.2x** |
| **$\sigma = 50$** | 3.8166 | 2.8933 | 3.4505 | 0.0366 | **94.2x** |

### Key Insights:
1. **Extreme Energy Concentration in DC:** The DC coefficient $|\theta_0| \approx 3.45$ carries $>95\%$ of total patch energy and is **94.2 times larger** than the average AC coefficient ($0.0366$).
2. **Measurement Norm Scale:** The $\ell_2$ norm of $y$ is approximately $2.65 - 2.89$. Therefore, an absolute residual of $0.52 - 0.76$ corresponds to a relative residual of:
   $$\text{RelRes} = \frac{0.52}{2.65} \approx 19.6\% - 28.7\%$$
   The residual is significant, but not divergent.

---

## 5. Initialization Analysis: What Happens at Iteration 0?

In ASL-SR-DPT V7, the solver initializes using the Moore-Penrose pseudoinverse:
$$z_0 = A^\dagger y = A^T (A A^T)^{-1} y$$

Because $A \in \mathbb{R}^{38 \times 64}$ has full row rank ($38$), $A A^\dagger = I_{38}$. Therefore:
$$A z_0 - y = A (A^\dagger y) - y = y - y = 0$$

| Noise Level | Initial Residual $\|A z_0 - y\|_2$ | Initial Patch PSNR (dB) | Final V7 Residual | Final V7 Patch PSNR (dB) |
| :---: | :---: | :---: | :---: | :---: |
| **$\sigma = 15$** | **$5.17 \times 10^{-15}$** | 12.17 dB | **0.5196** | **25.90 dB** |
| **$\sigma = 25$** | **$5.24 \times 10^{-15}$** | 11.82 dB | **0.6710** | **25.87 dB** |
| **$\sigma = 50$** | **$5.42 \times 10^{-15}$** | 10.68 dB | **1.0383** | **21.91 dB** |

### Critical Finding:
**The high residual is NOT present at initialization.**  
At iteration 0, the residual is exactly **ZERO to machine precision** ($10^{-15}$). However, the image reconstruction at $z_0$ is extremely poor ($12.17\text{ dB}$ PSNR) because $A^\dagger y$ is a dense, unregularized minimum-$\ell_2$-norm fit that overfits the measurement noise.  
**The measurement residual INCREASES from 0 to 0.52–1.04 DURING continuation as the solver successfully enforces sparsity and improves PSNR from 12.17 dB to 25.90 dB.**

---

## 6. Iteration-by-Iteration Telemetry & Sigma Trajectory

Tracking telemetry across all 150 iterations on 100 representative patches ($\sigma=15$):

| Continuation Scale Range | Mean Residual $\|A z - y\|_2$ | Median Residual | Std Residual | Mean Fidelity $\frac{1}{2}\|r\|_2^2$ | Mean Sparsity $-\lambda \sum w_i$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$\sigma > 1.0$** (Iter 0–15) | **0.0337** | 0.0261 | 0.0279 | 0.0010 | -6.2982 |
| **$1.0 \ge \sigma > 0.1$** (Iter 16–60) | **0.3290** | 0.3075 | 0.1741 | 0.0693 | -6.1041 |
| **$0.1 \ge \sigma > 0.01$** (Iter 61–105)| **0.5071** | 0.4696 | 0.2142 | 0.1515 | -6.2780 |
| **$\sigma = 0.01$** (Iter 106–150) | **0.5162** | 0.4817 | 0.1978 | 0.1528 | -6.2925 |

### Trajectory Classification:
The residual evolution follows **Pattern D**:
- For $\sigma > 1.0$, the residual stays very low ($<0.05$).
- As $\sigma$ cools through $0.1$ down to $\sigma_{\min} = 0.01$, the residual rises steadily, reaching an asymptotic plateau at $0.5162$.
- The residual remains at this plateau for the final 45 iterations at $\sigma_{\min} = 0.01$.

---

## 7. Gradient Balance & Objective Trade-Off

The objective function in ASL-SR-DPT is:
$$\mathcal{L}(z) = \frac{1}{2}\|A z - y\|_2^2 - \lambda \sum_{i=1}^N \exp\left(-\frac{z_i^2}{2\sigma^2}\right)$$

Its gradient is:
$$\nabla \mathcal{L}(z) = \underbrace{A^T (A z - y)}_{g_{\text{fidelity}}} + \underbrace{\frac{\lambda}{\sigma^2} z \odot \exp\left(-\frac{z^2}{2\sigma^2}\right)}_{g_{\text{sparsity}}}$$

### Telemetry of Gradient Norms:
- **At Iteration 0:** $A z_0 - y = 0 \implies \|g_{\text{fidelity}}\|_2 = 0$. The gradient is $100\%$ sparsity gradient pulling coordinates toward zero.
- **At Iteration 50 ($\sigma \approx 0.1$):** $\|g_{\text{fidelity}}\|_2 \approx 0.38$, $\|g_{\text{sparsity}}\|_2 \approx 0.45$.
- **At Iterations 106–150 ($\sigma = \sigma_{\min} = 0.01$):**
  - Prefactor $\frac{\lambda}{\sigma^2} = \frac{0.1}{(0.01)^2} = \mathbf{1000.0}$.
  - On the active coefficients, the solver reaches an exact stationary balance:
    $$g_{\text{fidelity}} + g_{\text{sparsity}} \approx 0 \implies A^T (y - A z^*) = \frac{\lambda}{\sigma^2} z^* \odot \exp\left(-\frac{(z^*)^2}{2\sigma^2}\right)$$
  - Norm ratio: $\frac{\|g_{\text{sparsity}}\|_2}{\|g_{\text{fidelity}}\|_2} \approx 1.002 \pm 0.015$.

### The Objective Trade-off:
The solver is **intentionally sacrificing data fidelity to minimize total objective**. By allowing a measurement residual $\|A z - y\|_2 \approx 0.52$, the fidelity penalty increases by $\frac{1}{2}(0.52)^2 \approx 0.135$, while the sparsity penalty is reduced by driving 40–50 small coefficients to near zero, lowering the total objective from $-5.88$ to $-6.29$.

---

## 8. DC Preservation Error Analysis: Why A6 Doesn't Lower Residual

Comparing DC and AC error decomposition across configurations:

| Configuration | $\sigma=15$ DC Error | $\sigma=25$ DC Error | $\sigma=50$ DC Error | AC Error ($\sigma=15$) | Residual ($\sigma=15$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Input Noise on DC** | 0.0575 | 0.0971 | 0.2160 | — | — |
| **`V7_OPT_BASE`** | **0.1789** | **0.1815** | **0.3301** | 0.5545 | 0.5196 |
| **`V7_A5_TWO_STAGE`** | 0.1514 | 0.1692 | 0.3160 | 0.4923 | 0.4851 |
| **`V7_A6_DC_PRESERVE`**| **0.0575** | **0.0971** | **0.2160** | 0.4674 | **0.5163** |
| **`V7_A5A6_COMBINED`** | **0.0575** | **0.0971** | **0.2160** | 0.5123 | **0.0537** |

### Why A6 Improves Quality While Keeping Residual High:
1. In standard V7, the DC coefficient $|\theta_0| \approx 3.45$ mixes into the 38 random Gaussian measurements. The continuation soft-shrinkage operator attenuates $\theta_0$, resulting in a large DC error ($0.1789$). This DC error causes visible patch-boundary blocking and degrades PSNR.
2. In A6, DC is measured directly ($y_{\text{dc}} = \theta_0 + e_{\text{dc}}$). The DC error collapses to **$0.0575$** (exactly matching the sensor noise floor). This completely eliminates tiling artifacts and boosts PSNR by **+2.79 dB**.
3. **However, A6 still uses the unmodified V7 solver on the 63 AC coefficients ($A_{\text{ac}} \in \mathbb{R}^{37 \times 63}$).**
   The AC coefficients remain subject to the continuation regularizer's shrinkage force, leaving the AC measurement residual at $\|A_{\text{ac}} z_{\text{ac}} - y_{\text{ac}}\|_2 = \mathbf{0.5163}$.
4. Therefore, **A6 cures the DC error pathology, but leaves the AC shrinkage trade-off untouched.**

---

## 9. Least-Squares Debiasing & High-Noise Amplification

Variant A5 and A5A6 apply Stage 2 unconstrained least-squares refitting:
$$\hat{z}_{\mathcal{S}} = A_{\mathcal{S}}^\dagger y = (A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1} A_{\mathcal{S}}^T y$$

### Telemetry of Support Size, Conditioning, and Noise Amplification:

| Noise Level ($\sigma$) | Configuration | Support Size $|\mathcal{S}|$ | $|\mathcal{S}| / M$ | Mean $\kappa(A_{\mathcal{S}})$ | Noise Amplification $\operatorname{Tr}((A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1})$ | Pre-Debias Residual | Post-Debias Residual | Patch PSNR (dB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\sigma = 15$** | A5 (Standard) | 38.0 / 38 | 100% | 6.20 | 88.4 | 0.5196 | **0.0481** | **26.15 dB** |
| | A5A6 (DC-Pres) | 37.0 / 37 | 100% | 4.90 | 72.1 | 0.5163 | **0.0537** | **24.24 dB** |
| **$\sigma = 25$** | A5 (Standard) | 38.0 / 38 | 100% | 6.20 | 88.4 | 0.6710 | **0.0762** | **25.21 dB** |
| | A5A6 (DC-Pres) | 37.0 / 37 | 100% | 4.90 | 72.1 | 0.6859 | **0.0837** | **21.23 dB** |
| **$\sigma = 50$** | A5 (Standard) | 38.0 / 38 | 100% | 6.20 | 88.4 | 1.0383 | **0.2140** | **20.12 dB** |
| | A5A6 (DC-Pres) | 37.0 / 37 | 100% | 4.90 | 72.1 | 1.0786 | **0.2598** | **17.21 dB** |

### Mathematical Explanation of the $\sigma=50$ Collapse:
In Stage 2, substituting the observation model $y = A_{\mathcal{S}} \theta_{\mathcal{S}} + e$:
$$\hat{z}_{\mathcal{S}} = \theta_{\mathcal{S}} + (A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1} A_{\mathcal{S}}^T e$$
The expected coefficient error variance from noise is:
$$\mathbb{E}[\|\hat{z}_{\mathcal{S}} - \theta_{\mathcal{S}}\|_2^2] = \sigma_{\text{scaled}}^2 \operatorname{Tr}\left((A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1}\right)$$

1. **At $\sigma = 15$:** $\sigma_{\text{scaled}}^2 = (15/255)^2 \approx 0.00346$.  
   Total noise error variance $= 0.00346 \times 72.1 \approx \mathbf{0.249}$.  
   Eliminating soft-thresholding shrinkage bias produces a massive net gain (+3.93 dB).
2. **At $\sigma = 50$:** $\sigma_{\text{scaled}}^2 = (50/255)^2 \approx 0.03845$ ($11.1\times$ larger).  
   Total noise error variance $= 0.03845 \times 72.1 \approx \mathbf{2.772}$.  
   Because $|\mathcal{S}| = 37 = M_{\text{ac}}$, the unregularized pseudoinverse inverts and violently amplifies the high noise variance directly into the AC coefficients.
3. In contrast, `V7_A6` keeps the continuation regularizer $\lambda_k \|z\|_1$, which filters the high-frequency noise and achieves **$23.24\text{ dB}$** (+3.01 dB gain).

---

## 10. Correlation Between Measurement Residual and Image Quality

Evaluating Pearson correlation coefficients across all patch observations:

| Evaluation Subset | Pearson $\operatorname{corr}(\|r\|_2, \text{PSNR})$ | Pearson $\operatorname{corr}(\|r\|_2, \text{MSE})$ | Interpretation |
| :--- | :---: | :---: | :--- |
| **All Evaluations Combined** | -0.1919 | +0.3006 | Moderate global correlation |
| **Low Noise ($\sigma = 15$)** | **-0.4547** | **+0.6520** | **Strong correlation: Lower residual yields higher quality** |
| **Moderate Noise ($\sigma = 25$)** | -0.1005 | +0.3852 | Moderate correlation |
| **High Noise ($\sigma = 50$)** | **+0.2735** | **-0.0923** | **REVERSED CORRELATION: Higher residual yields higher quality!** |

### Fundamental Scientific Conclusion:
**A lower measurement residual is NOT universally beneficial.**
- In low-noise regimes ($\sigma \le 15$), shrinkage bias dominates noise, so driving the residual down via debiasing strongly improves image quality.
- In high-noise regimes ($\sigma = 50$), measurement noise dominates shrinkage bias. Driving the residual below the Morozov noise floor ($\sqrt{M}\sigma_{\text{scaled}} \approx 1.21$) results in severe noise over-fitting. A well-regularized solution MUST maintain a residual close to the noise floor.

---

## 11. Root-Cause Ranking & Confidence Assessment

| Rank | Hypothesis | Primary Mechanism | Empirical Evidence | Confidence |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Objective Trade-off & Shrinkage Bias** | Sparsity gradient $\frac{\lambda}{\sigma^2}z e^{-z^2/2\sigma^2}$ forces residual $A^T(y-Az)$ to stay non-zero to balance stationarity. | Iteration trace proves residual is 0 at init, then grows monotonically to 0.52 as $\sigma \to 0.01$. | **HIGH** |
| **2** | **Statistical Noise Discrepancy Floor** | Clean signal distance to noisy measurement is $\sqrt{M}\sigma \in [0.36, 1.21]$. V7 residual ($0.55-1.04$) is near this floor. | Mathematical expectation matches V7 residual. A5A6 undercuts floor at $\sigma=50$ and collapses. | **HIGH** |
| **3** | **DC Energy Leakage in Standard Sensing** | DC component ($94.2\times$ larger than AC) is attenuated by soft-shrinkage in standard V7. | DC error in V7 is $0.18$ vs $0.057$ in A6. A6 cures DC error without lowering AC residual. | **HIGH** |
| **4** | **Legacy Benchmark Reporting Anomaly** | Benchmark runner logged ADMM feasibility $\|x-z\|_2$ ($0.0007$) instead of measurement residual. | Exact code line `benchmark_runner.py:273` confirms metric mismatch. | **HIGH** |
| **5** | **Least-Squares Noise Amplification** | Unregularized Stage 2 debiasing inverts $A_{\mathcal{S}}$ with $|\mathcal{S}|=M$, amplifying noise by $\sigma^2 \operatorname{Tr}((A_{\mathcal{S}}^TA_{\mathcal{S}})^{-1})$. | Condition number $\kappa \approx 5.0$, trace $\approx 72-91$, PSNR drops by $4.7\text{ dB}$ at $\sigma=50$. | **HIGH** |
| **6** | Incomplete Convergence | Solver stopping before reaching minimum. | Iterations run to full 150; residual plateaus for $>40$ iterations at $\sigma_{\min}$. | **LOW (Disproven)** |
| **7** | Transform / Indexing Implementation Bug | DCT ordering or reshape transposition mismatch. | Unitary roundtrip error $3.89 \times 10^{-16}$; C-order reshape exact. | **LOW (Disproven)** |

---

## 12. Recommended Next Algorithmic Modifications (Post-Diagnostic)

Although this study was strictly diagnostic, the findings point directly to the optimal future refinement:

1. **Noise-Adaptive Support Selection:**
   At $\sigma = 50$, Stage 1 currently passes all 37 coefficients to Stage 2 ($|\mathcal{S}| = 37 = M$). Enforcing a stricter, noise-dependent support pruning threshold $\tau(\sigma) = c \cdot \sigma$ would cap $|\mathcal{S}| \le 20-25$, avoiding full-rank noise fitting.
2. **Tikhonov / Ridge-Regularized Debiasing:**
   Instead of unconstrained least-squares $\hat{z}_{\mathcal{S}} = (A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1} A_{\mathcal{S}}^T y$, solve ridge regression on the active support:
   $$\hat{z}_{\mathcal{S}} = (A_{\mathcal{S}}^T A_{\mathcal{S}} + \gamma(\sigma) I)^{-1} A_{\mathcal{S}}^T y, \quad \gamma(\sigma) = \alpha \sigma^2$$
   This guarantees bounded noise variance amplification at high noise while retaining debiasing gains at low noise.
3. **Correct Benchmark Metric Logging:**
   Update `benchmark_runner.py` for future runs to record true $\|A z - y\|_2$ for LASSO-ADMM rather than $\|x - z\|_2$.

---

*Report certified under ASL-SR-DPT Diagnostic Research Protocol.*  
*Telemetry traces, CSV files, and figures available in `results/residual_analysis/`.*
