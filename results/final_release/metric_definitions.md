# ASL-SR-DPT Benchmark Metric Definitions & Audited Formulations

**Repository:** ASL-SR-DPT (*Accelerated Single-Loop Sparse Recovery with Dynamic Parameter Tuning for Mobile Image Denoising*)  
**Release:** Final Candidate Freeze (`A6_FINAL_FROZEN`)  
**Date:** September 8, 2026  
**Artifact Path:** `results/final_release/metric_definitions.md`  

---

## 1. Image-Domain Reconstruction Quality Metrics

All spatial metrics are computed on double-precision normalized images $X, \hat{X} \in [0.0, 1.0]^{H \times W}$.

### 1.1 Mean Squared Error (MSE)
$$\operatorname{MSE}(X, \hat{X}) = \frac{1}{H \cdot W} \sum_{h=1}^H \sum_{w=1}^W \left( X(h, w) - \hat{X}(h, w) \right)^2$$
- **Implementation:** `np.mean((clean_image - reconstructed_image) ** 2)`
- **Dynamic Range:** $[0, 1]$. Optimal value is $0$.

### 1.2 Peak Signal-to-Noise Ratio (PSNR)
$$\operatorname{PSNR}(X, \hat{X}) = 10 \log_{10} \left( \frac{\operatorname{MAX}_I^2}{\operatorname{MSE}(X, \hat{X})} \right) \text{ dB}$$
- **Peak Dynamic Range:** $\operatorname{MAX}_I = 1.0$ (for normalized float arrays).
- **Implementation:** `10.0 * np.log10(1.0 / max(mse, 1e-12))`
- **Unit:** Decibels (dB). Higher is strictly better.

### 1.3 Structural Similarity Index Measure (SSIM)
$$\operatorname{SSIM}(X, \hat{X}) = \frac{(2 \mu_X \mu_{\hat{X}} + C_1)(2 \sigma_{X\hat{X}} + C_2)}{(\mu_X^2 + \mu_{\hat{X}}^2 + C_1)(\sigma_X^2 + \sigma_{\hat{X}}^2 + C_2)}$$
- **Implementation:** `skimage.metrics.structural_similarity(clean, rec, data_range=1.0)`
- **Constants:** $C_1 = (0.01 \cdot 1.0)^2 = 0.0001$, $C_2 = (0.03 \cdot 1.0)^2 = 0.0009$.
- **Fairness Guarantee:** Identical Gaussian weighting window ($11 \times 11$, $\sigma_G = 1.5$) used across all four algorithms.

---

## 2. Compressive Measurement Residual Definitions

To prevent mathematical ambiguity across compressive sensing architectures, three scale-independent residual formulations are logged:

### 2.1 Absolute Measurement Residual ($r_{\text{meas}}$)
$$r_{\text{meas}} = \|A \hat{\theta} - y\|_2 = \sqrt{\sum_{m=1}^M (a_m^T \hat{\theta} - y_m)^2}$$
- **For Standard Sensing (`V7_OPT_BASE`, `OMP`, `LASSO-ADMM`):** $A \in \mathbb{R}^{38 \times 64}$, $y \in \mathbb{R}^{38}$.
- **For DC-Preserving Sensing (`V7_A6_DC_PRESERVATION`):** Computed strictly over the AC measurement subspace:
  $$r_{\text{meas, ac}} = \|A_{\text{ac}} \hat{\theta}_{\text{ac}} - y_{\text{ac}}\|_2, \quad A_{\text{ac}} \in \mathbb{R}^{37 \times 63}, \quad y_{\text{ac}} \in \mathbb{R}^{37}$$

### 2.2 Relative Measurement Residual ($r_{\text{rel}}$)
$$r_{\text{rel}} = \frac{\|A \hat{\theta} - y\|_2}{\|y\|_2}$$
- **Role:** Eliminates dependency on the absolute signal power $\|y\|_2$.
- **Safeguard:** $\|y\|_2$ denominator clamped to $\max(\|y\|_2, 10^{-8})$ to avoid zero division.

### 2.3 Normalized Measurement Residual ($r_{\text{norm}}$)
$$r_{\text{norm}} = \frac{\|A \hat{\theta} - y\|_2}{\sqrt{M}}$$
- **Role:** Measures the Root Mean Square (RMS) discrepancy per measurement dimension, facilitating comparison between $M = 38$ (standard) and $M_{\text{ac}} = 37$ (A6).

---

## 3. ADMM-Specific Convergence Residuals (LASSO Only)

In convex optimization via ADMM splitting for $\min_x \frac{1}{2}\|Ax-y\|_2^2 + \lambda \|z\|_1 \text{ s.t. } x - z = 0$:

### 3.1 ADMM Primal Residual ($r_{\text{primal}}$)
$$r_{\text{primal}} = \|x^{k+1} - z^{k+1}\|_2$$
- **Mathematical Meaning:** Measures constraint violation / feasibility between the auxiliary variable $x$ and sparse variable $z$.
- **Strict Prohibition:** $r_{\text{primal}}$ must **NEVER** be reported as or conflated with measurement residual $\|Az - y\|_2$.

### 3.2 ADMM Dual Residual ($r_{\text{dual}}$)
$$r_{\text{dual}} = \rho \|z^{k+1} - z^k\|_2$$
- **Mathematical Meaning:** Measures stationarity of the dual optimality condition.
- **Strict Prohibition:** $r_{\text{dual}}$ must **NEVER** be reported as or conflated with measurement residual $\|Az - y\|_2$.

---

## 4. Computational Performance Metrics

### 4.1 Setup Time ($t_{\text{setup}}$)
$$t_{\text{setup}} = \text{time.perf\_counter()}_{\text{end}} - \text{time.perf\_counter()}_{\text{start}}$$
- **Scope:** One-time matrix precomputations for a fixed sensing matrix:
  - For LASSO-ADMM: Cholesky factor $L$ of $A^T A + \rho I$.
  - For OMP: Column $\ell_2$ norms of $A$.
  - For V7/A6: Sensing operator initialization.

### 4.2 Solve Time ($t_{\text{solve}}$)
$$t_{\text{solve}} = \sum_{i=1}^{N_p} \Delta t_i$$
- **Scope:** Strictly the elapsed CPU monotonic time spent within the iterative solver routines across all $N_p$ patches of an image.
- **Exclusions:** Excludes patch extraction, transform, spatial IDCT, Hamming synthesis, disk I/O, and metric logging.

### 4.3 Throughput per Patch ($t_{\text{patch}}$)
$$t_{\text{patch}} = \left( \frac{t_{\text{solve}}}{N_p} \right) \times 1000 \text{ ms/patch}$$
