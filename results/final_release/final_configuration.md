# ASL-SR-DPT A6 Final Configuration Specification

**Repository:** ASL-SR-DPT (*Accelerated Single-Loop Sparse Recovery with Dynamic Parameter Tuning for Mobile Image Denoising*)  
**Release:** Final Candidate Freeze (`A6_FINAL_FROZEN`)  
**Date:** September 8, 2026  
**Artifact Path:** `results/final_release/final_configuration.md`  

---

## 1. Algorithmic Parameterization

The following table records the exact, immutable parameterization of the certified final solver `V7_A6_DC_PRESERVATION`:

| Parameter | Symbol / Identifier | Final Value | Scientific Role & Justification |
| :--- | :--- | :---: | :--- |
| **Regularization Weight** | $\lambda_{\text{reg}}$ (`lambda_reg`) | `0.1` | Trades measurement fidelity against nonconvex sparsity penalty $\phi(z; \sigma)$. |
| **Initial Continuation Scale** | $\sigma_{\text{init}}$ (`sigma_init`) | `null` (Dynamic) | Computed dynamically per patch as $\sigma_0 = \frac{1}{2} \|A_{\text{ac}}^T y_{\text{ac}}\|_\infty$, ensuring smooth convex initialization. |
| **Minimum Continuation Scale** | $\sigma_{\min}$ (`sigma_min`) | `0.01` | Final scale parameter of the smooth approximation to the $\ell_0$ norm. |
| **Geometric Continuation Decay** | $\gamma$ (`sigma_decay`) | `0.95` | Multiplicative decay schedule: $\sigma_{k+1} = \max(\gamma \sigma_k, \sigma_{\min})$. |
| **Maximum Iterations** | $K_{\max}$ (`max_iter`) | `150` | Continuation reaches $\sigma_{\min} = 0.01$ at iteration $k \approx 98$; early stopping triggers when stationary. |
| **Gradient Convergence Tolerance** | $\tau$ (`tol`) | `1e-5` | Stopping threshold for relative gradient norm on the active support. |
| **Initial Line Search Step Size** | $\mu_0$ (`initial_mu`) | `0.2` | Initial trial step along the negative gradient direction. |
| **Armijo Sufficient Decrease Parameter** | $c_1$ (`armijo_c`) | `1e-4` | Standard Armijo condition constant: $f(z + \alpha d) \le f(z) + c_1 \alpha \nabla f(z)^T d$. |
| **Backtracking Contraction Factor** | $\beta$ (`beta_decay`) | `0.5` | Step size contraction factor during line search backtracking: $\mu \leftarrow \beta \mu$. |
| **Max Backtracking Steps** | `max_backtracks` | `10` | Maximum line search step reductions per iteration. |
| **Support Screening Multiplier** | `support_threshold_multiplier` | `1e-5` | Threshold for active variable screening: variable $j$ active if $|z_j| > 10^{-5} \sigma_k$. |
| **Support Reopening Interval** | `support_reopen_interval` | `3` | Periodically evaluates gradient across inactive coordinates every 3 iterations to prevent false lock-out. |
| **Midpoint Continuation Scheme** | `use_midpoint` | `true` | Evaluates gradient at midpoint update for enhanced continuation stability. |
| **Execution Thread Budget** | `threads` | `1` | Strictly single-threaded execution per solver instance for benchmark reproducibility. |

---

## 2. Compressive Sensing Dimensions & Decoupled Architecture

| Dimension / Property | Symbol | Value | Architectural Specification |
| :--- | :--- | :---: | :--- |
| **Patch Spatial Size** | $p \times p$ (`patch_size`) | $8 \times 8$ | Standard $8 \times 8$ image patch (64 pixels). |
| **Patch Extraction Stride** | $s$ (`stride`) | `2` | Dense sliding window stride across both horizontal and vertical spatial axes. |
| **Ambient Dimension** | $N$ | `64` | Total 2D-DCT orthonormal transform basis atoms ($8 \times 8$). |
| **Total Measurement Budget** | $M$ | `38` | Sampling ratio $SR = 38/64 = 0.59375$ (~60% compressive sensing rate). |
| **DC Measurement Allocation** | $M_{\text{DC}}$ | `1` | Single scalar observation dedicated losslessly to the DC coefficient $\theta_0$. |
| **AC Measurement Dimension** | $M_{\text{AC}}$ (`M_ac`) | `37` | Compressive measurements dedicated strictly to the 63 AC transform coefficients. |
| **AC Signal Dimension** | $N_{\text{AC}}$ | `63` | Vector of AC transform coefficients $\theta_{1:63} \in \mathbb{R}^{63}$. |
| **DC Handling Protocol** | `dc_handling` | Direct Lossless | $y_{\text{dc}} = \theta_0$; restored directly without iterative optimization as $\hat{\theta}_0 = y_{\text{dc}}$. |
| **AC Measurement Operator** | $A_{\text{ac}}$ (`A_dc`) | Matrix $37 \times 63$ | Row-normalized Gaussian random sensing matrix: $\|a_{i,:}\|_2 = 1 \ \forall i \in \{1, \dots, 37\}$. |
| **AC Measurement Rule** | `y_ac` | Linear Projection | $y_{\text{ac}} = A_{\text{ac}} \theta_{1:63} \in \mathbb{R}^{37}$. |

---

## 3. Configuration Cryptographic Hash

- **File Path:** `configs/final_config.json`
- **SHA-256 Digest:** `d3619add2942fe04ea5eefccfd7c5570b2eae6a9156eec823acf0522b3c4346f`
- **Freeze Status:** Locked and verified.
