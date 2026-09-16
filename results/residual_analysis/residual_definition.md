# ASL-SR-DPT Measurement Residual Implementation & Definition Verification

**Document:** `results/residual_analysis/residual_definition.md`  
**Author:** Carlo Mendoza  
**Date:** September 8, 2026  
**Status:** Certified Audit  

---

## 1. Mathematical Definitions of Residual Metrics

In compressed sensing and sparse recovery problems:
$$y = A \theta + e, \quad y \in \mathbb{R}^M, \; A \in \mathbb{R}^{M \times N}, \; \theta \in \mathbb{R}^N$$

Given an estimated coefficient vector $\hat{z} \in \mathbb{R}^N$, the measurement discrepancy vector is:
$$r = A \hat{z} - y \in \mathbb{R}^M$$

We formally distinguish three separate residual metrics:

### 1.1 Absolute Measurement Residual Norm
$$\|r\|_2 = \|A \hat{z} - y\|_2 = \sqrt{\sum_{j=1}^M ( (A \hat{z})_j - y_j )^2}$$
- **Units:** Measurement units (unscaled).
- **Properties:** Dependent on the scale of $y$ and the row dimensionality $M$.

### 1.2 Relative Measurement Residual Norm
$$\text{RelRes}(\hat{z}, y) = \frac{\|A \hat{z} - y\|_2}{\max(\|y\|_2, \epsilon)}$$
- **Units:** Dimensionless ratio.
- **Properties:** Invariant to global scaling of $y$. A value of $1.0$ indicates that the reconstructed measurement energy has residual error equal to the entire signal measurement norm (e.g. if $\hat{z} = 0$, $\|A \hat{z} - y\|_2 / \|y\|_2 = 1.0$).

### 1.3 Normalized Root Mean Square Error (Normalized RMSE)
$$\text{NRMSE}_{\text{meas}}(\hat{z}, y) = \frac{\|A \hat{z} - y\|_2}{\sqrt{M}} = \sqrt{\frac{1}{M} \sum_{j=1}^M ((A \hat{z})_j - y_j)^2}$$
- **Units:** Measurement error standard deviation per measurement dimension.
- **Properties:** Invariant to the number of measurements $M$. Enables fair comparison between $M=38$ (standard sensing) and $M_{\text{ac}}=37$ (DC-preserving sensing).

### 1.4 DC-Preserving Residual Separation
For DC-preserving sensing, the measurement vector is partitioned into the scalar DC observation $y_{\text{dc}} \in \mathbb{R}^1$ and the 37 AC projections $y_{\text{ac}} \in \mathbb{R}^{37}$:
- **AC Absolute Residual:** $r_{\text{ac}} = \|A_{\text{ac}} \hat{z}_{\text{ac}} - y_{\text{ac}}\|_2$
- **AC Relative Residual:** $\text{RelRes}_{\text{ac}} = \frac{\|A_{\text{ac}} \hat{z}_{\text{ac}} - y_{\text{ac}}\|_2}{\max(\|y_{\text{ac}}\|_2, \epsilon)}$
- **AC Normalized RMSE:** $\text{NRMSE}_{\text{ac}} = \frac{\|A_{\text{ac}} \hat{z}_{\text{ac}} - y_{\text{ac}}\|_2}{\sqrt{37}}$
- **DC Measurement Error:** $\text{DC}_{\text{err}} = |\hat{z}_0 - y_{\text{dc}}|$

> **Protocol Rule:** The DC scalar measurement error MUST NOT be combined into the Euclidean norm of $r_{\text{ac}}$. They reside in distinct dimensional spaces and have different physical interpretations.

---

## 2. Codebase Implementation Audit

We inspected every file in `code/` responsible for computing, tracking, and logging residuals. Below is the line-by-line trace.

### 2.1 Solver-Level Computation: `code/hybrid_sparse_solver_v7_fixed.py`
In the fixed baseline solver:
- **Lines 385–390:**
  ```python
  residual = self.A @ z - y
  res_norm = float(np.linalg.norm(residual))
  t_residual_total += time.perf_counter() - t_s
  diagnostics["final_residual"] = res_norm
  ```
- **Determination:** The solver calculates the **ABSOLUTE EUCLIDEAN NORM** $\|A z - y\|_2$. It does NOT divide by $\|y\|_2$ and does NOT divide by $\sqrt{M}$.

### 2.2 Optimized Solver: `code/hybrid_sparse_solver_v7_optimized.py`
In the optimized batch solver:
- **Lines 312–313:**
  ```python
  final_residual_norm = float(np.linalg.norm(self.A @ z - y))
  diagnostics["final_residual"] = final_residual_norm
  ```
- **Determination:** Identical to the fixed baseline. Computes the **ABSOLUTE EUCLIDEAN NORM** $\|A z - y\|_2$.

### 2.3 Benchmark Runner: `code/benchmark_runner.py`
In the benchmark evaluation loop:
- **Line 72:** `"residual"` is declared in `RAW_CSV_FIELDS`.
- **Line 143:** `residuals_list = []`
- **Lines 198–199 (ASL-SR-DPT):**
  ```python
  residuals_list.append(diag["final_residual"])
  ```
- **Lines 233–234 (OMP):**
  ```python
  residuals_list.append(diag["final_residual"])
  ```
- **Lines 272–273 (LASSO-ADMM) — CRITICAL ANOMALY IDENTIFIED:**
  ```python
  residuals_list.append(diag["final_primal_residual"])
  ```
- **Line 294:**
  ```python
  mean_residual = float(np.mean(residuals_list))
  ```
- **Line 306:**
  ```python
  "residual": mean_residual,
  ```

### 2.4 Candidate Validation Runner: `code/run_final_candidate_validation.py`
In the candidate validation driver:
- **Lines 184–193:**
  ```python
  if config_name in ["V7_OPT_BASE", "V7_A5_TWO_STAGE"]:
      residuals = np.linalg.norm(Z_final @ A_std.T - Y_all.T, axis=1)
  else:
      residuals = np.linalg.norm(Z_final[:, 1:] @ A_dc.T - Y_all.T, axis=1)
  mean_residual = float(np.mean(residuals))
  ```
- **Determination:** Vectorized batch computation of the **ABSOLUTE EUCLIDEAN NORM** $\|A z_i - y_i\|_2$ per patch, followed by taking the arithmetic mean across all $P = 37,604$ patches in the image.

---

## 3. Definitive Classification of the Current "Residual" Metric

Based on the direct code audit:
1. **Is it absolute, relative, or normalized?**
   It is strictly the **ABSOLUTE** Euclidean norm:
   $$\text{reported\_residual} = \|A \hat{z} - y\|_2$$
2. **How is it aggregated across an image?**
   It is the **MEAN OF PATCH NORMS**:
   $$\text{reported\_residual}_{\text{image}} = \frac{1}{P} \sum_{i=1}^P \|A \hat{z}_i - y_i\|_2$$
   It is NOT the image-level norm $\|A_{\text{block}} Z - Y\|_F$, and it is NOT summed across patches.

---

## 4. Critical Anomaly Discovered: LASSO-ADMM Residual Discrepancy

In all prior benchmark reports (e.g., `results/summaries/E2_standard_summary.csv` and `E3_dc_summary.csv`), the reported residual for LASSO-ADMM was recorded as **$0.0007$** or **$0.0001$**.

The code audit reveals why:
- In `code/hybrid_sparse_solver_v7_fixed.py` line 599:
  ```python
  r_primal = float(np.linalg.norm(x - z))
  diagnostics["final_primal_residual"] = r_primal
  ```
- In `code/benchmark_runner.py` line 273:
  ```python
  residuals_list.append(diag["final_primal_residual"])
  ```
**Root Cause of the Apparent Discrepancy:**  
`diag["final_primal_residual"]` in ADMM is the **primal constraint violation** $\|x - z\|_2$ between the split variables $x$ and $z$!  
It was **NOT** the measurement residual $\|A z - y\|_2$!  
Because ADMM was configured with tolerance $\text{tol} = 10^{-4}$, the solver stopped when $\|x - z\|_2 \approx 10^{-4}$.  
Consequently, comparing ASL-SR-DPT's measurement residual ($\|A z - y\|_2 \approx 0.62 - 0.76$) against LASSO-ADMM's reported $0.0007$ was an **apples-to-oranges comparison** created by logging ADMM's internal splitting convergence metric rather than its measurement residual.

For OMP, OMP directly solves $\min_{\theta_{\mathcal{S}}} \|y - A_{\mathcal{S}} \theta_{\mathcal{S}}\|_2$ via unconstrained least squares. When OMP reaches 38 selected atoms, $A_{\mathcal{S}}$ is a full-rank $38 \times 38$ square matrix, forcing the measurement residual $\|y - A_{\mathcal{S}} \theta_{\mathcal{S}}\|_2$ to reach exactly $0.0$.
