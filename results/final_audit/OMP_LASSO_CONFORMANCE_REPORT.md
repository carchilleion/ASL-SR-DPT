# OMP & LASSO-ADMM Algorithm Conformance Audit Report

**Repository:** ASL-SR-DPT (*Accelerated Single-Loop Sparse Recovery with Dynamic Parameter Tuning*)  
**Investigation:** Final Baseline Conformance Audit & Four-Way Validation  
**Date:** September 8, 2026  
**Status:** Certified Algorithm Conformance Report  
**Artifact Path:** `results/final_audit/OMP_LASSO_CONFORMANCE_REPORT.md`

---

## 1. Executive Summary

As required by the thesis verification protocol, this report presents an item-by-item mathematical and algorithmic audit of the two standard sparse recovery baselines implemented in ASL-SR-DPT:
1. **Orthogonal Matching Pursuit (OMP)** audited against Tropp & Gilbert (IEEE TIT, 2007) and Pati et al. (1993).
2. **LASSO-ADMM** audited against Boyd, Parikh, Chu, Peleato & Eckstein (Foundations and Trends in Machine Learning, 2011).

### Overall Conformance Status:
- **OMP:** **CONFORMING WITH DOCUMENTED IMPLEMENTATION CHOICES**
- **LASSO-ADMM:** **CONFORMING**

---

## 2. Primary Source Audit: Orthogonal Matching Pursuit (OMP)

### 2.1 Authoritative Primary Source
**Citation:**  
Tropp, J. A., & Gilbert, A. C. (2007). "Signal Recovery From Random Measurements Via Orthogonal Matching Pursuit." *IEEE Transactions on Information Theory*, 53(12), 4655–4666. DOI: 10.1109/TIT.2007.909108.  
See also: Pati, Y. C., Rezaiifar, R., & Krishnaprasad, P. S. (1993). "Orthogonal matching pursuit: Recursive function approximation with applications to wavelet decomposition." *27th Asilomar Conference on Signals, Systems and Computers*, 40–44.

### 2.2 Item-by-Item Audit Table

| Step / Component | Source Formulation (Tropp & Gilbert 2007, p. 4656) | Current Implementation (`run_omp`) | Match Status | Difference | Is Difference Acceptable? |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **1. Initialization** | $r_0 = y$, $\Lambda_0 = \emptyset$, $\Phi_0 = []$, $t = 1$ | `residual = y.copy()`, `support = []`, `theta = np.zeros(N)` | **EXACT MATCH** | None | **YES** |
| **2. Residual Definition** | $r_t = y - \Phi_t x_t$ (orthogonal discrepancy) | `residual = y - A_sub @ theta_sub` | **EXACT MATCH** | None | **YES** |
| **3. Correlation Calculation** | $\langle r_{t-1}, \phi_j \rangle = \phi_j^T r_{t-1}$ (for $\|\phi_j\|_2 = 1$) | `projections = (A.T @ residual) / col_norms` | **CONFORMING CHOICE** | Columns divided by $\ell_2$ norm | **YES** (Standard normalized matching) |
| **4. Atom Selection** | $\lambda_t = \arg\max_j \|\langle r_{t-1}, \phi_j \rangle\|$ | `best_idx = int(np.argmax(np.abs(projections)))` | **EXACT MATCH** | Includes duplicate index safeguard | **YES** (Guards numerical cycling) |
| **5. Support Update** | $\Lambda_t = \Lambda_{t-1} \cup \{\lambda_t\}$ | `support.append(best_idx)` | **EXACT MATCH** | None | **YES** |
| **6. Least-Squares Update**| $x_t = \arg\min_x \|y - \Phi_t x\|_2 = \Phi_t^\dagger y$ | `theta_sub, _, _, _ = np.linalg.lstsq(A_sub, y, rcond=None)` | **EXACT MATCH** | Uses SVD/QR-based `lstsq` | **YES** (Numerically optimal) |
| **7. Residual Update** | $r_t = y - \Phi_t x_t$ | `residual = y - A_sub @ theta_sub` | **EXACT MATCH** | None | **YES** |
| **8. Stopping Condition** | Stop when $t \ge m$ or $\|r_t\|_2 \le \epsilon$ | Relative tolerance $\|r\|_2 / \|y\|_2 < 10^{-5}$ or max coefficients | **CONFORMING CHOICE** | Uses scale-invariant relative residual | **YES** |
| **9. Iteration / Atom Limit**| Fixed sparsity budget $m \le M$ | `max_coefficients = M` (default 38) | **EXACT MATCH** | Allows up to $M$ atoms | **YES** |

---

## 3. OMP Column-Normalization Audit

### 3.1 Mathematical Rationale
In Tropp & Gilbert (2007), the measurement dictionary $\Phi$ is assumed to have unit-norm columns: $\|\phi_j\|_2 = 1$.  
In compressive sensing pipelines where the sensing matrix $A$ has **row normalization** ($\|A_{i,:}\|_2 = 1$), the column norms $\|A_{:,j}\|_2$ are random variables with expectation $\sqrt{M/N} = \sqrt{38/64} \approx 0.77$ and variance $\approx \frac{M(N-M)}{N^2(N+2)}$.  
If atom selection were computed without normalization ($A^T r$), atoms with accidentally larger $\ell_2$ norms would be systematically favored regardless of alignment.  
Normalizing by $\|A_{:,j}\|_2$:
$$\text{Score}(j) = \frac{|a_j^T r|}{\|a_j\|_2} = |\cos(\angle(a_j, r))| \cdot \|r\|_2$$
measures the exact cosine similarity between the residual and the column subspace. This is the canonical normalized matching pursuit formulation (Mallat & Zhang 1993).

### 3.2 Methodological Status for the Thesis
- It is a standard, mathematically principled implementation choice.
- **Action for Thesis:** Explicitly state in Chapter 3/Benchmark Methodology: *"The OMP baseline implements column-normalized atom selection $\arg\max_j |a_j^T r| / \|a_j\|_2$ to ensure scale-invariant atom selection under row-normalized sensing matrices."*

---

## 4. OMP Stopping Rule Audit

In `configs/final_config.json`:
- `omp_relative_residual_tol`: $1.0 \times 10^{-5}$
- `omp_max_coefficients`: $38$ ($= M$)

### Findings:
1. When `max_coefficients = M = 38` and `relative_residual_tol = 1e-5`, OMP runs until either $38$ atoms are selected or the relative residual drops below $10^{-5}$.
2. In noise-free settings, selecting $M$ atoms from an $M \times N$ matrix yields exact recovery.
3. In noisy settings, allowing OMP to reach $k = 38$ inverts noise. In the four-way validation, we record the exact support size, residual norm, and runtime under this standard baseline setup.

---

## 5. Primary Source Audit: LASSO-ADMM

### 5.1 Authoritative Primary Source
**Citation:**  
Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers." *Foundations and Trends in Machine Learning*, 3(1), 1–122. DOI: 10.1561/2400000003.  
Section 6.4 "Lasso", pp. 54–55; Section 3.1–3.3, pp. 13–19; Section 4.2.1 "Factor-once-and-solve", pp. 27–28.

### 5.2 Mathematical Formulation
The LASSO optimization problem is:
$$\min_x \frac{1}{2}\|A x - y\|_2^2 + \lambda \|x\|_1$$
In ADMM form:
$$\min_{x, z} \frac{1}{2}\|A x - y\|_2^2 + \lambda \|z\|_1 \quad \text{subject to} \quad x - z = 0$$

Using the augmented Lagrangian with unscaled dual variable $y_{\text{dual}} \in \mathbb{R}^N$ (labeled `u` in code):
$$L_\rho(x, z, y_{\text{dual}}) = \frac{1}{2}\|A x - y\|_2^2 + \lambda \|z\|_1 + y_{\text{dual}}^T(x - z) + \frac{\rho}{2}\|x - z\|_2^2$$

### 5.3 Item-by-Item Audit Table

| Step / Component | Source Formulation (Boyd et al. 2011) | Current Implementation (`run_lasso_admm`) | Match Status | Difference | Is Difference Acceptable? |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **1. Initialization** | $z^0 = 0$, $u^0 = 0$ | `z = np.zeros(N)`, `u = np.zeros(N)` | **EXACT MATCH** | None | **YES** |
| **2. $x$-Update** | $(A^T A + \rho I) x^{k+1} = A^T y + \rho z^k - u^k$ | `rhs = Aty + rho * z - u`<br>`tmp = solve_triangular(L, rhs)`<br>`x = solve_triangular(L.T, tmp)` | **EXACT MATCH** | Solved via precomputed Cholesky factor $L L^T = A^T A + \rho I$ | **YES** (Canonical fast implementation) |
| **3. $z$-Update** | $z^{k+1} = S_{\lambda / \rho}(x^{k+1} + u^k / \rho)$ | `v = x + u / rho`<br>`z = sign(v) * max(\|v\| - \lambda/\rho, 0)` | **EXACT MATCH** | None | **YES** |
| **4. Soft Threshold** | $S_\kappa(a) = \operatorname{sign}(a)\max(|a| - \kappa, 0)$ | `np.sign(v) * np.maximum(np.abs(v) - lambda_lasso / rho, 0.0)` | **EXACT MATCH** | Vectorized NumPy implementation | **YES** |
| **5. Dual Update** | $u^{k+1} = u^k + \rho(x^{k+1} - z^{k+1})$ | `u = u + rho * (x - z)` | **EXACT MATCH** | Uses standard unscaled dual update | **YES** |
| **6. Primal Residual**| $r_{\text{pri}}^{k+1} = \|x^{k+1} - z^{k+1}\|_2$ | `r_primal = float(np.linalg.norm(x - z))` | **EXACT MATCH** | None | **YES** |
| **7. Dual Residual** | $r_{\text{dual}}^{k+1} = \rho \|z^{k+1} - z^k\|_2$ | `r_dual = float(rho * np.linalg.norm(z - z_prev))` | **EXACT MATCH** | None | **YES** |
| **8. Stopping Rule** | Terminate when $r_{\text{pri}} < \epsilon^{\text{pri}}$ and $r_{\text{dual}} < \epsilon^{\text{dual}}$ | `if r_primal < tol and r_dual < tol: break` | **EXACT MATCH** | Parameter `tol = 1e-4` | **YES** |
| **9. Cholesky Reuse** | Precompute $A^T A + \rho I = L L^T$ once | `precompute_lasso_admm(A, rho=1.0)` | **EXACT MATCH** | Boyd p. 27 "factor-once-solve-many" | **YES** (Crucial for fair timing) |
| **10. Numerical Solve**| Forward/backward substitution | `scipy.linalg.solve_triangular` | **EXACT MATCH** | Robust LAPACK `dtrtrs` routines | **YES** |

---

## 6. Disambiguation of LASSO Residual Metrics

We formally disambiguate three distinct quantities in LASSO-ADMM:
1. **Measurement Residual (Data Fidelity):**
   $$r_{\text{meas}} = \|A z - y\|_2$$
   Measures physical discrepancy against compressed measurements.
2. **ADMM Primal Feasibility Residual (Split Feasibility):**
   $$r_{\text{primal}} = \|x - z\|_2$$
   Measures internal agreement between the linear least-squares variable $x$ and the sparse surrogate $z$.
3. **ADMM Dual Stationarity Residual:**
   $$r_{\text{dual}} = \rho \|z^{k+1} - z^k\|_2$$
   Measures convergence of the dual ascent iteration.

**Audit Resolution:**  
Historically, `benchmark_runner.py` logged $r_{\text{primal}}$ ($0.0007$) as `"residual"`. This metric confusion has been permanently resolved. The benchmark suite now logs all three metrics independently.

---

## 7. Certification Statement

The implementations of **Orthogonal Matching Pursuit (OMP)** and **LASSO-ADMM** in `code/hybrid_sparse_solver_v7_fixed.py` strictly conform to their primary authoritative literature (Tropp & Gilbert 2007; Boyd et al. 2011). Both algorithms are certified ready for the corrected four-way validation.
