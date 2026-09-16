# Theoretical Operation-Count Estimate: ASL-SR-DPT (Fixed V7)

**Target Dimensions:** $M = 38$ measurements, $N = 64$ coefficients ($8 \times 8$ patch)  
**Algorithmic Parameters:** $\text{max\_iter} = 150$, $\text{max\_backtracks} = 10$, $\text{patch\_count} = 37{,}604$  
**Nature of Metric:** Theoretical operation-count estimate (not directly measured hardware performance counters).

---

## 1. Matrix-Vector Products per Iteration

In the current implementation of `HybridSparseSolverV7`, matrix-vector multiplications represent the dominant floating-point cost:

| Component | Mathematical Operation | Dimension | Matrix Products per Call | Typical Calls per Iteration | Matrix Products per Iteration |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Objective Evaluation** | $A z - y$ | $38 \times 64$ | $1$ | $1$ | $1$ |
| **Gradient Evaluation** | $A_{\text{active}} z_{\text{active}} - y$ | $38 \times |S|$ | $1$ | $1$ | $1$ |
| **Gradient Projection** | $A_{\text{active}}^T r$ | $|S| \times 38$ | $1$ | $1$ | $1$ |
| **Armijo Candidates** | $A z_{\text{cand}} - y$ | $38 \times 64$ | $1$ | $B \in [1, 10]$ | $B$ |
| **Midpoint Candidates** | $A z_{\text{mid}} - y$ | $38 \times 64$ | $1$ | $B_{\text{mid}} \in [0, 10]$ | $B_{\text{mid}}$ |
| **Total per Iteration** | | | | | **$3 + B + B_{\text{mid}}$** |

* **Base Non-Backtracking Cost:** $3$ matrix-vector products per iteration.
* **Worst-Case Cost (10 backtracks with midpoint on every step):** $3 + 10 + 10 = \mathbf{23}$ matrix-vector products per iteration.
* **Empirical Average Cost (from 500-patch profiling):**
  * Average backtracks per iteration: $B \approx 1.05$
  * Average midpoint evaluations per iteration: $B_{\text{mid}} \approx 0.05$
  * Empirical average per iteration: $3 + 1.05 + 0.05 \approx \mathbf{4.1}$ matrix-vector products per iteration.

---

## 2. Approximate Upper-Bound Calculation

For a worst-case convergence profile ($150$ iterations, $10$ backtracks, $37{,}604$ image patches):

### A. Per-Patch Upper Bound
$$\text{Max Matrix Products per Patch} = 150 \times 23 = \mathbf{3{,}450}\text{ matrix-vector multiplications}$$

At $2 M N = 2 \times 38 \times 64 = 4{,}864$ FLOPs per product:
$$\text{Max FLOPs per Patch} \approx 3{,}450 \times 4{,}864 \approx \mathbf{16.78 \times 10^6}\text{ FLOPs (16.78 MFLOPs)}$$

### B. Full-Image Upper Bound ($37{,}604$ Patches)
$$\text{Max Matrix Products per Image} = 37{,}604 \times 3{,}450 = \mathbf{129{,}733{,}800}\text{ matrix-vector multiplications}$$

$$\text{Max Theoretical FLOPs per Image} \approx 129{,}733{,}800 \times 4{,}864 \approx \mathbf{6.31 \times 10^{11}}\text{ FLOPs (631 GFLOPs)}$$

---

## 3. Comparison with Baselines

| Solver | Iteration Budget | Matrix Operations per Iteration | Dominant Iterative Operation | Est. Matrix Products per Patch | Est. Matrix Products per Image ($37{,}604$ patches) |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **ASL-SR-DPT (Worst-Case)** | $150$ | $23$ | Dense matrix-vector products ($A z$) | $3{,}450$ | $129.7 \times 10^6$ |
| **ASL-SR-DPT (Empirical)** | $133$ | $4.1$ | Dense matrix-vector products ($A z$) | $545$ | $20.5 \times 10^6$ |
| **OMP** | $\le 38$ | $1$ | Correlation ($A^T r$) + least-squares solve | $\le 38$ | $\le 1.43 \times 10^6$ |
| **LASSO-ADMM** | $\le 100$ | $0$ | Triangular back-solves ($L, L^T$) | $0$ (pre-factorized) | $0$ (pre-factorized) |

---

## 4. Key Takeaways

1. **Massive Algorithmic Operation Gap:** Even under typical empirical conditions ($545$ products per patch), ASL-SR-DPT performs **$14.3\times$ more matrix multiplications per patch than OMP** ($38$ products) and infinitely more than LASSO-ADMM (which executes zero matrix-vector products during its iterative phase by relying entirely on fast triangular forward/backward substitutions).
2. **Backtracking Explosion:** Under poor step-size scaling, each backtracking failure forces redundant matrix multiplications.
3. **Linear Update Optimization Potential:** Exploiting $A (z + \alpha d) = A z + \alpha (A d)$ can eliminate virtually all matrix products inside the line-search loop, capping matrix products to strictly $2$ per iteration ($A z$ and $A d$).
