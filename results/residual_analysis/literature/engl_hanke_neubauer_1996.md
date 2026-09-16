# Literature Review: Engl, Hanke & Neubauer (1996) — Regularization of Inverse Problems

**Citation:**  
Engl, H. W., Hanke, M., & Neubauer, A. (1996). *Regularization of Inverse Problems*. Mathematics and Its Applications, Vol. 375, Kluwer Academic Publishers (Springer), Dordrecht/Boston/London.

---

## 1. Classical Inverse Problems and Iterative Regularization

Engl, Hanke & Neubauer provide the definitive rigorous mathematical foundation for solving ill-posed linear and nonlinear operator equations $A x = y$ with noisy data $y^\delta$, $\|y^\delta - y\| \le \delta$.

Chapters 4 and 6 focus specifically on:
1. **A Posteriori Parameter Choice Rules** (including Morozov's Discrepancy Principle).
2. **Iterative Regularization Methods** (e.g., Landweber iteration, Steepest Descent, Conjugate Gradients).

---

## 2. The Semi-Convergence Phenomenon

In iterative methods for inverse problems, the iteration index $k$ acts as the inverse of the regularization parameter:
$$k \sim \frac{1}{\alpha}$$
- **Early iterations ($k$ small):** The solver recovers low-frequency, stable components of the true signal. The reconstruction error $\|x_k - x_{\text{true}}\|$ decreases rapidly.
- **Late iterations ($k$ large):** The solver begins inverting small singular values of $A$, reconstructing noise $e$. The reconstruction error $\|x_k - x_{\text{true}}\|$ **increases and eventually diverges to infinity**, even though the residual $\|A x_k - y^\delta\|$ continues to monotonically decrease to zero!

This is the classic **semi-convergence phenomenon** of iterative inverse problem solvers.

```
Error / Residual
  ^
  |      \                     / Reconstruction Error ||x_k - x_true||
  |       \    Semi-          /
  |        \  Convergence    /
  |         \   Minimum     /
  |          \   v         /
  |-----------\-----------/------------------- Discrepancy Threshold tau * delta
  |            \_________/
  |             \
  |              \____________________________ Measurement Residual ||A x_k - y||
  +--------------------------------------------> Iteration k
                       ^
                 Optimal Stop k*
```

---

## 3. Discrepancy Stopping Criterion for Iterative Solvers

**Rule 6.2 (Engl, Hanke & Neubauer):**  
Let $\tau > 1$ be a fixed constant. Terminate the iterative process at the first iteration $k^* = k^*(\delta, y^\delta)$ such that:
$$\|A x_{k^*} - y^\delta\| \le \tau \delta < \|A x_k - y^\delta\|, \quad \forall 0 \le k < k^*$$

**Convergence Theorem (Theorem 6.4):**  
If the iterative method is stopped via the discrepancy principle with $\tau > \sup_k \frac{\|A x_k - y^\delta\|}{\|A x_k - y\|}$, then the sequence of iterates is regularizing:
$$\lim_{\delta \to 0} \|x_{k^*(\delta, y^\delta)} - x_{\text{true}}\| = 0$$
Conversely, if the solver is allowed to run to completion ($k \to \infty$) so that $\|A x_\infty - y^\delta\| \to 0$, the iterates diverge:
$$\limsup_{\delta \to 0} \|x_\infty(\delta) - x_{\text{true}}\| = \infty$$

---

## 4. Takeaway for ASL-SR-DPT

- ASL-SR-DPT is an iterative continuation solver.
- In ASL-SR-DPT V7, the solver currently runs for a fixed number of iterations ($150$) down to $\sigma_{\min} = 0.01$, without a Morozov stopping check on $\|A z - y\|_2$.
- Engl et al. prove that driving the measurement residual below $\tau \delta \approx \tau \sigma_{\text{norm}} \sqrt{M}$ is mathematically guaranteed to worsen reconstruction quality.
- This provides the theoretical foundation for considering an early-stopping discrepancy rule (A6-DP) in future designs.
