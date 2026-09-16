# Literature Review: Morozov's Discrepancy Principle (1966)

**Citation:**  
Morozov, V. A. (1966). "On the solution of functional equations by the method of regularization." *Soviet Mathematics Doklady*, 7, 414–417.  
See also: Morozov, V. A. (1984). *Methods for Solving Incorrectly Posed Problems*. Springer-Verlag, New York.

---

## 1. Classical Formulation

Consider an ill-posed linear inverse problem:
$$A x = y$$
where $A: X \to Y$ is a bounded linear operator between Hilbert spaces. In practice, only noisy data $y^\delta$ are available, satisfying the deterministic or statistical noise bound:
$$\|y^\delta - y_{\text{true}}\|_Y \le \delta$$

When $A$ is not boundedly invertible (or in discrete settings where $A \in \mathbb{R}^{M \times N}$ with $M < N$ or $A$ is ill-conditioned), attempting to solve the exact fidelity equation:
$$\|A x - y^\delta\|_2 = 0$$
yields the minimum-norm least-squares solution:
$$x^\dagger = A^\dagger y^\delta = x_{\text{true}} + A^\dagger e$$
where $e = y^\delta - y_{\text{true}}$. The error norm satisfies:
$$\|x^\dagger - x_{\text{true}}\|_2 = \|A^\dagger e\|_2 \ge \frac{1}{\sigma_{\min}(A)} \|e\|_2$$
If $\sigma_{\min}(A)$ is small or $M < N$, noise components in the range of $A$ are catastrophically amplified, and the reconstructed solution diverges.

---

## 2. Morozov's Discrepancy Principle (MDP)

To prevent noise fitting while restoring stable approximations, Tikhonov regularization considers:
$$x_\alpha = \arg\min_x \left\{ \|A x - y^\delta\|_2^2 + \alpha \mathcal{R}(x) \right\}$$
where $\mathcal{R}(x)$ is a stabilizing functional (e.g., $\|x\|_2^2$, or in modern sparse recovery, $\|x\|_1$ or smoothed $\ell_0$).

**Theorem (Morozov):**  
Let the discrepancy function be defined as:
$$\rho(\alpha) = \|A x_\alpha - y^\delta\|_2$$
Under mild convexity and regularity conditions, $\rho(\alpha)$ is continuous, strictly monotonically increasing with $\alpha$, and satisfies:
$$\lim_{\alpha \to 0} \rho(\alpha) = \|A x^\dagger - y^\delta\|_2 = 0, \quad \lim_{\alpha \to \infty} \rho(\alpha) = \|y^\delta\|_2$$
Therefore, for any target discrepancy $\tau \delta$ with $\tau \ge 1$, there exists a **unique** regularization parameter $\alpha^* = \alpha(\delta)$ such that:
$$\|A x_{\alpha^*} - y^\delta\|_2 = \tau \delta$$

Choosing $\tau \in [1, 1.2]$ guarantees that the regularized solution $x_{\alpha^*}$ converges to $x_{\text{true}}$ as $\delta \to 0$:
$$\lim_{\delta \to 0} \|x_{\alpha(\delta)} - x_{\text{true}}\|_2 = 0$$

---

## 3. Why Zero Residual is Pathological

Morozov's mathematical framework demonstrates that:
1. **The measurement vector $y^\delta$ is corrupt:** It does not belong to the noise-free range $A(\mathcal{X})$.
2. **Matching noise-corrupted data exactly is an error:** Setting $\|A x - y^\delta\|_2 = 0$ forces the estimator $x$ to reproduce the high-frequency measurement perturbations $e$.
3. **The true signal has non-zero discrepancy:**
   $$\|A x_{\text{true}} - y^\delta\|_2 = \|y_{\text{true}} - (y_{\text{true}} + e)\|_2 = \|e\|_2 \approx \delta$$
   Therefore, any estimator $\hat{x}$ with $\|A \hat{x} - y^\delta\|_2 \ll \delta$ is **statistically closer to the noise than the true signal is**.

---

## 4. Key Takeaways for ASL-SR-DPT

- In ASL-SR-DPT, $y = A \theta_{\text{clean}} + e$, where $e \sim \mathcal{N}(0, \sigma_{\text{norm}}^2 I_M)$. The expected noise norm is $\delta = \mathbb{E}[\|e\|_2] \approx \sigma_{\text{norm}} \sqrt{M}$.
- The measurement residual $\|A z - y\|_2$ should **not** converge to zero. An optimal sparse estimator **must** maintain a residual of order $\sigma_{\text{norm}} \sqrt{M}$.
- Variant A6's residual of $0.52$ ($\sigma=15$), $0.69$ ($\sigma=25$), and $1.08$ ($\sigma=50$) precisely tracks $\delta$, fulfilling Morozov's principle.
- Forcing the residual to $0.05 - 0.22$ via least-squares debiasing (as in A5A6) undercuts Morozov's discrepancy floor, leading directly to the observed quality collapse at $\sigma=50$.
