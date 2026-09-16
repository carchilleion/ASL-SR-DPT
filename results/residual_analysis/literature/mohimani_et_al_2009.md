# Literature Review: Mohimani et al. (2009) — Smoothed L0 (SL0) Algorithm

**Citation:**  
Mohimani, H., Babaie-Zadeh, M., & Jutten, C. (2009). "A fast approach for addressing $\ell_0$ minimization problems and its applications." *IEEE Transactions on Signal Processing*, 57(1), 289–301.

---

## 1. Algorithmic Origin of ASL-SR-DPT

The ASL-SR-DPT framework is a direct intellectual descendant of the Smoothed $\ell_0$ (SL0) algorithm introduced by Mohimani, Babaie-Zadeh & Jutten.

SL0 approximates the discontinuous $\ell_0$ pseudo-norm using a family of smooth Gaussian functions:
$$\|s\|_0 \approx N - \sum_{i=1}^N f_\sigma(s_i), \quad f_\sigma(s_i) = \exp\left(-\frac{s_i^2}{2\sigma^2}\right)$$
As $\sigma \to 0$, $f_\sigma(0) = 1$ and $f_\sigma(s_i) \to 0$ for $s_i \neq 0$, so that $\sum_i (1 - f_\sigma(s_i)) \to \|s\|_0$.

---

## 2. Noiseless vs. Noisy Formulations in SL0

Mohimani et al. explicitly distinguish two cases:

### Case 1: Exact / Noiseless Recovery
$$\min_s \left( N - \sum_{i=1}^N \exp(-s_i^2 / 2\sigma^2) \right) \quad \text{subject to} \quad A s = y$$
In each iteration of SL0, an ascent step is taken on $\sum_i f_\sigma(s_i)$, followed by an exact orthogonal projection onto the affine constraint plane $\{s : A s = y\}$:
$$s \leftarrow s - A^\dagger (A s - y)$$
In this noiseless setting, the residual $\|A s - y\|_2$ is restored to **exactly zero** after every single step.

### Case 2: Noisy Measurements (Section IV.B of Mohimani et al.)
When $y = A s + e$ with $e \sim \mathcal{N}(0, \sigma_n^2 I)$, Mohimani et al. point out that forcing $A s = y$ via exact projection is detrimental:
> *"When noise is present, projecting onto $A s = y$ forces the estimator to fit the noise $e$. As $\sigma \to 0$, this creates large spurious spikes in the recovered signal."*

To handle noise, Mohimani et al. propose two remedies:
1. **Regularized Continuation Floor:** Do **not** let the continuation parameter $\sigma$ decrease to zero. Stop continuation when $\sigma \approx \sigma_n$.
2. **Discrepancy Stopping Rule:** Monitor the distance $\|A s - y\|_2$. Terminate continuation as soon as the residual drops to the expected noise level $\sqrt{M} \sigma_n$.

---

## 3. Structural Evolution in ASL-SR-DPT

ASL-SR-DPT modified the classical SL0 architecture by moving from an equality-constrained projection problem to an **unconstrained penalty formulation**:
$$\min_z \mathcal{L}(z) = \frac{1}{2}\|A z - y\|_2^2 - \lambda \sum_{i=1}^N \exp\left(-\frac{z_i^2}{2\sigma^2}\right)$$

### Key Consequence:
- In classical SL0, the projection step forced $\|A z - y\| = 0$ at each step.
- In ASL-SR-DPT, the gradient of the objective is:
  $$\nabla \mathcal{L}(z) = A^T(A z - y) + \frac{\lambda}{\sigma^2} z \odot \exp(-z^2/2\sigma^2)$$
- At any stationary point $\nabla \mathcal{L}(z^*) = 0$:
  $$A^T (y - A z^*) = \frac{\lambda}{\sigma^2} z^* \odot \exp(-{z^*}^2/2\sigma^2)$$
- **The unconstrained formulation automatically prevents the residual from collapsing to zero!**
- The residual observed in ASL-SR-DPT ($0.52 - 1.08$) is the natural equilibrium between data fidelity and smoothed $\ell_0$ regularization.

---

## 4. Takeaways for ASL-SR-DPT

- The founders of the SL0 methodology explicitly recognized that seeking zero residual under noise is mathematically erroneous.
- The non-zero residual of ASL-SR-DPT is not an implementation flaw; it is the built-in noise-buffering property of the unconstrained formulation.
- In contrast, the two-stage debiasing of A5/A5A6 reverts to an unconstrained projection ($z_{\mathcal{S}} = A_{\mathcal{S}}^\dagger y$), re-introducing the very vulnerability that Mohimani et al. warned against.
