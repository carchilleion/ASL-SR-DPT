# Literature Review: Chen, Donoho & Saunders (2001) — Atomic Decomposition by Basis Pursuit

**Citation:**  
Chen, S. S., Donoho, D. L., & Saunders, M. A. (2001). "Atomic decomposition by basis pursuit." *SIAM Review*, 43(1), 129–159.  
(Initial conference versions published in 1995/1998).

---

## 1. Duality Between Basis Pursuit (BP) and Basis Pursuit De-Noising (BPDN)

Chen, Donoho & Saunders formalize the distinction between noiseless signal decomposition and reconstruction in noisy environments:

1. **Basis Pursuit (BP, Noiseless):**
   $$\min_x \|x\|_1 \quad \text{subject to} \quad A x = y$$
   This is applicable ONLY when $y$ is strictly noise-free.

2. **Basis Pursuit De-Noising (BPDN, Noisy):**
   $$\min_x \frac{1}{2}\|A x - y\|_2^2 + \lambda \|x\|_1$$
   This is the unconstrained Lagrangian form, mathematically equivalent to the Morozov constraint $\|A x - y\|_2 \le \epsilon$.

---

## 2. Optimality Conditions and Shrinkage Residual

From convex analysis, the first-order optimality condition for BPDN is:
$$0 \in A^T (A x^* - y) + \lambda \partial \|x^*\|_1$$
Equivalently:
$$A^T (y - A x^*) \in \lambda \partial \|x^*\|_1$$
For each coordinate $i \in \{1, \dots, N\}$:
$$\begin{cases} (A^T (y - A x^*))_i = \lambda \operatorname{sign}(x^*_i) & \text{if } x^*_i \neq 0 \\ |(A^T (y - A x^*))_i| \le \lambda & \text{if } x^*_i = 0 \end{cases}$$

### Direct Consequence on the Residual Norm:
Let $\mathcal{S} = \operatorname{supp}(x^*)$ be the active support. Projecting onto $\mathcal{S}$:
$$A_{\mathcal{S}}^T (y - A_{\mathcal{S}} x^*_{\mathcal{S}}) = \lambda \operatorname{sign}(x^*_{\mathcal{S}})$$
Assuming $A_{\mathcal{S}}$ has full column rank, the active solution satisfies:
$$x^*_{\mathcal{S}} = (A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1} A_{\mathcal{S}}^T y - \lambda (A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1} \operatorname{sign}(x^*_{\mathcal{S}})$$
and the residual is:
$$r = y - A_{\mathcal{S}} x^*_{\mathcal{S}} = (I - A_{\mathcal{S}} (A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1} A_{\mathcal{S}}^T) y + \lambda A_{\mathcal{S}} (A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1} \operatorname{sign}(x^*_{\mathcal{S}})$$

**Key Insights:**
1. The measurement residual $r$ has two components:
   - The orthogonal projection of $y$ onto the orthogonal complement of the range of $A_{\mathcal{S}}$: $(I - P_{\mathcal{S}}) y$.
   - A **shrinkage bias residual** proportional to the regularization parameter $\lambda$: $\lambda A_{\mathcal{S}} (A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1} \operatorname{sign}(x^*_{\mathcal{S}})$.
2. If $\lambda > 0$, the residual $\|A x^* - y\|_2$ is **strictly non-zero**, even if $A_{\mathcal{S}}$ spans $y$.
3. The residual norm is an increasing function of $\lambda$. For optimal noise suppression, Chen et al. establish that $\lambda$ should be proportional to the noise standard deviation: $\lambda \approx \sigma \sqrt{2 \log N}$.

---

## 3. The Hazard of Eliminating Residual in BPDN

Chen, Donoho & Saunders explicitly analyze what occurs when $\lambda \to 0$ in the presence of noise:
- As $\lambda \to 0$, BPDN degenerates to BP.
- The solver activates extra atoms in the overcomplete dictionary $A$ to interpolate the high-frequency random fluctuations of $e$.
- The resulting representation contains severe spurious non-zero coefficients ("atom hallucinations").
- The true underlying sparsity is destroyed, and the MSE in image space increases drastically.

---

## 4. Takeaway for ASL-SR-DPT

- ASL-SR-DPT's continuation algorithm uses $\lambda = 0.1$.
- At the stationary point, the gradient condition mirrors BPDN:
  $$A^T (y - A z^*) = \frac{\lambda}{\sigma_k^2} z^* \odot \exp\left(-\frac{(z^*)^2}{2\sigma_k^2}\right)$$
- The non-zero residual observed in A6 ($0.52 - 1.08$) is the natural, mathematically required shrinkage residual predicted by Chen, Donoho & Saunders.
- Seeking to force this residual to zero is equivalent to setting $\lambda \to 0$, which invites noise overfitting.
