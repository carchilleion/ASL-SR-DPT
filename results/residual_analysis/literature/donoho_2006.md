# Literature Review: Donoho (2006) — Compressed Sensing

**Citation:**  
Donoho, D. L. (2006). "Compressed sensing." *IEEE Transactions on Information Theory*, 52(4), 1289–1306.

---

## 1. Core Principles of Compressed Sensing

Donoho established that an approximately sparse signal $x \in \mathbb{R}^N$ with sparsity $k \ll N$ can be reconstructed from $M \ll N$ non-adaptive linear measurements:
$$y = A x + e$$
where $M \ge C \cdot k \log(N / k)$.

When $A$ is a random projection matrix (e.g., Gaussian ensemble with orthonormalized rows), the geometry of the high-dimensional ball $\ell_1^N$ projected into $\mathbb{R}^M$ exhibits profound concentration phenomena (neighborliness of convex polytopes).

---

## 2. Information-Theoretic Bounds and Noise Floor

In Section IV of Donoho (2006), the behavior under noisy observations is derived:
- The measurement vector $y = A x + e$ lives in $\mathbb{R}^M$.
- When $e_i \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2)$, the length of the noise vector is concentrated tightly in a thin spherical shell:
  $$\|e\|_2 = \sigma \sqrt{M} \left( 1 + \mathcal{O}(M^{-1/2}) \right)$$
- No estimator $\hat{x}(y)$ can achieve an expected residual $\|A \hat{x} - y\|_2$ significantly less than $\sigma \sqrt{M}$ without encoding the specific realization of the noise $e$.
- In fact, the minimax error rate for sparse recovery over the $\ell_0$ or weak-$\ell_p$ ball under Gaussian noise is bounded below by:
  $$\inf_{\hat{x}} \sup_{x \in \Sigma_k} \mathbb{E}[\|\hat{x} - x\|_2^2] \asymp \sigma^2 k \log(N/k)$$

---

## 3. Geometric Interpretation of Measurement Residual

Donoho provides a geometric picture:
1. The measurement space $\mathbb{R}^M$ decomposes into the low-dimensional manifold/subspace spanned by the true sparse signal's support $A_{\mathcal{S}}$, and its orthogonal complement.
2. The noise $e \sim \mathcal{N}(0, \sigma^2 I_M)$ is isotropic in $\mathbb{R}^M$.
3. When projecting onto the $k$-dimensional support subspace, the noise component in the range of $A_{\mathcal{S}}$ has dimension $k$, with energy $k \sigma^2$.
4. The remaining $(M - k)$ dimensions of noise lie in the orthogonal complement of the support. This orthogonal noise has energy $(M - k) \sigma^2$.
5. **Crucial Geometric Fact:** If an estimator $\hat{x}$ has exact support $\mathcal{S}$, the residual $r = y - A \hat{x}$ MUST capture all the noise energy orthogonal to $A_{\mathcal{S}}$:
   $$\|r\|_2^2 \ge \|(I - P_{\mathcal{S}}) e\|_2^2 \approx (M - k) \sigma^2$$
   Thus, even an **ideal oracle estimator** knowing the exact true support will have a residual of norm at least $\sigma \sqrt{M - k}$.

---

## 4. Takeaway for ASL-SR-DPT

- In ASL-SR-DPT with $M = 38$ (or $M_{\text{ac}} = 37$) and patch DCT dimension $N = 64$:
  If the effective sparsity of an $8 \times 8$ image patch is $k \approx 8 - 15$, the oracle noise residual is:
  $$\|r_{\text{oracle}}\|_2 \approx \sigma \sqrt{37 - 12} = \sigma \sqrt{25} = 5 \sigma$$
  For $\sigma = 25/255 = 0.0980$, $\|r_{\text{oracle}}\|_2 \approx 0.49$.
- The observed residual of A6 ($0.686$) reflects this fundamental geometric floor plus the mild shrinkage bias required to preserve sparsity.
- Demanding $\|A z - y\|_2 \to 0$ violates Donoho's geometric bounds by forcing the $k$ active columns to span the $(M - k)$-dimensional orthogonal noise space.
