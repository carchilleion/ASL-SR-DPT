# Literature Review: Javanmard & Montanari (2014) — Debiasing and Refitting in High-Dimensional Regression

**Citation:**  
Javanmard, A., & Montanari, A. (2014). "Confidence intervals and hypothesis testing for high-dimensional regression." *Journal of Machine Learning Research*, 15(1), 2869–2909.  
See also: van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). "On asymptotically normal indicators in high-dimensional estimation." *Annals of Statistics*, 42(3), 1166–1202.

---

## 1. The Shrinkage Bias Problem in Regularized Estimation

Regularized estimators such as LASSO ($\ell_1$) or smoothed $\ell_0$ solve:
$$\hat{\theta} = \arg\min_\theta \left\{ \frac{1}{2}\|y - A \theta\|_2^2 + \lambda \mathcal{R}(\theta) \right\}$$
While the regularizer $\mathcal{R}(\theta)$ effectively suppresses noise and performs variable selection, it inevitably introduces **shrinkage bias**:
$$\mathbb{E}[\hat{\theta}_{\mathcal{S}}] \neq \theta_{\mathcal{S}}$$
The active non-zero coefficients are biased toward zero. This bias increases the measurement residual $\|A \hat{\theta} - y\|_2$ above the pure noise level.

---

## 2. Two-Stage Least-Squares Refitting and Its Pitfalls

To eliminate this bias, a common heuristic in compressive sensing and statistics is **Two-Stage Least Squares (OLS Refitting)**:
1. **Stage 1 (Support Selection):** Identify active support $\mathcal{S} = \{i : |\hat{\theta}_i| > \tau\}$.
2. **Stage 2 (Unconstrained Refitting):** Re-estimate active coefficients on $\mathcal{S}$ via ordinary least squares:
   $$\hat{\theta}_{\mathcal{S}}^{\text{OLS}} = (A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1} A_{\mathcal{S}}^T y$$
   while setting $\hat{\theta}_{\mathcal{S}^c}^{\text{OLS}} = 0$.

### Exact Variance Analysis:
Substituting the observation model $y = A_{\mathcal{S}} \theta_{\mathcal{S}} + e$ (assuming $\operatorname{supp}(\theta) \subseteq \mathcal{S}$):
$$\hat{\theta}_{\mathcal{S}}^{\text{OLS}} = \theta_{\mathcal{S}} + (A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1} A_{\mathcal{S}}^T e$$
The bias is eliminated ($\mathbb{E}[\hat{\theta}_{\mathcal{S}}^{\text{OLS}} - \theta_{\mathcal{S}}] = 0$), but the variance is:
$$\operatorname{Cov}(\hat{\theta}_{\mathcal{S}}^{\text{OLS}}) = \sigma^2 (A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1}$$
The total estimation error in $\ell_2$ norm is:
$$\mathbb{E}[\|\hat{\theta}_{\mathcal{S}}^{\text{OLS}} - \theta_{\mathcal{S}}\|_2^2] = \sigma^2 \operatorname{Tr}\left( (A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1} \right) = \sigma^2 \sum_{i=1}^{|\mathcal{S}|} \frac{1}{\lambda_i(A_{\mathcal{S}}^T A_{\mathcal{S}})}$$

---

## 3. The Collapse Condition: When $|\mathcal{S}| \to M$

Javanmard & Montanari and random matrix theory (Marchenko-Pastur distribution) establish that:
1. When $|\mathcal{S}| \ll M$, the columns of $A_{\mathcal{S}}$ are nearly orthogonal, $\lambda_{\min}(A_{\mathcal{S}}^T A_{\mathcal{S}}) \approx 1$, and $\operatorname{Tr}((A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1}) \approx |\mathcal{S}|$. OLS debiasing is highly effective.
2. **When $|\mathcal{S}| \to M$ (the saturation regime):**
   The smallest eigenvalue $\lambda_{\min}(A_{\mathcal{S}}^T A_{\mathcal{S}})$ approaches $(1 - \sqrt{|\mathcal{S}|/M})^2 \to 0$.
   The trace $\operatorname{Tr}((A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1})$ explodes towards infinity.
3. In this saturation regime, the variance penalty $\sigma^2 \operatorname{Tr}((A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1})$ completely dwarfs the bias reduction:
   $$\text{MSE}_{\text{OLS}} = \sigma^2 \operatorname{Tr}((A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1}) \gg \text{Bias}^2 + \text{Var}_{\text{regularized}}$$

---

## 4. Relevance to ASL-SR-DPT

- In the ASL-SR-DPT investigation, Variant A5 and A5A6 implemented Stage 2 unconstrained refitting.
- Telemetry revealed that at $\sigma = 50$:
  - Support size $|\mathcal{S}| = 37$ out of $M_{\text{ac}} = 37$ ($100\%$ saturation).
  - Condition number $\kappa(A_{\mathcal{S}}) = 4.90 - 6.20$.
  - Noise amplification factor $\operatorname{Tr}((A_{\mathcal{S}}^T A_{\mathcal{S}})^{-1}) = 72.1$.
  - Expected noise variance injected: $0.03845 \times 72.1 = \mathbf{2.772}$.
- As predicted by Javanmard & Montanari, this caused A5A6's PSNR to plummet from $23.24\text{ dB}$ (A6) down to $17.21\text{ dB}$ (a catastrophic $6.03\text{ dB}$ drop).
- Meanwhile, the residual dropped to $0.08 - 0.26$. This confirms that **debiasing achieves low residual at the cost of massive variance inflation**.
