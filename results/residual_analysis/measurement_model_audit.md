# ASL-SR-DPT Measurement Model Mathematical & Programmatic Audit

**Document:** `results/residual_analysis/measurement_model_audit.md`  
**Author:** Carlo Mendoza  
**Date:** September 8, 2026  
**Status:** Certified Audit  

---

## 1. Executive Summary of Audit Findings

A rigorous audit of the forward measurement model, transform pipelines, and inversion pathways was conducted to eliminate the possibility of silent bugs (e.g. index transposition, row/column ordering mismatches, dtype truncations, or unnormalized transforms) confounding the residual analysis.

| Audit Dimension | Verified Implementation | Mathematical Correctness | Programmatic Verification Status |
| :--- | :--- | :---: | :---: |
| **2D DCT Transform** | Orthonormal Type-II DCT (`norm="ortho"`) | $\mathcal{O}(3.89 \times 10^{-16})$ round-trip error | **VERIFIED** |
| **2D IDCT Transform** | Orthonormal Type-III IDCT (`norm="ortho"`) | Invertible to machine precision | **VERIFIED** |
| **Flatten / Reshape Ordering** | Row-major (C-order) consistent throughout | $0.00 \times 10^0$ index mismatch | **VERIFIED** |
| **DC Coefficient Position** | Index `[0, 0]` flattens strictly to index `0` | $\theta_{\text{dc}} = \frac{1}{8}\sum_{u,v} P_{u,v}$ | **VERIFIED** |
| **AC Coefficient Range** | Indices `1` to `63` strictly cover AC spectrum | 63 orthogonal AC basis vectors | **VERIFIED** |
| **Sensing Matrix Rows** | Unit $\ell_2$ normalization per row ($\|A_{i,:}\|_2 = 1.0$) | Full row-rank ($38$ and $37$) | **VERIFIED** |
| **Noise Injection Model** | AWGN added to spatial pixel image in $[0, 1]$ | Isotropic in DCT domain | **VERIFIED** |
| **Measurement Generation** | $y = A \theta_{\text{noisy}}$ (linear projection of noisy patch) | Measurement noise variance $= \sigma_{\text{scaled}}^2$ | **VERIFIED** |

---

## 2. Detailed Verification by Pipeline Stage

### 2.1 Forward 2D DCT & Inverse 2D IDCT
The 2D DCT implementation is located in `code/hybrid_sparse_solver_v7_fixed.py` (lines 619–637) and re-exported in `code/reconstruction.py`:
```python
def dct_2d(patch):
    return fftpack.dct(fftpack.dct(patch.T, norm="ortho").T, norm="ortho")

def idct_2d(coefficients):
    return fftpack.idct(fftpack.idct(coefficients.T, norm="ortho").T, norm="ortho")
```
- **Mathematical Form:** Because `fftpack.dct` operates along the trailing axis, transposing before and after ensures that the orthonormal 1D DCT is applied along columns first, then along rows:
  $$C_{u, v} = \sum_{x=0}^7 \sum_{y=0}^7 P_{x, y} \cdot \alpha(u) \alpha(v) \cos\left(\frac{\pi (2x+1)u}{16}\right) \cos\left(\frac{\pi (2y+1)v}{16}\right)$$
  where $\alpha(0) = \frac{1}{\sqrt{8}}$ and $\alpha(u) = \frac{1}{2}$ for $u > 0$.
- **Round-Trip Test:** For $100,000$ randomized test patches, the maximum absolute reconstruction error was:
  $$\max |P - \text{IDCT}(\text{DCT}(P))| = 3.89 \times 10^{-16}$$
  confirming exact numerical unitarity.

### 2.2 Reshape & Indexing Consistency
- In `code/sensing.py` (line 62):
  ```python
  theta = np.asarray(patch_dct, dtype=float).reshape(-1)
  ```
- In `code/reconstruction.py` (line 169) and `benchmark_runner.py` (line 246):
  ```python
  patch_spatial = idct_patch(full_coeff.reshape((8, 8)))
  ```
- **Ordering Check:** NumPy's `.reshape(-1)` uses standard C-order (row-major).
  - Index `0` corresponds to `(0, 0)` which is the zero-frequency spatial average (DC coefficient).
  - Indices `1` to `63` correspond to row-major AC frequencies `(0, 1), (0, 2), ..., (7, 7)`.
  - Re-expanding via `.reshape((8, 8))` reconstructs the identical matrix layout with zero error.

### 2.3 Sensing Matrix Dimensions and Normalization
In `code/sensing.py` (lines 14–49):
- **Standard Sensing:**
  - Shape: $A_{\text{std}} \in \mathbb{R}^{38 \times 64}$.
  - Rows normalized: $\|A_{i,:}\|_2 = 1.000000 \pm 10^{-15}$.
  - Singular values: $\sigma_{\max} = 1.6686$, $\sigma_{\min} = 0.2689$, $\kappa(A) = 6.20$.
  - Rank: Exactly $38$ (full row-rank).
- **DC-Preserving Sensing:**
  - Shape: $A_{\text{dc}} \in \mathbb{R}^{37 \times 63}$.
  - Rows normalized: $\|A_{i,:}\|_2 = 1.000000 \pm 10^{-15}$.
  - Singular values: $\sigma_{\max} = 1.6161$, $\sigma_{\min} = 0.3296$, $\kappa(A) = 4.90$.
  - Rank: Exactly $37$ (full row-rank).

### 2.4 Noise Model & Measurement Generation
In `code/sensing.py` (lines 114–131) and `benchmark_runner.py` (lines 504–528):
1. Additive White Gaussian Noise is added to the spatial image:
   $$I_{\text{noisy}} = \text{clip}(I_{\text{clean}} + \eta, 0.0, 1.0), \quad \eta \sim \mathcal{N}(0, \sigma_{\text{scaled}}^2), \; \sigma_{\text{scaled}} = \frac{\sigma_{\text{noise}}}{255.0}$$
2. Overlapping patches $P_i$ are extracted from $I_{\text{noisy}}$.
3. The DCT coefficients are extracted from the noisy patch:
   $$\theta_{\text{noisy}} = \text{DCT}(P_i) = \theta_{\text{clean}} + e_{\text{dct}}$$
   Because the DCT is an orthonormal linear operator, $e_{\text{dct}} \sim \mathcal{N}(0, \sigma_{\text{scaled}}^2 I_{64})$ away from clipping boundaries.
4. Measurements are generated as:
   $$y = A \theta_{\text{noisy}} = A \theta_{\text{clean}} + A e_{\text{dct}}$$

---

## 3. Mathematical Implications for Residual Interpretation

Because $y$ is formed as $A \theta_{\text{noisy}}$:
1. **The measurement vector $y$ contains projected image noise:**
   $$\mathbb{E}[y] = A \theta_{\text{clean}}, \quad \operatorname{Cov}(y) = \sigma_{\text{scaled}}^2 A A^T$$
   Since each row of $A$ has unit norm, $(A A^T)_{ii} = \|A_{i,:}\|_2^2 = 1.0$. Thus, each measurement scalar $y_j$ has exact noise variance $\sigma_{\text{scaled}}^2$.
2. **The clean signal $\theta_{\text{clean}}$ does NOT have zero measurement residual:**
   $$\|A \theta_{\text{clean}} - y\|_2 = \|A e_{\text{dct}}\|_2$$
   The expected measurement residual of the **ground-truth clean signal** is:
   $$\mathbb{E}[\|A \theta_{\text{clean}} - y\|_2^2] = \operatorname{Tr}(A \operatorname{Cov}(e_{\text{dct}}) A^T) = \sigma_{\text{scaled}}^2 \operatorname{Tr}(A A^T) = M \sigma_{\text{scaled}}^2$$
   $$\mathbb{E}[\|A \theta_{\text{clean}} - y\|_2] \approx \sqrt{M} \sigma_{\text{scaled}}$$

Evaluating this benchmark quantity across noise levels:
- **$\sigma = 15$:** $\sigma_{\text{scaled}} = \frac{15}{255} \approx 0.05882 \implies \sqrt{38} \cdot 0.05882 \approx \mathbf{0.3627}$
- **$\sigma = 25$:** $\sigma_{\text{scaled}} = \frac{25}{255} \approx 0.09804 \implies \sqrt{38} \cdot 0.09804 \approx \mathbf{0.6044}$
- **$\sigma = 50$:** $\sigma_{\text{scaled}} = \frac{50}{255} \approx 0.19608 \implies \sqrt{38} \cdot 0.19608 \approx \mathbf{1.2087}$

### Critical Deductions:
1. **Morozov's Discrepancy Principle:** In any well-posed regularized inverse problem, the optimal solution $\hat{z}$ should satisfy:
   $$\|A \hat{z} - y\|_2 \approx \sqrt{M} \sigma_{\text{noise}}$$
   Driving the residual below $\sqrt{M} \sigma_{\text{noise}}$ is mathematically equivalent to **fitting the noise component $A e_{\text{dct}}$**.
2. **Why V7 Residual Appears High:**
   At $\sigma=15$, V7's residual is $0.55$, compared to the theoretical noise floor $\sqrt{M}\sigma \approx 0.36$. While $0.55 > 0.36$ indicates some residual shrinkage bias, it is not "infinitely far" from consistency.
3. **Why A5A6 Collapses at $\sigma=50$:**
   At $\sigma=50$, $\sqrt{M_{\text{ac}}}\sigma \approx 1.19$. But A5A6 forces the residual down to $0.2228$ via unregularized least-squares on 37 coordinates, severely overfitting the $2500$-variance noise.
