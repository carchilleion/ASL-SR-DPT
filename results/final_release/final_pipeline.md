# ASL-SR-DPT A6 Final Mathematical Pipeline Specification

**Repository:** ASL-SR-DPT (*Accelerated Single-Loop Sparse Recovery with Dynamic Parameter Tuning for Mobile Image Denoising*)  
**Release:** Final Candidate Freeze (`A6_FINAL_FROZEN`)  
**Date:** September 8, 2026  
**Artifact Path:** `results/final_release/final_pipeline.md`  

---

## 1. Pipeline Flow Architecture

The image denoising and compressive reconstruction pipeline for `V7_A6_DC_PRESERVATION` follows an exact 14-stage mathematical progression:

```
[Clean Image X \in \mathbb{R}^{H \times W}]
           |
           v
[Stage 1: Additive White Gaussian Noise (AWGN)]
    X_noisy = X + \eta,   \eta \sim \mathcal{N}(0, \sigma^2 I)
           |
           v
[Stage 2: 8x8 Sliding Patch Extraction (Stride s=2)]
    P_i \in \mathbb{R}^{8 \times 8},   i = 1, \dots, N_p
           |
           v
[Stage 3: 2D Orthonormal Discrete Cosine Transform (2D-DCT)]
    \Theta_i = \mathcal{D} P_i \mathcal{D}^T \in \mathbb{R}^{8 \times 8} \implies \theta_i \in \mathbb{R}^{64}
           |
           v
[Stage 4: DC / AC Spectral Separation]
    \theta_{i,0} \in \mathbb{R} (DC),   \theta_{i,1:63} \in \mathbb{R}^{63} (AC)
           |
           +------------------------------------------+
           |                                          |
           v                                          v
[Stage 5: Direct DC Preservation]          [Stage 6: 37x63 AC Sensing Projection]
    y_{i,\text{dc}} = \theta_{i,0}             y_{i,\text{ac}} = A_{\text{ac}} \theta_{i,1:63} \in \mathbb{R}^{37}
           |                                          |
           |                                          v
           |                               [Stage 7: ASL-SR-DPT Nonconvex Solver]
           |                                   \min_{z} \frac{1}{2}\|A_{\text{ac}} z - y_{i,\text{ac}}\|_2^2 + \lambda \sum \phi(z_j; \sigma_k)
           |                                          |
           |                                          v
           |                               [Stage 8: Recovered AC Coefficients]
           |                                   \hat{\theta}_{i,\text{ac}} = \hat{z}_i \in \mathbb{R}^{63}
           |                                          |
           +--------------------+---------------------+
                                |
                                v
[Stage 9: Coefficient Vector Assembly & DC Restoration]
    \hat{\theta}_i = [y_{i,\text{dc}}, \ \hat{\theta}_{i,\text{ac}}^T]^T \in \mathbb{R}^{64}
                                |
                                v
[Stage 10: Inverse 2D-DCT Reconstruction]
    \hat{P}_i = \mathcal{D}^T \operatorname{reshape}(\hat{\theta}_i, 8, 8) \mathcal{D} \in \mathbb{R}^{8 \times 8}
                                |
                                v
[Stage 11: 2D Hamming Window Synthesis & Overlap Aggregation]
    W_{u,v} = w_u \cdot w_v,   \hat{X} = \frac{\sum_i W * \hat{P}_i}{\sum_i W * \mathbf{1}}
                                |
                                v
[Stage 12: Intensity Clipping & Normalization]
    \hat{X}_{\text{final}} = \operatorname{clip}(\hat{X}, 0.0, 1.0)
                                |
                                v
[Stage 13: Quantitative Metric Evaluation (PSNR / SSIM / MSE)]
```

---

## 2. Mathematical Details of Each Stage

### Stage 1: Noise Model (AWGN)
Given a normalized continuous ground truth image $X \in [0, 1]^{H \times W}$, corrupted observations are generated via zero-mean Gaussian perturbation:
$$X_{\text{noisy}} = X + \eta, \quad \eta_{h,w} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_{\text{norm}}^2), \quad \sigma_{\text{norm}} = \frac{\sigma}{255.0}$$

### Stage 2: Patch Extraction
Dense sliding window extraction operates with patch size $p = 8$ and stride $s = 2$. For an image of height $H$ and width $W$, the number of extracted patches is:
$$N_p = \left( \left\lfloor \frac{H - 8}{2} \right\rfloor + 1 \right) \times \left( \left\lfloor \frac{W - 8}{2} \right\rfloor + 1 \right)$$
For standard $481 \times 321$ BSD68 images, $N_p = 237 \times 157 = 37,209$ (or $37,604$ depending on exact bounding).

### Stage 3: Orthonormal 2D-DCT
Each spatial patch $P_i \in \mathbb{R}^{8 \times 8}$ is mapped into transform space using orthonormal DCT-II basis matrices $\mathcal{D} \in \mathbb{R}^{8 \times 8}$:
$$\Theta_i = \mathcal{D} P_i \mathcal{D}^T, \quad \theta_i = \operatorname{vec}(\Theta_i) \in \mathbb{R}^{64}$$
Due to orthonormality ($\mathcal{D}^T \mathcal{D} = I$), Parseval's identity holds: $\|P_i\|_F = \|\Theta_i\|_F = \|\theta_i\|_2$.

### Stage 4 & 5: DC / AC Decoupling & Direct DC Preservation
The DC component corresponds to the lowest spatial frequency atom ($u=0, v=0$):
$$\theta_{i,0} = \frac{1}{8} \sum_{x=0}^7 \sum_{y=0}^7 P_i(x, y)$$
In standard random compressed sensing, $\theta_{i,0}$ is mixed across all random measurements, leading to patch luminance drift and blocking artifacts. In A6:
$$y_{i,\text{dc}} = \theta_{i,0} \in \mathbb{R}$$
The DC coefficient is preserved directly without linear projection.

### Stage 6: AC Compressive Sensing
The remaining 63 AC transform coefficients $\theta_{i,\text{ac}} = \theta_{i, 1:63} \in \mathbb{R}^{63}$ represent high-frequency image textures and edges. They are compressed into $M_{\text{ac}} = 37$ scalar measurements using a fixed row-normalized Gaussian sensing matrix $A_{\text{ac}} \in \mathbb{R}^{37 \times 63}$:
$$y_{i,\text{ac}} = A_{\text{ac}} \theta_{i,\text{ac}} \in \mathbb{R}^{37}, \quad \|a_{m,:}\|_2 = 1 \quad \forall m \in \{1, \dots, 37\}$$

### Stage 7 & 8: ASL-SR-DPT Nonconvex Continuation Solver
The AC vector is recovered by minimizing a sequence of smooth surrogate objectives:
$$\min_{z \in \mathbb{R}^{63}} F(z; \sigma_k) = \frac{1}{2} \|A_{\text{ac}} z - y_{i,\text{ac}}\|_2^2 + \lambda \sum_{j=1}^{63} \phi(z_j; \sigma_k)$$
where $\phi(t; \sigma) = \frac{t^2}{t^2 + \sigma^2}$.  
Continuation starts at $\sigma_0 = \frac{1}{2} \|A_{\text{ac}}^T y_{i,\text{ac}}\|_\infty$ and decays geometrically ($\sigma_{k+1} = \max(0.95 \sigma_k, 0.01)$) with active-support screening, midpoint gradient evaluation, and Armijo backtracking line search.

### Stage 9 & 10: Coefficient Assembly & Inverse Transform
The full 64-dimensional coefficient vector is assembled by restoring the direct DC value:
$$\hat{\theta}_i = \begin{bmatrix} y_{i,\text{dc}} \\ \hat{z}_i \end{bmatrix} \in \mathbb{R}^{64}$$
Spatial patch reconstruction is computed via 2D-IDCT:
$$\hat{P}_i = \mathcal{D}^T \operatorname{reshape}(\hat{\theta}_i, 8, 8) \mathcal{D}$$

### Stage 11 & 12: Hamming Window Synthesis & Pixel Re-normalization
Overlapping patches are aggregated using a separable 2D Hamming window $W \in \mathbb{R}^{8 \times 8}$:
$$w(n) = 0.54 - 0.46 \cos\left(\frac{2\pi n}{7}\right), \quad W(u, v) = w(u) \cdot w(v)$$
$$\hat{X}(x, y) = \frac{\sum_{i \in \Omega(x,y)} W(x - x_i, y - y_i) \cdot \hat{P}_i(x - x_i, y - y_i)}{\sum_{i \in \Omega(x,y)} W(x - x_i, y - y_i)}$$
Final spatial intensities are strictly clipped to the valid visual range $[0.0, 1.0]$.

### Stage 13: Metric Computation
Reconstructed images $\hat{X}$ are compared against ground truth $X$ via:
$$\text{MSE} = \frac{1}{H \cdot W} \sum_{x=1}^H \sum_{y=1}^W (X(x, y) - \hat{X}(x, y))^2$$
$$\text{PSNR} = 10 \log_{10}\left(\frac{1.0}{\text{MSE}}\right) \text{ dB}$$
$$\text{SSIM} = \frac{(2\mu_X \mu_{\hat{X}} + C_1)(2\sigma_{X\hat{X}} + C_2)}{(\mu_X^2 + \mu_{\hat{X}}^2 + C_1)(\sigma_X^2 + \sigma_{\hat{X}}^2 + C_2)}$$
