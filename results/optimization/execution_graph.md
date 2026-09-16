# Execution Graph & Computational Pipeline Trace

**Project:** ASL-SR-DPT (Accelerated Single-Loop Sparse Recovery with Dynamic Parameter Tuning)  
**Document:** `results/optimization/execution_graph.md`  
**Scope:** Complete end-to-end trace from CLI invocation to disk logging.

---

## 1. End-to-End Pipeline Architecture

```mermaid
flowchart TD
    CLI["benchmark_runner.py (CLI Driver)"] --> Config["Load configs/final_config.json"]
    Config --> Dataset["Load BSD68 Dataset (dataset.py)"]
    Dataset --> TrialLoop["Trial Loop (1 .. N_trials)"]
    
    subgraph TrialSetup ["Trial Setup (Once per Trial)"]
        SensingGen["Generate Sensing Matrix A (sensing.py)\n- Shape: (38, 64) standard or (37, 63) DC\n- Row unit normalization"]
        PinvA["Precompute SVD Pseudoinverse pinv(A)\n- Stored on solver instance"]
        OMPPre["precompute_omp(A)\n- Column norms, normalized A, AtA"]
        LASSOPre["precompute_lasso_admm(A)\n- Cholesky factorization of (AtA + rho*I)"]
    end
    TrialLoop --> TrialSetup
    
    subgraph ImagePipeline ["Image Loop (1 .. 68 Images)"]
        LoadImg["Load Grayscale Image testXXX.png"]
        AddNoise["add_awgn (sigma_noise=15/255)"]
        PatchExtract["extract_patches (8x8, stride=2)\n- Yields 37,604 patches"]
        
        subgraph PatchLoop ["Patch-by-Patch Execution (37,604 iterations)"]
            ForwardDCT["Forward 2D DCT (dct_patch)\n- scipy.fftpack.dct 2D"]
            MeasurementGen["Generate Measurement y\n- Standard: y = A @ theta (M=38)\n- DC: theta_dc = theta[0], y_ac = A_ac @ theta_1: (M=37)"]
            
            subgraph ASL_Solver ["ASL-SR-DPT Inner Loop (<= 150 iterations)"]
                InitZ["Init: z0 = pinv(A) @ y"]
                InitRes["Initial Residual: r = A @ z - y"]
                IterLoop["Iteration Loop (k = 0 .. 149)"]
                Mask["Support Mask: |z| > 1e-5 * sigma\n- Full support reset every 3 iters"]
                ExpVector["Surrogate Exp: w = exp(-z^2 / (2*sigma^2))"]
                GradCalc["Analytical Gradient:\n- Fidelity: A^T @ r\n- Sparsity: (z / sigma^2) * w"]
                LineSearch["Armijo Backtracking Line Search\n- Precompute A_d = A @ (-grad)\n- Linear residual update: r + mu * A_d\n- Candidate exp: exp(-z_cand^2 / (2*sigma^2))"]
                Midpoint["Deferred Midpoint Step (if Armijo fails)"]
                StateUpdate["Step Update: z_new = z + mu*d, r = r_cand"]
                Continuation["Continuation: sigma = max(sigma * 0.95, 0.01)"]
            end
            MeasurementGen --> InitZ
            InitZ --> IterLoop
            
            InverseDCT["Inverse 2D DCT (idct_patch)\n- Full 64-coeff synthesis\n- (DC restored if DC mode)"]
        end
        
        HammingAgg["Normalized 2D Hamming Aggregation (reconstruct_image)"]
        Metrics["Quality Evaluation (metrics.py)\n- Spatial MSE, PSNR, SSIM"]
    end
    TrialSetup --> ImagePipeline
    PatchLoop --> HammingAgg
    HammingAgg --> Metrics
    Metrics --> CSV["Append to Raw CSV (results/raw/)"]
```

---

## 2. Granular Inventory of Computational Operations & Allocations

### 2.1 Sensing Matrix & Setup Precomputations (Trial Level)
- **Sensing Matrix $A$ Generation:**
  - Standard mode: `generate_random_sensing(M=38, N=64, seed=...)` in `sensing.py`.
  - Generates $38 \times 64$ Gaussian matrix; normalizes each row to $\|A_{i,:}\|_2 = 1.0$.
  - DC mode: `generate_dc_sensing(M_ac=37, N_ac=63, seed=...)`.
  - Frequency: **Once per trial** (reused across all 68 images and all $37,604 \times 68$ patches).
- **Pseudoinverse $\operatorname{pinv}(A)$ Calculation:**
  - `np.linalg.pinv(self.A)` in `HybridSparseSolverV7Optimized.__init__`.
  - Computes SVD decomposition $A = U \Sigma V^T$ and forms $V \Sigma^+ U^T$ ($64 \times 38$).
  - Frequency: **Once per trial / solver construction**.
- **OMP Precomputations:**
  - `precompute_omp(A)` in `hybrid_sparse_solver_v7_fixed.py`.
  - Computes column norms ($64,$), column-normalized matrix ($38 \times 64$), and Gram matrix $A^T A$ ($64 \times 64$).
  - Frequency: **Once per trial**.
- **LASSO-ADMM Cholesky Precomputation:**
  - `precompute_lasso_admm(A, rho=1.0)` in `hybrid_sparse_solver_v7_fixed.py`.
  - Computes $A^T A$ ($64 \times 64$) and Lower Cholesky factor $L = \operatorname{chol}(A^T A + \rho I)$ ($64 \times 64$).
  - Frequency: **Once per trial**.

---

### 2.2 Image Pre-Processing & Measurement Generation (Patch Level)
- **Patch Extraction:**
  - `extract_patches(image, patch_size=8, stride=2)` in `reconstruction.py`.
  - Slices overlapping $8 \times 8$ sub-arrays from $H \times W$ float64 image with stride 2.
  - Allocates 37,604 patch records per $481 \times 321$ image.
- **2D DCT Transformation:**
  - `dct_patch(p)` calls `dct_2d(p)`:
    $$\operatorname{DCT}(p) = \operatorname{dct}(\operatorname{dct}(p^T, \text{ortho})^T, \text{ortho})$$
  - Uses `scipy.fftpack.dct`. Allocates temporary transpose and 2D float64 output ($8 \times 8$).
  - Frequency: **37,604 times per image**.
- **Measurement Vector Generation:**
  - Standard: $y = A \theta \in \mathbb{R}^{38}$. Matrix-vector multiplication ($38 \times 64$) allocating a new 38-element array.
  - DC: $\theta_{\text{DC}} = \theta_0$, $y_{\text{AC}} = A_{\text{AC}} \theta_{1:} \in \mathbb{R}^{37}$. Allocates 37-element array.
  - Frequency: **37,604 times per image**.

---

### 2.3 Solver Inner-Loop Execution (Iterative Level)

Across 150 iterations per patch (approx. 5,640,600 iterations per full image):

| Operation | Implementation & Location | Memory / Allocation Profile | Redundancy / Bottleneck Status |
| :--- | :--- | :--- | :--- |
| **Input Validation** | `_validate_inputs` in solver entry | `np.asarray(y, dtype=float).reshape(-1)` + `np.all(np.isfinite(y))` | Runs once per patch at public boundary. |
| **Initialization** | $z_0 = P_{\text{init}} y$ | $64 \times 38$ mat-vec product. Allocates $z_0 \in \mathbb{R}^{64}$. | Done once per patch. Candidate for batching across patches. |
| **Initial Residual** | $r_0 = A z_0 - y$ | $38 \times 64$ mat-vec product. Allocates $r_0 \in \mathbb{R}^{38}$. | Done once per patch. |
| **Active Support Mask** | `np.abs(z) > support_threshold` | Allocates boolean array shape $(64,)$ on every iteration where $k \not\equiv 0 \pmod 3$. | Re-allocates boolean buffer 100 times per patch. |
| **Exponential Vector** | `np.exp(-(z*z) * inv_2sigma2)` | Allocates float64 array shape $(64,)$ on every iteration. Transcendentals on SIMD. | Re-allocated every iteration. |
| **Gradient Computation** | $A^T r + \lambda (z/\sigma^2) \odot w$ | Allocates $A^T r$ ($64,$) and elementwise products ($64,$). | $A^T$ transpose view or allocation. $A^T r$ is BLAS-2. |
| **Line Search Projection** | $A_d = A d$ | Allocates $A_d \in \mathbb{R}^{38}$. | Evaluated once per iteration. |
| **Trial Residuals** | $r_{\text{cand}} = r + \mu A_d$ | Allocates temporary vector $(38,)$ per trial step. | Allocations in trial loop. |
| **Candidate Exponentials** | $\exp(-z_{\text{cand}}^2 / (2\sigma^2))$ | Allocates temporary vector $(64,)$ per trial step. | Evaluated for candidate vector. |
| **Solution Vector Update** | `z_new = z.copy()`, `z_new[...] = ...` | Allocates new copy of $z$ ($64,$) on every accepted step. | Unnecessary full array copies when in-place or double-buffering is possible. |
| **Diagnostics Telemetry** | `diagnostics["objective"].append(...)`, etc. | 4 to 6 Python list appends per iteration (600–900 appends/patch). | Substantial Python interpreter overhead when detailed logging is enabled or maintained. |

---

### 2.4 Post-Processing & Image Reconstruction (Patch & Full Image)
- **DC Restoration (DC Mode):**
  - `restore_dc_component(theta_dc, z_hat)`: Allocates new 64-element array, sets index 0 to $\theta_{\text{DC}}$ and slice $1:64$ to recovered AC coefficients.
  - Frequency: 37,604 times per image.
- **2D IDCT Synthesis:**
  - `idct_patch(full_coeff.reshape(8, 8))` calls `scipy.fftpack.idct`.
  - Allocates spatial patch $(8, 8)$ float64 array.
  - Frequency: 37,604 times per image.
- **Normalized 2D Hamming Aggregation:**
  - Accumulates into `weighted_accumulator` ($H \times W$) and `weight_accumulator` ($H \times W$).
  - Per patch: `weighted_accumulator[y:y+8, x:x+8] += window_2d * p`, `weight_accumulator[y:y+8, x:x+8] += window_2d`.
  - Normalization: `img = weighted_accumulator / weight_accumulator`, `np.clip(img, 0, 1)`.
  - Memory: Two $481 \times 321$ float64 arrays ($\approx 2.4\text{ MB}$).
- **Image Quality Metrics:**
  - `compute_all_metrics(test_img, img_rec)`:
    - `compute_mse`: Full image elementwise difference squared, float mean.
    - `compute_psnr`: $10 \log_{10}(1 / \text{MSE})$.
    - `compute_ssim`: Skimage structural similarity index with 7x7 Gaussian window.
  - Frequency: Once per reconstructed image.
- **CSV Logging:**
  - Writes summary dict row into `results/raw/{experiment}_{sensing_mode}_raw.csv`.

---

## 3. Key Optimization Targets Ranked by Flops & Allocations

1. **Inner-Loop Diagnostics Overhead:**
   - Appending to Python lists on every iteration accounts for measurable interpreter overhead. In production benchmarks, scalar accumulators replace lists.
2. **Buffer Pre-allocation & In-Place Scratch:**
   - In each of the 150 iterations, arrays $w$ ($64,$), $\nabla$ ($64,$), $A_d$ ($38,$), $r_{\text{cand}}$ ($38,$), and $z_{\text{cand}}$ ($64,$) are allocated and garbage-collected. Allocating fixed scratch buffers eliminates ~900 array heap allocations per patch ($3.38 \times 10^7$ heap allocations per full image).
3. **Full-Support Contiguous BLAS Specialization:**
   - When all 64 coordinates are active ($>98\%$ of iterations), direct continuous matrix-vector products $A^T r$ and $A d$ avoid any Python slicing, indexing, or intermediate view construction.
4. **Precomputed Transpose & Constants:**
   - Pre-transposing $A$ once as $A^T$ in C-contiguous order optimizes memory stride in cache for back-projection $A^T r$.
5. **Batching Opportunities (Outside & Inside Solver):**
   - Batched sensing $Y = A \Theta$ (matrix-matrix multiplication) is $O(M \times N \times P)$ computed as a single BLAS GEMM call rather than 37,604 independent GEMV calls.
   - Batched pseudoinverse initialization $Z_0 = P_{\text{init}} Y$ as a single GEMM call.
