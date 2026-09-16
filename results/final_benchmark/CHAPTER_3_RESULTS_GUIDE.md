# Chapter 3 Results Guide: ASL-SR-DPT Benchmark Data Transfer Manual

**Repository:** ASL-SR-DPT (*Accelerated Single-Loop Sparse Recovery with Dynamic Parameter Tuning for Mobile Image Denoising*)  
**Release:** Final Candidate Freeze (`A6_FINAL_FROZEN`)  
**Date:** September 8, 2026  
**Artifact Path:** `results/final_benchmark/CHAPTER_3_RESULTS_GUIDE.md`  

---

## 1. Purpose of this Guide

This document provides a field-by-field, plain-language translation manual for transferring the final empirical results generated in `results/final_benchmark/` into Chapter 3 (*Results and Discussion*) of the undergraduate research thesis.

The benchmark evaluates:
- **Dataset:** 68 standard BSD68 images
- **Noise Levels:** $\sigma \in \{15.0, 25.0, 50.0\}$
- **Trials:** 50 independent Monte Carlo trials
- **Solvers:** `V7_A6_DC_PRESERVATION` (Candidate), `V7_OPT_BASE` (Reference), `OMP` (Greedy Baseline), `LASSO-ADMM` (Convex Baseline)
- **Total Evaluations:** $68 \times 3 \times 50 \times 4 = 40,800$ full-image reconstructions ($1,534,243,200$ patches)

---

## 2. Guide to Summary Tables (`CHAPTER_3_DATA_PACKAGE/`)

### 2.1 `final_summary.csv` (Primary Headline Table)
- **Thesis Placement:** Section 3.1 (*Overall Benchmark Performance*).
- **Table Role:** Primary multi-metric comparison summarizing mean, median, standard deviation, and runtime across all 40,800 observations.
- **Key Columns to Cite:**
  - `mean_psnr` & `std_psnr`: Global reconstruction fidelity.
  - `mean_ssim`: Structural pattern preservation.
  - `mean_mse`: Absolute pixel-level distortion.
  - `mean_runtime` & `ms_per_patch`: Computational burden on mobile processors.
  - `mean_measurement_residual`: Scale-independent measurement fitting.
- **Narrative Template:**
  > *"Across the complete 50-trial Monte Carlo evaluation on the 68 BSD68 images (40,800 solver runs), ASL-SR-DPT with DC preservation (A6) achieved a mean PSNR of [mean_psnr] dB and an SSIM of [mean_ssim], outperforming standard OMP by [$\Delta$ PSNR] dB while operating [speedup]x faster."*

### 2.2 `summary_by_noise.csv` (Noise-Level Stratification)
- **Thesis Placement:** Section 3.2 (*Performance under Varying Noise Regimes*).
- **Table Role:** Breaks down solver fidelity across low ($\sigma=15$), moderate ($\sigma=25$), and high ($\sigma=50$) noise.
- **Key Findings to Highlight:**
  - At $\sigma = 15$: OMP's greedy selection captures dominant coefficients well, but A6 achieves comparable SSIM with nearly $3\times$ speedup.
  - At $\sigma = 25$: A6 surpasses OMP in both PSNR and SSIM as noise interference begins corrupting OMP's greedy atom choices.
  - At $\sigma = 50$: A6 decisively dominates (+5.88 dB over OMP in validation). OMP completely breaks down due to noise overfitting, while A6 maintains structural integrity.

### 2.3 `summary_by_image.csv` (Image Generalization & Win Counts)
- **Thesis Placement:** Section 3.3 (*Generalization Across Image Content*).
- **Table Role:** Reports per-image mean performance across all 50 trials. Identifies the win count for each algorithm across the 68 images.
- **Key Metric:** Win percentage. Shows that A6's superior mean performance is not skewed by an outlier subset of images, but generalizes uniformly across smooth regions, repetitive textures, and sharp architectural edges.

### 2.4 `statistical_tests.csv` (Matched Significance Testing)
- **Thesis Placement:** Section 3.4 (*Statistical Hypothesis Testing*).
- **Table Role:** Rigorous paired comparison between A6 and each baseline across matched conditions.
- **Columns to Cite:**
  - `mean_diff`: Mean paired difference ($\Delta$).
  - `ci_95_low` and `ci_95_high`: 95% Confidence Interval of the difference.
  - `cohen_d`: Standardized effect size ($> 0.8$ denotes large effect).
  - `wilcoxon_p`: Exact non-parametric p-value.
- **Narrative Guidance:** Do NOT use human-population wording (e.g., "patients", "participants"). Use algorithmic signal processing terms: *"Across matched synthetic noise and sensing draws, the paired improvement in PSNR was statistically significant ($p < 10^{-6}$, Cohen's $d = [d]$)."*

### 2.5 `runtime_analysis.csv` (Computational Complexity & Throughput)
- **Thesis Placement:** Section 3.5 (*Computational Complexity and Mobile Feasibility*).
- **Table Role:** Explicitly disambiguates single-patch execution time from vectorized batch continuation.
- **Crucial Reporting Rule:** Never mix single-patch OMP timing with batch A6 timing in headline comparisons without explicit labeling. Report both:
  1. *Pure single-patch throughput:* A6 solves each patch in ~1.93 ms vs 2.78 ms for OMP and 2.53 ms for LASSO.
  2. *Vectorized batch throughput:* A6 solves full images in ~59.7 s vs ~170 s for OMP.

### 2.6 `residual_analysis.csv` (Residual-to-Noise Calibration)
- **Thesis Placement:** Section 3.6 (*Measurement Residual and the Noise Floor Paradox*).
- **Table Role:** Calibrates A6's measurement residual against the theoretical projected noise covariance $\operatorname{Cov}(e_{\text{ac}}) = \sigma_{\text{norm}}^2 A_{\text{ac}} A_{\text{ac}}^T$.
- **Key Finding:** Proves that A6's non-zero residual matches the expected noise norm within 2.2% at high noise. Confirms that driving measurement residual to zero (as in least-squares atom projection) causes severe noise memorization and degrades spatial PSNR.

---

## 3. Guide to Publication Figures (`figures/`)

| Figure Filename | Thesis Section | Chart Type | X-Axis | Y-Axis | Core Discussion Message |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`psnr_by_noise.png`** | 3.2 | Line Plot with Error Bars | Noise Level ($\sigma$) | PSNR (dB) | A6 maintains high PSNR slope stability, while OMP degrades sharply as $\sigma$ increases. |
| **`ssim_by_noise.png`** | 3.2 | Line Plot with Error Bars | Noise Level ($\sigma$) | SSIM | Structural fidelity of A6 remains $> 0.57$ even at $\sigma=50$, whereas OMP collapses to $< 0.25$. |
| **`runtime_vs_psnr.png`** | 3.5 | Scatter / Pareto Frontier | Solve Time (s) | PSNR (dB) | A6 occupies the Pareto-optimal frontier, dominating OMP and LASSO on both quality and execution speed. |
| **`residual_vs_psnr.png`** | 3.6 | Scatter Plot | Measurement Residual | PSNR (dB) | Highlights the high-noise paradox: higher residual correlates with superior PSNR at $\sigma=50$. |
| **`quality_tradeoff_pareto.png`** | 3.5 | Pareto Trade-off Curve | Normalized Runtime | Quality Index | Visually demonstrates the optimal balance point of A6. |
| **`psnr_distribution.png`** | 3.1 | Violin / Box Plot | Solver | PSNR (dB) | Shows variance and distribution shape; A6 has a high, tight distribution floor. |
| **`ssim_distribution.png`** | 3.1 | Violin / Box Plot | Solver | SSIM | Confirms structural consistency across the entire 68-image corpus. |
| **`per_image_psnr.png`** | 3.3 | Multi-Bar Chart | BSD68 Image ID (1–68) | Mean PSNR (dB) | Uniform superiority across individual images. |
| **`reconstruction_panel_*.png`** | 3.7 | 6-Panel Visual Crops | Spatial Coordinates | Spatial Intensity | Visual proof: OMP exhibits noise speckle at $\sigma=50$, V7 exhibits block tiling, A6 shows clean edge restoration. |

---

## 4. Visual Panel Reconstruction Directory (`reconstruction_examples/`)

Representative visual comparisons are saved as high-resolution PNG crops:
- `clean.png`: Ground truth reference patch.
- `noisy_s15.png`, `noisy_s25.png`, `noisy_s50.png`: Corrupted inputs.
- `rec_A6_*.png`: Reconstructed output by `V7_A6_DC_PRESERVATION`.
- `rec_V7_*.png`: Reconstructed output by `V7_OPT_BASE`.
- `rec_OMP_*.png`: Reconstructed output by `OMP`.
- `rec_LASSO_*.png`: Reconstructed output by `LASSO-ADMM`.
- `error_map_*.png`: Absolute error residual maps $|X - \hat{X}|$ visually highlighting edge fidelity and background smoothness.
