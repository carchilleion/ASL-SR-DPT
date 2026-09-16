# ASL-SR-DPT Thesis Manuscript & Documentation

This directory contains the manuscript chapters and research documentation for the undergraduate thesis:

**"Accelerated Single-Loop Sparse Recovery with Dynamic Parameter Tuning for Mobile Image Denoising"**

---

## Directory Contents

| File | Format | Description |
| :--- | :--- | :--- |
| `chapter_1_and_2_final_draft.pdf` | PDF | Original submitted final draft of Chapters 1 and 2. |
| `chapter_1_introduction.md` | Markdown | Chapter 1: Introduction, Problem Background, Literature Review, Theoretical Framework, Conceptual Paradigm, and Definitions. |
| `chapter_2_methodology.md` | Markdown | Chapter 2: Research Methodology, Virtualized Hardware Environment, Controlled AWGN Injection, Sensing Models, Patch Decomposition, Solvers, and Data Analysis Metrics. |

---

## Codebase Alignment & Verification Matrix

The benchmark system implemented in `code/` strictly adheres to the specifications and ethical standards set forth in Chapter 2:

1. **Dataset Integrity**: Exactly all 68 images of the BSD68 dataset are utilized (`data/BSD68/`), matching Section 2.2.
2. **Noise Protocol**: AWGN standard deviations $\sigma \in \{15, 25, 50\}$ on $[0, 255]$ scale, matching Phase 2.
3. **Sensing Models**:
   - **Standard Random Gaussian Sensing**: $M=38, N=64$, unit row-normalized.
   - **DC-Preserving Sensing**: Uncompressed noisy DC coefficient ($\theta_{\text{DC}}$) retained directly, while 63 AC coefficients are compressively measured ($M_{\text{AC}}=37$), matching Section 2.4.
4. **Reconstruction**: Overlapping patches extracted at `stride=2` and aggregated via normalized 2D Hamming windowing ($\text{Hamming}(8) \otimes \text{Hamming}(8)$).
5. **Matched Solver Comparison**: Exactly identical measurement vector $y$ and noisy patch coordinates provided to ASL-SR-DPT, OMP, and LASSO-ADMM.
6. **Timer Isolation**: Host setup time (Cholesky/pseudoinverse) is isolated from iterative `solve_time` per patch using `time.perf_counter()`.
7. **Scientific Honesty**: In compliance with Section 2.3 ("Scientific Honesty in Reporting"), all empirical results, runtimes, speedups, and quality metrics are reported transparently with zero fabrication.
