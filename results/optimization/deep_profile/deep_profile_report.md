# Deep Profile Dashboard: V7_BEST_CURRENT

- **Source File:** `hybrid_sparse_solver_v7_optimized.py`
- **Code SHA-256:** `e2c74fa3f89555f08f392eec5db36d3bd62656ad9b3f6aea52394fa9f7cb5d2e`
- **Sample Size:** 500 patches (`test001.png`, $\sigma=15/255$)
- **Timing Trials:** 10 independent runs

## 1. Timing Summary

| Metric | Total (s) | Per Patch (ms) |
| :--- | :---: | :---: |
| **Mean** | 0.4381 s | **0.876 ms** |
| **Median** | 0.4403 s | 0.881 ms |
| **Std Dev** | 0.0063 s | 0.013 ms |
| **Min** | 0.4290 s | 0.858 ms |
| **Max** | 0.4477 s | 0.895 ms |

## 2. Fine-Grained Inner-Loop Breakdown (14 Metrics)

1. **Total Solver Time:** 0.4381 s (0.876 ms/patch)
2. **Time Per Iteration:** 0.0058 ms
3. **Objective Evaluation Time:** 0.000 ms/patch
4. **Gradient Evaluation Time:** 0.000 ms/patch
5. **Line Search Time:** 0.000 ms/patch
6. **Midpoint Evaluation Time:** 0.000 ms/patch
7. **Support Mask Time:** 0.000 ms/patch
8. **Residual Computation Time:** 0.000 ms/patch
9. **Peak Heap Memory:** 1847.10 KiB
10. **Iterations Per Patch:** 150.00
11. **NumPy Mat-Vec Products:** 0 total
12. **Exponential Evaluations:** 0 total
13. **Active Coordinate Ratio:** 0.9998
14. **Accepted Steps:** 75000 / 75000 (100.0%)

## 3. Reconstruction Quality Baseline

- **Mean Residual:** 0.725298
- **Mean Objective:** -5.762770
- **Patch PSNR:** 16.33 dB (Overall: 15.42 dB)
- **Patch SSIM:** 0.2062
- **Patch MSE:** 2.868972e-02
