# Active-Support Mechanism Diagnostic Report

## 1. Empirical Distribution of Active Coefficients

- **Total Iterations Sampled:** 68,855 iterations (500 patches)
- **Mean Active Ratio:** 0.9998 (63.99 / 64 coefficients)
- **Percentage of Iterations with 100% Active Support:** 98.98%
- **Minimum Active Count Observed:** 60 / 64 coefficients

## 2. Why FAL0 Zero-Element Neglect is Ineffective in V7

1. **Threshold Scaling (`tau = 1e-5 * sigma`):** At initial sigma ~ 10, tau ~ 1e-4. At sigma_min = 0.01, tau ~ 1e-7. Image DCT coefficients under AWGN noise level sigma_n = 15/255 ~ 0.0588 almost never fall below 1e-7.
2. **Periodic Reopening Frequency (`T = 3`):** Every 3 iterations, support is forcibly reset to all 64 coordinates. Any coefficient dipping below 1e-7 is immediately reactivated two steps later.
3. **Slicing Overhead:** Subsetting NumPy arrays `A[:, active_indices]` creates memory slice objects and indexing overhead without yielding computational reduction, since dropping 1-3 coefficients in a 38x64 matrix yields zero BLAS acceleration.
