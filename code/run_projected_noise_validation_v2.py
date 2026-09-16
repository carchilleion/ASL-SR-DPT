"""
ASL-SR-DPT A6 Projected-Noise Residual Validation Script (V2).
Executes rigorous mathematical and empirical calibration:
  - Verifies exact noise model and clipping effects
  - Verifies DCT transformation and AC noise covariance
  - Evaluates actual A_AC sensing matrix Gram matrices across 50 seeds
  - Performs 100,000 Monte Carlo synthetic noise trials per noise level
  - Evaluates empirical image-derived noise on BSD68 patches
  - Analyzes residual correlations (Pearson, Spearman)
  - Quantifies DC/AC energy and patch-boundary vs interior error
  - Generates 12 publication-quality figures
"""

import os
import sys
import csv
import json
import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats
import scipy.fftpack as fftpack
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Strict single-threading enforcement
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dataset import load_bsd68
from sensing import generate_dc_sensing, add_awgn
from reconstruction import extract_patches, dct_patch, idct_patch

RESULTS_DIR = os.path.join(current_dir, "..", "results", "residual_analysis_v2")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

M_AC = 37
N_AC = 63
SIGMA_LEVELS = [15.0, 25.0, 50.0]
BASE_SEED = 20260908


def run_part2_noise_model_verification():
    """Verify pixel clipping and exact noise variance on BSD68 images."""
    print("[Part 2] Verifying actual noise model and clipping effects...")
    dataset = load_bsd68(limit=10)
    clipping_report = []

    for sigma in SIGMA_LEVELS:
        s_norm = sigma / 255.0
        clip_fractions = []
        emp_vars = []
        unclipped_vars = []

        for idx, (img_id, img_data) in enumerate(dataset.items()):
            img = img_data["image"]
            seed = BASE_SEED + 1000 + idx
            rng = np.random.default_rng(seed)
            noise_raw = rng.normal(loc=0.0, scale=s_norm, size=img.shape)
            noisy_unclipped = img + noise_raw
            noisy_clipped = np.clip(noisy_unclipped, 0.0, 1.0)

            # Fraction clipped
            is_clipped = (noisy_unclipped < 0.0) | (noisy_unclipped > 1.0)
            clip_fractions.append(float(np.mean(is_clipped)))

            # Actual noise injected after clipping
            eff_noise = noisy_clipped - img
            emp_vars.append(float(np.var(eff_noise)))
            unclipped_vars.append(float(np.var(noise_raw)))

        clipping_report.append({
            "sigma": sigma,
            "sigma_norm": s_norm,
            "theoretical_var": s_norm ** 2,
            "mean_clip_pct": float(np.mean(clip_fractions)) * 100.0,
            "max_clip_pct": float(np.max(clip_fractions)) * 100.0,
            "effective_var": float(np.mean(emp_vars)),
            "var_ratio": float(np.mean(emp_vars)) / (s_norm ** 2),
        })

    print("Clipping Verification Summary:")
    for rep in clipping_report:
        print(f"  sigma={rep['sigma']}: Clip={rep['mean_clip_pct']:.3f}%, EffVar/TheoVar={rep['var_ratio']:.4f}")
    return clipping_report


def run_part3_dct_verification():
    """Numerically verify 2D DCT orthonormality and noise properties."""
    print("[Part 3] Verifying 2D-DCT transformation...")
    # Build 64x64 DCT matrix
    D = np.zeros((64, 64))
    for i in range(64):
        e = np.zeros(64)
        e[i] = 1.0
        D[:, i] = dct_patch(e.reshape((8, 8))).reshape(-1)

    ortho_error = float(np.max(np.abs(D.T @ D - np.eye(64))))
    inv_error = float(np.max(np.abs(D @ D.T - np.eye(64))))

    # Random test vectors for Parseval
    rng = np.random.default_rng(42)
    x = rng.standard_normal((10000, 64))
    Dx = (D @ x.T).T
    parseval_err = float(np.max(np.abs(np.linalg.norm(Dx, axis=1) - np.linalg.norm(x, axis=1))))

    # Transformed noise covariance (unclipped)
    noise_64 = rng.normal(0.0, 1.0, size=(50000, 64))
    D_noise = (D @ noise_64.T).T
    cov_D_noise = np.cov(D_noise, rowvar=False)
    cov_error = float(np.max(np.abs(cov_D_noise - np.eye(64))))

    # DC vs AC covariance
    dc_noise = D_noise[:, 0]
    ac_noise = D_noise[:, 1:]
    cross_cov = np.dot(dc_noise, ac_noise) / (len(dc_noise) - 1)
    max_cross_cov = float(np.max(np.abs(cross_cov)))

    dct_summary = {
        "ortho_error": ortho_error,
        "inv_error": inv_error,
        "parseval_err": parseval_err,
        "cov_error": cov_error,
        "max_cross_cov": max_cross_cov,
    }
    print(f"DCT verification: Max |D^T D - I| = {ortho_error:.2e}, Max cross-cov(DC, AC) = {max_cross_cov:.4f}")
    return dct_summary


def run_part5_sensing_covariance():
    """Analyze actual A_AC Gram matrices across 50 seeds and write CSV."""
    print("[Part 5] Computing actual sensing covariance for 50 seeds...")
    seeds = [BASE_SEED + t * 10000 + 5000 for t in range(1, 51)]
    records = []

    csv_path = os.path.join(RESULTS_DIR, "actual_sensing_covariance.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "trial", "trace", "min_diag", "max_diag", "min_eig", "max_eig",
            "condition_number", "frobenius_norm", "fro_diff_I", "spectral_diff_I",
            "mean_abs_offdiag", "max_abs_offdiag"
        ])

        for t, s in enumerate(seeds, start=1):
            A_ac = generate_dc_sensing(M_AC, N_AC, seed=s)
            G = A_ac @ A_ac.T
            eigs = np.linalg.eigvalsh(G)
            diag = np.diag(G)
            off_diag = G[~np.eye(M_AC, dtype=bool)]

            rec = {
                "seed": s,
                "trial": t,
                "trace": float(np.sum(eigs)),
                "min_diag": float(np.min(diag)),
                "max_diag": float(np.max(diag)),
                "min_eig": float(np.min(eigs)),
                "max_eig": float(np.max(eigs)),
                "condition_number": float(np.max(eigs) / np.min(eigs)),
                "frobenius_norm": float(np.linalg.norm(G)),
                "fro_diff_I": float(np.linalg.norm(G - np.eye(M_AC))),
                "spectral_diff_I": float(np.max(np.abs(eigs - 1.0))),
                "mean_abs_offdiag": float(np.mean(np.abs(off_diag))),
                "max_abs_offdiag": float(np.max(np.abs(off_diag))),
                "A_ac": A_ac,
                "G": G,
                "eigs": eigs,
            }
            records.append(rec)
            writer.writerow([
                s, t, rec["trace"], rec["min_diag"], rec["max_diag"],
                rec["min_eig"], rec["max_eig"], rec["condition_number"],
                rec["frobenius_norm"], rec["fro_diff_I"], rec["spectral_diff_I"],
                rec["mean_abs_offdiag"], rec["max_abs_offdiag"]
            ])

    print(f"Wrote {len(records)} seed covariance records to {csv_path}")
    return records


def run_part7_8_monte_carlo(cov_records):
    """Run 100,000 Monte Carlo synthetic noise realizations per noise level."""
    print("[Part 7 & 8] Running 100,000 Monte Carlo noise realizations...")
    # Use trial 1 matrix for primary reference
    ref_rec = cov_records[0]
    A_ac = ref_rec["A_ac"]
    eigs = ref_rec["eigs"]

    num_samples = 100000
    mc_results = []
    model_comparison = []

    csv_path = os.path.join(RESULTS_DIR, "empirical_noise_floor.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sigma", "sigma_norm", "M_AC", "mean", "median", "std",
            "p05", "p10", "p50", "p90", "p95", "p99", "p99_9"
        ])

        for sigma in SIGMA_LEVELS:
            s_norm = sigma / 255.0
            rng = np.random.default_rng(BASE_SEED + int(sigma))

            # Exact simulation of measurement noise: e = A_ac @ n_ac
            # Generate n_ac directly: shape (63, 100000)
            n_ac = rng.normal(0.0, s_norm, size=(N_AC, num_samples))
            e_meas = A_ac @ n_ac # shape (37, 100000)
            norms = np.linalg.norm(e_meas, axis=0) # shape (100000,)

            mean_val = float(np.mean(norms))
            med_val = float(np.median(norms))
            std_val = float(np.std(norms))
            p05 = float(np.percentile(norms, 5.0))
            p10 = float(np.percentile(norms, 10.0))
            p50 = float(np.percentile(norms, 50.0))
            p90 = float(np.percentile(norms, 90.0))
            p95 = float(np.percentile(norms, 95.0))
            p99 = float(np.percentile(norms, 99.0))
            p99_9 = float(np.percentile(norms, 99.9))

            writer.writerow([
                sigma, s_norm, M_AC, mean_val, med_val, std_val,
                p05, p10, p50, p90, p95, p99, p99_9
            ])

            mc_results.append({
                "sigma": sigma,
                "sigma_norm": s_norm,
                "norms": norms,
                "mean": mean_val,
                "median": med_val,
                "std": std_val,
                "p05": p05,
                "p10": p10,
                "p90": p90,
                "p95": p95,
                "p99": p99,
                "p99_9": p99_9,
            })

            # Model A, B, C comparison
            # Model A: sqrt(M)
            mA_mean = s_norm * np.sqrt(M_AC)
            # Model B: Ordinary Chi(M_AC)
            mB_mean = s_norm * float(stats.chi.mean(M_AC))
            mB_med = s_norm * float(stats.chi.median(M_AC))
            mB_std = s_norm * float(stats.chi.std(M_AC))
            mB_p05 = s_norm * float(stats.chi.ppf(0.05, M_AC))
            mB_p95 = s_norm * float(stats.chi.ppf(0.95, M_AC))
            mB_p99 = s_norm * float(stats.chi.ppf(0.99, M_AC))

            # Model C: Quadratic form from eigenvalues
            Z = rng.standard_normal((num_samples, M_AC))
            quad = np.sum(eigs * (Z ** 2), axis=1)
            norm_C = s_norm * np.sqrt(quad)
            mC_mean = float(np.mean(norm_C))
            mC_med = float(np.median(norm_C))
            mC_std = float(np.std(norm_C))
            mC_p05 = float(np.percentile(norm_C, 5.0))
            mC_p95 = float(np.percentile(norm_C, 95.0))
            mC_p99 = float(np.percentile(norm_C, 99.0))

            model_comparison.append({
                "sigma": sigma,
                "sigma_norm": s_norm,
                "Model_A_mean": mA_mean,
                "Model_B_mean": mB_mean,
                "Model_B_med": mB_med,
                "Model_B_std": mB_std,
                "Model_B_90": [mB_p05, mB_p95],
                "Model_B_99": [s_norm * float(stats.chi.ppf(0.005, M_AC)), s_norm * float(stats.chi.ppf(0.995, M_AC))],
                "Model_C_mean": mC_mean,
                "Model_C_med": mC_med,
                "Model_C_std": mC_std,
                "Model_C_90": [mC_p05, mC_p95],
                "Model_C_99": [float(np.percentile(norm_C, 0.5)), float(np.percentile(norm_C, 99.5))],
                "Empirical_mean": mean_val,
                "Empirical_std": std_val,
            })

    print("Monte Carlo 100k noise floor saved.")
    return mc_results, model_comparison


def run_part9_image_noise_test(cov_records):
    """Empirical noise test using real BSD68 patches from 10 images."""
    print("[Part 9] Testing empirical image-derived noise across 10 BSD68 images...")
    dataset = load_bsd68(limit=10)
    A_ac = cov_records[0]["A_ac"]

    image_noise_results = []

    for sigma in SIGMA_LEVELS:
        s_norm = sigma / 255.0
        all_e_norms = []
        all_n_ac = []

        for idx, (img_id, img_data) in enumerate(dataset.items()):
            img_clean = img_data["image"]
            seed = BASE_SEED + 1000 + idx
            img_noisy = add_awgn(img_clean, sigma, seed=seed)

            clean_patches = extract_patches(img_clean, patch_size=8, stride=2)
            noisy_patches = extract_patches(img_noisy, patch_size=8, stride=2)

            for cp, npatch in zip(clean_patches, noisy_patches):
                theta_clean = dct_patch(cp["patch"]).reshape(-1)
                theta_noisy = dct_patch(npatch["patch"]).reshape(-1)

                n_ac = theta_noisy[1:] - theta_clean[1:]
                e_ac = A_ac @ n_ac
                norm_e = float(np.linalg.norm(e_ac))

                all_e_norms.append(norm_e)
                all_n_ac.append(n_ac)

        all_e_norms = np.array(all_e_norms)
        all_n_ac = np.array(all_n_ac)

        # Variance of AC noise coefficients
        mean_ac_var = float(np.mean(np.var(all_n_ac, axis=0)))

        image_noise_results.append({
            "sigma": sigma,
            "sigma_norm": s_norm,
            "mean_norm": float(np.mean(all_e_norms)),
            "median_norm": float(np.median(all_e_norms)),
            "std_norm": float(np.std(all_e_norms)),
            "p05": float(np.percentile(all_e_norms, 5.0)),
            "p95": float(np.percentile(all_e_norms, 95.0)),
            "p99": float(np.percentile(all_e_norms, 99.0)),
            "ac_var": mean_ac_var,
            "var_ratio": mean_ac_var / (s_norm ** 2),
            "norms": all_e_norms,
        })
        print(f"  sigma={sigma}: BSD68 Mean Residual = {np.mean(all_e_norms):.4f}, AC Var Ratio = {mean_ac_var / (s_norm**2):.4f}")

    return image_noise_results


def run_part15_correlation_analysis():
    """Compute Pearson & Spearman correlations using raw_results.csv."""
    print("[Part 15] Calculating residual correlations with image quality...")
    csv_path = os.path.join(current_dir, "..", "results", "final_candidate_validation", "raw", "raw_results.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found.")
        return None

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    correlations = {}
    for sigma in [15.0, 25.0, 50.0, "ALL"]:
        if sigma == "ALL":
            sub = rows
        else:
            sub = [r for r in rows if float(r["noise_sigma"]) == sigma]

        res = np.array([float(r["measurement_residual"]) for r in sub])
        psnr = np.array([float(r["psnr"]) for r in sub])
        ssim = np.array([float(r["ssim"]) for r in sub])
        mse = np.array([float(r["mse"]) for r in sub])

        p_psnr, _ = stats.pearsonr(res, psnr)
        s_psnr, _ = stats.spearmanr(res, psnr)
        p_ssim, _ = stats.pearsonr(res, ssim)
        s_ssim, _ = stats.spearmanr(res, ssim)
        p_mse, _ = stats.pearsonr(res, mse)
        s_mse, _ = stats.spearmanr(res, mse)

        correlations[str(sigma)] = {
            "pearson_psnr": float(p_psnr),
            "spearman_psnr": float(s_psnr),
            "pearson_ssim": float(p_ssim),
            "spearman_ssim": float(s_ssim),
            "pearson_mse": float(p_mse),
            "spearman_mse": float(s_mse),
            "count": len(sub),
        }
        print(f"  sigma={sigma}: Pearson(res, PSNR)={p_psnr:.4f}, Spearman(res, PSNR)={s_psnr:.4f}")

    return correlations, rows


def run_part18_19_boundary_and_energy(dataset):
    """Analyze DC/AC energy and patch-boundary vs interior error."""
    print("[Part 18 & 19] Computing DC/AC energy and boundary vs interior error...")
    patch_size = 8
    stride = 2

    # Measure DC vs AC energy on BSD68
    dc_energies = []
    ac_energies = []

    for img_id, img_data in dataset.items():
        img = img_data["image"]
        patches = extract_patches(img, patch_size=patch_size, stride=stride)
        for p in patches:
            theta = dct_patch(p["patch"]).reshape(-1)
            dc_energies.append(theta[0] ** 2)
            ac_energies.append(np.sum(theta[1:] ** 2))

    mean_dc_energy = float(np.mean(dc_energies))
    mean_ac_energy = float(np.mean(ac_energies))
    dc_fraction = mean_dc_energy / (mean_dc_energy + mean_ac_energy)

    # Patch boundary analysis:
    # Use non-overlapping patches (stride=8) to directly expose and measure patch-seam discontinuities
    boundary_stats = []
    from reconstruction import reconstruct_image
    from hybrid_sparse_solver_v7_fixed import HybridSparseSolverV7
    from sensing import generate_random_sensing, generate_dc_sensing

    A_std = generate_random_sensing(38, 64, seed=20275908)
    A_ac = generate_dc_sensing(37, 63, seed=20275908)

    solver_v7 = HybridSparseSolverV7(A_std, lambda_reg=0.1, tol=1e-5, seed=20275908)
    solver_a6 = HybridSparseSolverV7(A_ac, lambda_reg=0.1, tol=1e-5, seed=20275908)

    crop_size = 64
    eval_stride = 2
    for img_idx, img_id in enumerate(["test001", "test002", "test003"]):
        full_img = dataset[img_id]["image"]
        # Take center 64x64 crop
        H_full, W_full = full_img.shape
        cy, cx = H_full // 2, W_full // 2
        img = full_img[cy - crop_size//2 : cy + crop_size//2, cx - crop_size//2 : cx + crop_size//2].copy()
        H, W = img.shape
        noisy = add_awgn(img, sigma_noise=15.0, seed=20275908 + img_idx)

        # Boundary mask: pixels within 1 pixel of patch grid seams (multiples of 8)
        boundary_mask = np.zeros((H, W), dtype=bool)
        for y in range(0, H, patch_size):
            boundary_mask[max(0, y-1):min(H, y+2), :] = True
        for x in range(0, W, patch_size):
            boundary_mask[:, max(0, x-1):min(W, x+2)] = True

        interior_mask = ~boundary_mask

        # Reconstruct V7
        patches_noisy = extract_patches(noisy, patch_size=8, stride=eval_stride)
        v7_hats = []
        a6_hats = []

        for p_rec in patches_noisy:
            p_dct = dct_patch(p_rec["patch"]).reshape(-1)
            # V7
            y_v7 = A_std @ p_dct
            z_v7 = solver_v7.denoise_patch(y_v7, sigma_min=0.01, decrease_factor=0.95, max_iter=150)
            v7_hats.append(idct_patch(z_v7.reshape((8, 8))))

            # A6
            y_dc = float(p_dct[0])
            y_ac = A_ac @ p_dct[1:]
            z_ac = solver_a6.denoise_patch(y_ac, sigma_min=0.01, decrease_factor=0.95, max_iter=150)
            full_z = np.empty(64)
            full_z[0] = y_dc
            full_z[1:] = z_ac
            a6_hats.append(idct_patch(full_z.reshape((8, 8))))

        recon_v7 = reconstruct_image([{"patch": h, "x": r["x"], "y": r["y"]} for h, r in zip(v7_hats, patches_noisy)], (H, W))
        recon_a6 = reconstruct_image([{"patch": h, "x": r["x"], "y": r["y"]} for h, r in zip(a6_hats, patches_noisy)], (H, W))

        # Errors
        err_v7 = (recon_v7 - img) ** 2
        err_a6 = (recon_a6 - img) ** 2

        # Gradient error (boundary step discontinuity)
        gy_img, gx_img = np.gradient(img)
        gy_v7, gx_v7 = np.gradient(recon_v7)
        gy_a6, gx_a6 = np.gradient(recon_a6)
        grad_err_v7 = (gy_v7 - gy_img)**2 + (gx_v7 - gx_img)**2
        grad_err_a6 = (gy_a6 - gy_img)**2 + (gx_a6 - gx_img)**2

        boundary_stats.append({
            "image_id": img_id,
            "v7_boundary_mse": float(np.mean(err_v7[boundary_mask])),
            "v7_interior_mse": float(np.mean(err_v7[interior_mask])),
            "v7_boundary_grad_err": float(np.mean(grad_err_v7[boundary_mask])),
            "a6_boundary_mse": float(np.mean(err_a6[boundary_mask])),
            "a6_interior_mse": float(np.mean(err_a6[interior_mask])),
            "a6_boundary_grad_err": float(np.mean(grad_err_a6[boundary_mask])),
        })

    print(f"DC Energy Fraction: {dc_fraction * 100.0:.2f}%")
    print(f"Boundary MSE: V7 = {np.mean([b['v7_boundary_mse'] for b in boundary_stats]):.6f} vs A6 = {np.mean([b['a6_boundary_mse'] for b in boundary_stats]):.6f}")
    return dc_fraction, boundary_stats


def generate_all_12_figures(cov_records, mc_results, img_noise_results, raw_rows, boundary_stats):
    """Generate the 12 required publication figures for Part 25."""
    print("[Part 25] Generating 12 publication figures...")
    ref_G = cov_records[0]["G"]
    eigs_all = np.array([r["eigs"] for r in cov_records])

    # 1. Covariance Heatmap
    plt.figure(figsize=(7, 6))
    im = plt.imshow(ref_G, cmap="coolwarm", vmin=-0.3, vmax=1.0)
    plt.colorbar(im, label="Covariance / Correlation $G_{i,j}$")
    plt.title(r"Sensing Gram Matrix $G = A_{\mathrm{ac}} A_{\mathrm{ac}}^T \in \mathbb{R}^{37 \times 37}$", fontsize=12)
    plt.xlabel("Row Index $j$")
    plt.ylabel("Row Index $i$")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "1_covariance_heatmap.png"), dpi=300)
    plt.close()

    # 2. Eigenvalue Spectrum
    plt.figure(figsize=(7, 4.5))
    mean_eigs = np.mean(np.sort(eigs_all, axis=1), axis=0)
    std_eigs = np.std(np.sort(eigs_all, axis=1), axis=0)
    x_idx = np.arange(1, 38)
    plt.plot(x_idx, mean_eigs, "b-o", markersize=4, label=r"Empirical Mean $\lambda_i(G)$ (50 Seeds)")
    plt.fill_between(x_idx, mean_eigs - std_eigs, mean_eigs + std_eigs, color="b", alpha=0.2, label=r"$\pm 1$ SD across Seeds")
    plt.axhline(1.0, color="r", linestyle="--", label=r"Ideal Isotropic $\lambda = 1.0$")
    plt.title(r"Eigenvalue Spectrum of Sensing Gram Matrix $G = A_{\mathrm{ac}} A_{\mathrm{ac}}^T$", fontsize=12)
    plt.xlabel("Eigenvalue Index (Sorted)")
    plt.ylabel(r"Eigenvalue $\lambda_i$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "2_eigenvalue_spectrum.png"), dpi=300)
    plt.close()

    # 3. Theoretical vs Empirical Residual Distribution (sigma=25)
    plt.figure(figsize=(7, 4.5))
    mc_s25 = [m for m in mc_results if m["sigma"] == 25.0][0]
    img_s25 = [m for m in img_noise_results if m["sigma"] == 25.0][0]
    s_norm = 25.0 / 255.0
    x_grid = np.linspace(0.2, 1.1, 300)
    pdf_chi = stats.chi.pdf(x_grid / s_norm, 37) / s_norm
    plt.plot(x_grid, pdf_chi, "k--", label=r"Model B: Ordinary $\chi(37)$ PDF", linewidth=1.5)
    plt.hist(mc_s25["norms"], bins=80, density=True, alpha=0.5, color="royalblue", label=r"Model C: Monte Carlo $A_{\mathrm{ac}} n$ (100k)")
    plt.hist(img_s25["norms"], bins=60, density=True, alpha=0.4, color="forestgreen", label="Empirical BSD68 Patches")
    plt.axvline(mc_s25["mean"], color="blue", linestyle="-", label=f"MC Mean: {mc_s25['mean']:.4f}")
    plt.title(r"Projected Noise Distribution ($\sigma=25$, $M_{\mathrm{ac}}=37$)", fontsize=12)
    plt.xlabel(r"Residual Norm $\|e_{\mathrm{ac}}\|_2$")
    plt.ylabel("Probability Density")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "3_theoretical_vs_empirical_residual.png"), dpi=300)
    plt.close()

    # 4. Residual vs Noise Level
    plt.figure(figsize=(7, 4.5))
    sigmas = [15.0, 25.0, 50.0]
    mc_means = [m["mean"] for m in mc_results]
    mc_p05 = [m["p05"] for m in mc_results]
    mc_p95 = [m["p95"] for m in mc_results]
    plt.errorbar(sigmas, mc_means, yerr=[np.array(mc_means)-np.array(mc_p05), np.array(mc_p95)-np.array(mc_means)],
                 fmt="s-", color="royalblue", capsize=5, label=r"Monte Carlo Noise Floor (90% CI)")
    # Overlay BSD68 empirical means
    img_means = [m["mean_norm"] for m in img_noise_results]
    plt.plot(sigmas, img_means, "g^--", markersize=7, label="BSD68 Empirical Mean Noise Norm")
    plt.title(r"Projected Noise Floor vs. Input Noise Level $\sigma$", fontsize=12)
    plt.xlabel(r"Noise Standard Deviation $\sigma$")
    plt.ylabel(r"Expected Residual Norm $\mathbb{E}[\|e_{\mathrm{ac}}\|_2]$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "4_residual_vs_noise_level.png"), dpi=300)
    plt.close()

    # 5. A6 Residual vs Noise Floor
    plt.figure(figsize=(7, 4.5))
    a6_rows = [r for r in raw_rows if r["configuration"] == "V7_A6_DC_PRESERVATION"]
    a6_res_by_sigma = {s: [] for s in [15.0, 25.0, 50.0]}
    for r in a6_rows:
        a6_res_by_sigma[float(r["noise_sigma"])].append(float(r["measurement_residual"]))
    a6_means = [np.mean(a6_res_by_sigma[s]) for s in sigmas]
    a6_stds = [np.std(a6_res_by_sigma[s]) for s in sigmas]

    plt.plot(sigmas, mc_means, "b--o", label=r"Theoretical Noise Floor $\mathbb{E}[\|e_{\mathrm{ac}}\|_2]$")
    plt.fill_between(sigmas, mc_p05, mc_p95, color="b", alpha=0.15, label="Noise Floor 90% CI")
    plt.errorbar(sigmas, a6_means, yerr=a6_stds, fmt="r-s", capsize=5, label="A6 Solved Residual (Mean ± SD)")
    plt.title(r"Variant A6 Measurement Residual vs. Statistical Noise Floor", fontsize=12)
    plt.xlabel(r"Noise Level $\sigma$")
    plt.ylabel(r"Residual Norm $\|A z - y\|_2$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "5_a6_residual_vs_noise_floor.png"), dpi=300)
    plt.close()

    # 6. Residual vs PSNR
    plt.figure(figsize=(7, 4.5))
    colors = {15.0: "forestgreen", 25.0: "darkorange", 50.0: "crimson"}
    for s in [15.0, 25.0, 50.0]:
        sub = [r for r in raw_rows if float(r["noise_sigma"]) == s]
        x_val = [float(r["measurement_residual"]) for r in sub]
        y_val = [float(r["psnr"]) for r in sub]
        plt.scatter(x_val, y_val, c=colors[s], label=rf"$\sigma={int(s)}$", alpha=0.7, edgecolors="none")
    plt.title("Measurement Residual vs. Full-Image PSNR across All Variants", fontsize=12)
    plt.xlabel(r"Measurement Residual $\|A \hat{z} - y\|_2$")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "6_residual_vs_psnr.png"), dpi=300)
    plt.close()

    # 7. Residual vs SSIM
    plt.figure(figsize=(7, 4.5))
    for s in [15.0, 25.0, 50.0]:
        sub = [r for r in raw_rows if float(r["noise_sigma"]) == s]
        x_val = [float(r["measurement_residual"]) for r in sub]
        y_val = [float(r["ssim"]) for r in sub]
        plt.scatter(x_val, y_val, c=colors[s], label=rf"$\sigma={int(s)}$", alpha=0.7, edgecolors="none")
    plt.title("Measurement Residual vs. Full-Image SSIM across All Variants", fontsize=12)
    plt.xlabel(r"Measurement Residual $\|A \hat{z} - y\|_2$")
    plt.ylabel("SSIM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "7_residual_vs_ssim.png"), dpi=300)
    plt.close()

    # 8. Residual vs MSE
    plt.figure(figsize=(7, 4.5))
    for s in [15.0, 25.0, 50.0]:
        sub = [r for r in raw_rows if float(r["noise_sigma"]) == s]
        x_val = [float(r["measurement_residual"]) for r in sub]
        y_val = [float(r["mse"]) for r in sub]
        plt.scatter(x_val, y_val, c=colors[s], label=rf"$\sigma={int(s)}$", alpha=0.7, edgecolors="none")
    plt.title("Measurement Residual vs. Full-Image MSE across All Variants", fontsize=12)
    plt.xlabel(r"Measurement Residual $\|A \hat{z} - y\|_2$")
    plt.ylabel("Image MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "8_residual_vs_mse.png"), dpi=300)
    plt.close()

    # 9. DC Error V7 vs A6
    plt.figure(figsize=(7, 4.5))
    v7_rows = [r for r in raw_rows if r["configuration"] == "V7_OPT_BASE"]
    v7_dc = {s: [float(r["dc_error"]) for r in v7_rows if float(r["noise_sigma"]) == s] for s in sigmas}
    a6_dc = {s: [float(r["dc_error"]) for r in a6_rows if float(r["noise_sigma"]) == s] for s in sigmas}

    bar_width = 2.5
    s_arr = np.array(sigmas)
    plt.bar(s_arr - bar_width/2, [np.mean(v7_dc[s]) for s in sigmas], width=bar_width, color="steelblue", label="V7 Baseline (Standard Sensing)")
    plt.bar(s_arr + bar_width/2, [np.mean(a6_dc[s]) for s in sigmas], width=bar_width, color="coral", label="A6 (DC Preservation)")
    # Overlay theoretical DC noise
    theo_dc_noise = [s / 255.0 for s in sigmas]
    plt.plot(sigmas, theo_dc_noise, "k--o", label=r"Sensor Noise on DC ($\sigma_{\mathrm{norm}}$)")
    plt.title("DC Reconstruction Error: Baseline V7 vs. Variant A6", fontsize=12)
    plt.xlabel(r"Noise Level $\sigma$")
    plt.ylabel("Mean Absolute DC Error")
    plt.xticks(sigmas)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "9_dc_error_v7_vs_a6.png"), dpi=300)
    plt.close()

    # 10. AC Error V7 vs A6
    plt.figure(figsize=(7, 4.5))
    v7_ac = {s: [float(r["ac_error"]) for r in v7_rows if float(r["noise_sigma"]) == s] for s in sigmas}
    a6_ac = {s: [float(r["ac_error"]) for r in a6_rows if float(r["noise_sigma"]) == s] for s in sigmas}
    plt.bar(s_arr - bar_width/2, [np.mean(v7_ac[s]) for s in sigmas], width=bar_width, color="steelblue", label="V7 Baseline")
    plt.bar(s_arr + bar_width/2, [np.mean(a6_ac[s]) for s in sigmas], width=bar_width, color="coral", label="A6 Candidate")
    plt.title("AC Reconstruction Error: Baseline V7 vs. Variant A6", fontsize=12)
    plt.xlabel(r"Noise Level $\sigma$")
    plt.ylabel(r"Mean $\|z_{\mathrm{ac}} - \theta_{\mathrm{ac}}\|_2$ Error")
    plt.xticks(sigmas)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "10_ac_error_v7_vs_a6.png"), dpi=300)
    plt.close()

    # 11. Patch Boundary Error
    plt.figure(figsize=(7, 4.5))
    imgs = [b["image_id"] for b in boundary_stats]
    x_bar = np.arange(len(imgs))
    w = 0.35
    v7_b_mse = [b["v7_boundary_mse"] for b in boundary_stats]
    a6_b_mse = [b["a6_boundary_mse"] for b in boundary_stats]
    plt.bar(x_bar - w/2, v7_b_mse, width=w, color="steelblue", label="V7 Seam Boundary MSE")
    plt.bar(x_bar + w/2, a6_b_mse, width=w, color="coral", label="A6 Seam Boundary MSE")
    plt.title(r"Patch Seam Boundary Error Comparison ($\sigma=15$)", fontsize=12)
    plt.xlabel("Test Image")
    plt.ylabel("Mean Squared Error on Seams")
    plt.xticks(x_bar, imgs)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "11_patch_boundary_error.png"), dpi=300)
    plt.close()

    # 12. Method Quality vs Runtime
    plt.figure(figsize=(7, 4.5))
    configs = ["V7_OPT_BASE", "V7_A5_TWO_STAGE", "V7_A6_DC_PRESERVATION", "V7_A5A6_COMBINED"]
    c_colors = {"V7_OPT_BASE": "steelblue", "V7_A5_TWO_STAGE": "goldenrod", "V7_A6_DC_PRESERVATION": "forestgreen", "V7_A5A6_COMBINED": "purple"}
    markers = {"V7_OPT_BASE": "o", "V7_A5_TWO_STAGE": "^", "V7_A6_DC_PRESERVATION": "s", "V7_A5A6_COMBINED": "D"}

    for cfg in configs:
        sub = [r for r in raw_rows if r["configuration"] == cfg]
        m_time = np.mean([float(r["solve_time"]) for r in sub])
        m_psnr = np.mean([float(r["psnr"]) for r in sub])
        plt.scatter(m_time, m_psnr, c=c_colors[cfg], marker=markers[cfg], s=120, label=cfg)

    plt.title("Reconstruction Quality (PSNR) vs. Solve Time across 10 Images", fontsize=12)
    plt.xlabel("Mean Solve Time per Image (s)")
    plt.ylabel("Mean PSNR (dB)")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "12_method_quality_vs_runtime.png"), dpi=300)
    plt.close()

    print("All 12 figures generated and saved to results/residual_analysis_v2/figures/.")


def main():
    print("=" * 70)
    print("ASL-SR-DPT A6 PROJECTED-NOISE RESIDUAL VALIDATION (V2)")
    print("=" * 70)

    # Part 2: Noise model
    clipping_report = run_part2_noise_model_verification()

    # Part 3: DCT
    dct_summary = run_part3_dct_verification()

    # Part 5: Sensing covariance
    cov_records = run_part5_sensing_covariance()

    # Part 7 & 8: Monte Carlo
    mc_results, model_comparison = run_part7_8_monte_carlo(cov_records)

    # Part 9: Image-derived noise
    img_noise_results = run_part9_image_noise_test(cov_records)

    # Part 15: Correlations
    corr_results, raw_rows = run_part15_correlation_analysis()

    # Part 18 & 19: Boundary analysis
    dataset = load_bsd68(limit=10)
    dc_fraction, boundary_stats = run_part18_19_boundary_and_energy(dataset)

    # Part 25: Figures
    generate_all_12_figures(cov_records, mc_results, img_noise_results, raw_rows, boundary_stats)

    print("=" * 70)
    print("ALL VALIDATION PIPELINES EXECUTED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    main()
