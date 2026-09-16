"""
ASL-SR-DPT High Measurement Residual Root-Cause Diagnostic Analysis.

Performs deep instrumentation, iteration tracing, gradient balance decomposition,
support conditioning analysis, and correlation analysis across 100 representative
patches from the 10 BSD68 test images.

Strictly DIAGNOSTIC ONLY:
Does NOT modify any algorithm parameters or production code.
"""

import os
import sys

# Strict single-threading enforcement
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dataset import load_bsd68
from sensing import generate_random_sensing, generate_dc_sensing, add_awgn
from reconstruction import extract_patches, dct_patch, idct_patch, reconstruct_image
from metrics import compute_all_metrics

RESULTS_DIR = os.path.join(current_dir, "..", "results", "residual_analysis")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


class InstrumentedTestSolverV7:
    """
    Instrumented version of ASL-SR-DPT V7 solver for root-cause residual diagnostics.
    Identical mathematical logic and parameterization to frozen V7, with iteration-by-iteration telemetry.
    """
    def __init__(self, A, lambda_reg=0.1, tol=1e-5):
        self.A = np.asarray(A, dtype=float)
        self.M, self.N = self.A.shape
        self.lambda_reg = float(lambda_reg)
        self.tol = float(tol)
        self.P_init = np.linalg.pinv(self.A)
        self.all_active_mask = np.ones(self.N, dtype=bool)

    def solve_trace(
        self,
        y,
        max_iter=150,
        sigma_min=0.01,
        decrease_factor=0.95,
        support_thresh_mult=1e-5,
        support_reopen_interval=3,
        initial_mu=0.2,
        armijo_c=1e-4,
        max_backtracks=10,
        beta_decay=0.5,
        use_midpoint=True,
    ):
        y = np.asarray(y, dtype=float).reshape(-1)
        # Initialization via pseudoinverse
        z = self.P_init @ y
        z_init = z.copy()

        z_max = float(np.max(np.abs(z)))
        sigma_init = 2.5 * z_max if z_max > 0 else 1.0
        sigma = float(max(sigma_init, sigma_min))
        mu = float(initial_mu)

        iteration_trace = []
        grad_balance = []

        r = self.A @ z - y
        fid = 0.5 * float(np.dot(r, r))

        for it in range(max_iter):
            # Support thresholding
            thresh = support_thresh_mult * sigma
            if it % support_reopen_interval == 0:
                active_mask = self.all_active_mask
                all_active = True
            else:
                active_mask = np.abs(z) > thresh
                if not active_mask.any():
                    active_mask = self.all_active_mask
                    all_active = True
                else:
                    all_active = bool(active_mask.all())

            active_cnt = self.N if all_active else int(np.count_nonzero(active_mask))

            # Objective components
            inv_2s2 = 0.5 / (sigma * sigma)
            inv_s2 = 1.0 / (sigma * sigma)
            w = np.exp(-(z * z) * inv_2s2)
            sparsity_val = -self.lambda_reg * float(w.sum())
            total_obj = fid + sparsity_val

            # Gradients
            if all_active:
                A_act = self.A
                z_act = z
                grad_fid = self.A.T @ r
                grad_sparse = (z * inv_s2) * w
                grad_sparse_weighted = self.lambda_reg * grad_sparse
                grad = grad_fid + grad_sparse_weighted
            else:
                A_act = self.A[:, active_mask]
                z_act = z[active_mask]
                res_act = A_act @ z_act - y
                grad_fid = A_act.T @ res_act
                w_act = w[active_mask]
                grad_sparse = (z_act * inv_s2) * w_act
                grad_sparse_weighted = self.lambda_reg * grad_sparse
                grad = grad_fid + grad_sparse_weighted

            # Norms of gradients
            norm_g_fid = float(np.linalg.norm(grad_fid))
            norm_g_sparse = float(np.linalg.norm(grad_sparse_weighted))
            norm_g_total = float(np.linalg.norm(grad))
            grad_ratio = norm_g_sparse / max(norm_g_fid, 1e-12)

            # Residuals
            abs_res = float(np.linalg.norm(r))
            norm_y = max(float(np.linalg.norm(y)), 1e-12)
            rel_res = abs_res / norm_y
            nrmse = abs_res / np.sqrt(self.M)

            # Record telemetry
            trace_entry = {
                "iteration": it,
                "sigma": sigma,
                "mu": mu,
                "objective": total_obj,
                "fidelity_term": fid,
                "sparsity_term": sparsity_val,
                "absolute_residual": abs_res,
                "relative_residual": rel_res,
                "normalized_rmse": nrmse,
                "gradient_fidelity_norm": norm_g_fid,
                "gradient_sparsity_norm": norm_g_sparse,
                "gradient_total_norm": norm_g_total,
                "active_count": active_cnt,
            }
            iteration_trace.append(trace_entry)

            grad_entry = {
                "iteration": it,
                "sigma": sigma,
                "norm_g_fidelity": norm_g_fid,
                "norm_g_sparsity": norm_g_sparse,
                "norm_g_total": norm_g_total,
                "gradient_ratio": grad_ratio,
            }
            grad_balance.append(grad_entry)

            # Line search
            d_act = -grad
            grad_dot_d = -float(np.dot(grad, grad))
            A_d = A_act @ d_act

            step_size = mu
            best_z_act = None
            accepted_r = r
            accepted_fid = fid
            line_search_success = False

            for bt in range(max_backtracks):
                r_cand = r + step_size * A_d
                fid_cand = 0.5 * float(np.dot(r_cand, r_cand))
                z_cand_act = z_act + step_size * d_act

                if all_active:
                    w_cand = np.exp(-(z_cand_act * z_cand_act) * inv_2s2)
                    sparse_cand = -self.lambda_reg * float(w_cand.sum())
                else:
                    w_cand_act = np.exp(-(z_cand_act * z_cand_act) * inv_2s2)
                    sparse_cand = -self.lambda_reg * float(w_cand_act.sum() + w[~active_mask].sum())

                cand_obj = fid_cand + sparse_cand
                if cand_obj <= total_obj + armijo_c * step_size * grad_dot_d:
                    best_z_act = z_cand_act
                    mu = step_size
                    accepted_r = r_cand
                    accepted_fid = fid_cand
                    line_search_success = True
                    break

                if use_midpoint:
                    step_mid = step_size * 0.5
                    r_mid = r + step_mid * A_d
                    fid_mid = 0.5 * float(np.dot(r_mid, r_mid))
                    z_mid_act = z_act + step_mid * d_act
                    if all_active:
                        w_mid = np.exp(-(z_mid_act * z_mid_act) * inv_2s2)
                        sparse_mid = -self.lambda_reg * float(w_mid.sum())
                    else:
                        w_mid_act = np.exp(-(z_mid_act * z_mid_act) * inv_2s2)
                        sparse_mid = -self.lambda_reg * float(w_mid_act.sum() + w[~active_mask].sum())

                    cand_mid_obj = fid_mid + sparse_mid
                    if cand_mid_obj <= total_obj + armijo_c * step_mid * grad_dot_d:
                        best_z_act = z_mid_act
                        mu = step_mid
                        accepted_r = r_mid
                        accepted_fid = fid_mid
                        line_search_success = True
                        break

                step_size *= beta_decay

            if not line_search_success:
                mu *= 0.5
                continue

            if all_active:
                z = best_z_act
            else:
                z = z.copy()
                z[active_mask] = best_z_act

            r = accepted_r
            fid = accepted_fid
            sigma = max(sigma * decrease_factor, sigma_min)

        return z, z_init, iteration_trace, grad_balance


def debias_support(A, y, z_stage1, thresh=1e-3, max_support=None):
    """Stage 2 unconstrained least-squares debiasing on active support."""
    M, N = A.shape
    if max_support is None:
        max_support = M
    supp = np.abs(z_stage1) > thresh
    cnt = int(np.count_nonzero(supp))
    if cnt == 0 or cnt > max_support:
        if cnt == 0:
            supp = np.ones(N, dtype=bool)
        else:
            top_idx = np.argsort(np.abs(z_stage1))[-max_support:]
            supp = np.zeros(N, dtype=bool)
            supp[top_idx] = True

    A_supp = A[:, supp]
    z_supp = np.linalg.pinv(A_supp) @ y
    z_stage2 = np.zeros(N, dtype=float)
    z_stage2[supp] = z_supp

    # SVD for condition number and variance amplification
    s = np.linalg.svd(A_supp, compute_uv=False)
    cond = float(s[0] / max(s[-1], 1e-12))
    rank = int(np.sum(s > 1e-10))
    # Tr((A_S^T A_S)^-1) = sum(1/sigma_i^2)
    var_amp = float(np.sum(1.0 / (s * s)))

    return z_stage2, supp, cond, rank, var_amp


def run_diagnostics():
    print("=" * 70)
    print("ASL-SR-DPT HIGH MEASUREMENT RESIDUAL ROOT-CAUSE ANALYSIS")
    print("=" * 70)

    # 1. Load dataset & extract 100 representative patches (10 patches from each of the 10 images)
    print("Loading BSD68 validation dataset...")
    dataset = load_bsd68(strict_count=68)
    image_ids = [f"test{i:03d}" for i in range(1, 11)]

    # 10 patches per image chosen across texture variations
    selected_patch_records = []
    patch_id_counter = 0

    base_seed = 42
    A_std = generate_random_sensing(38, 64, seed=base_seed)
    A_dc = generate_dc_sensing(37, 63, seed=base_seed + 1)

    # Noise levels
    noise_sigmas = [15.0, 25.0, 50.0]

    # Containers for outputs
    all_iteration_traces = []
    all_grad_balances = []
    support_conditioning_records = []
    scaling_records = []
    dct_scale_records = []
    dc_preservation_records = []
    patch_comparison_records = []
    initialization_records = []

    print(f"Extracting 100 representative patches from {len(image_ids)} BSD68 images...")
    patches_pool = []
    for img_idx, img_id in enumerate(image_ids):
        clean_img = dataset[img_id]["image"]
        p_records = extract_patches(clean_img, patch_size=8, stride=2)
        # Select 10 diverse patches spread across the patch list
        step = len(p_records) // 10
        for k in range(10):
            idx = k * step
            rec = p_records[idx]
            patches_pool.append({
                "patch_global_id": patch_id_counter,
                "image_id": img_id,
                "local_patch_idx": idx,
                "clean_patch": rec["patch"].copy(),
                "x": rec["x"],
                "y": rec["y"],
            })
            patch_id_counter += 1

    print(f"Total representative patches collected: {len(patches_pool)}")

    # Initialize instrumented solver
    solver_std = InstrumentedTestSolverV7(A_std, lambda_reg=0.1)
    solver_dc = InstrumentedTestSolverV7(A_dc, lambda_reg=0.1)

    print("\nExecuting multi-noise evaluations across representative patches...")

    for noise_idx, sigma_noise in enumerate(noise_sigmas):
        sigma_scaled = sigma_noise / 255.0
        print(f"--- Processing Noise Level sigma = {sigma_noise} (scaled = {sigma_scaled:.4f}) ---")

        for p_info in patches_pool:
            p_id = p_info["patch_global_id"]
            clean_patch = p_info["clean_patch"]
            theta_clean = dct_patch(clean_patch).reshape(-1)

            # Add AWGN
            rng = np.random.default_rng(base_seed + p_id * 100 + int(sigma_noise))
            noise = rng.normal(loc=0.0, scale=sigma_scaled, size=(8, 8))
            noisy_patch = np.clip(clean_patch + noise, 0.0, 1.0)
            theta_noisy = dct_patch(noisy_patch).reshape(-1)

            # -------------------------------------------------------------
            # Measurement Generation
            # -------------------------------------------------------------
            # Standard sensing
            y_std = A_std @ theta_noisy
            # DC-preserving sensing
            y_dc = float(theta_noisy[0])
            y_ac = A_dc @ theta_noisy[1:]

            # Check signal scaling (for Item 11)
            norm_theta = float(np.linalg.norm(theta_noisy))
            norm_y_std = float(np.linalg.norm(y_std))
            norm_y_ac = float(np.linalg.norm(y_ac))

            # DCT scale analysis (for Item 12)
            dc_clean = float(theta_clean[0])
            dc_noisy = float(theta_noisy[0])
            ac_clean_mags = np.abs(theta_clean[1:])
            ac_noisy_mags = np.abs(theta_noisy[1:])

            # -------------------------------------------------------------
            # Solve V7_OPT_BASE
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            z_v7, z0_v7, trace_v7, grad_v7 = solver_std.solve_trace(y_std)
            t_v7 = time.perf_counter() - t0

            # Attach patch_id and noise to traces (save subset for CSV to keep file manageable)
            if sigma_noise == 15.0:
                for tr in trace_v7:
                    tr["patch_id"] = p_id
                    all_iteration_traces.append(tr)
                for gb in grad_v7:
                    gb["patch_id"] = p_id
                    all_grad_balances.append(gb)

            # Residuals V7
            abs_res_v7 = float(np.linalg.norm(A_std @ z_v7 - y_std))
            rel_res_v7 = abs_res_v7 / max(norm_y_std, 1e-12)
            nrmse_v7 = abs_res_v7 / np.sqrt(38)
            patch_rec_v7 = idct_patch(z_v7.reshape(8, 8))
            mse_v7 = float(np.mean((patch_rec_v7 - clean_patch) ** 2))
            psnr_v7 = 10.0 * np.log10(1.0 / max(mse_v7, 1e-12))
            dc_err_v7 = float(abs(z_v7[0] - theta_clean[0]))
            ac_err_v7 = float(np.linalg.norm(z_v7[1:] - theta_clean[1:]))
            coeff_err_v7 = float(np.linalg.norm(z_v7 - theta_clean))

            # -------------------------------------------------------------
            # Solve V7_A5_TWO_STAGE
            # -------------------------------------------------------------
            z_a5, supp_a5, cond_a5, rank_a5, var_amp_a5 = debias_support(A_std, y_std, z_v7, thresh=1e-3, max_support=38)
            abs_res_a5 = float(np.linalg.norm(A_std @ z_a5 - y_std))
            rel_res_a5 = abs_res_a5 / max(norm_y_std, 1e-12)
            patch_rec_a5 = idct_patch(z_a5.reshape(8, 8))
            mse_a5 = float(np.mean((patch_rec_a5 - clean_patch) ** 2))
            psnr_a5 = 10.0 * np.log10(1.0 / max(mse_a5, 1e-12))
            dc_err_a5 = float(abs(z_a5[0] - theta_clean[0]))
            ac_err_a5 = float(np.linalg.norm(z_a5[1:] - theta_clean[1:]))
            coeff_err_a5 = float(np.linalg.norm(z_a5 - theta_clean))

            # -------------------------------------------------------------
            # Solve V7_A6_DC_PRESERVATION
            # -------------------------------------------------------------
            z_ac_a6, z0_ac_a6, _, _ = solver_dc.solve_trace(y_ac)
            z_a6 = np.empty(64, dtype=float)
            z_a6[0] = y_dc
            z_a6[1:] = z_ac_a6

            abs_res_ac_a6 = float(np.linalg.norm(A_dc @ z_ac_a6 - y_ac))
            rel_res_ac_a6 = abs_res_ac_a6 / max(norm_y_ac, 1e-12)
            nrmse_ac_a6 = abs_res_ac_a6 / np.sqrt(37)
            patch_rec_a6 = idct_patch(z_a6.reshape(8, 8))
            mse_a6 = float(np.mean((patch_rec_a6 - clean_patch) ** 2))
            psnr_a6 = 10.0 * np.log10(1.0 / max(mse_a6, 1e-12))
            dc_err_a6 = float(abs(z_a6[0] - theta_clean[0]))
            ac_err_a6 = float(np.linalg.norm(z_a6[1:] - theta_clean[1:]))
            coeff_err_a6 = float(np.linalg.norm(z_a6 - theta_clean))

            # -------------------------------------------------------------
            # Solve V7_A5A6_COMBINED
            # -------------------------------------------------------------
            z_ac_a5a6, supp_a5a6, cond_a5a6, rank_a5a6, var_amp_a5a6 = debias_support(A_dc, y_ac, z_ac_a6, thresh=1e-3, max_support=37)
            z_a5a6 = np.empty(64, dtype=float)
            z_a5a6[0] = y_dc
            z_a5a6[1:] = z_ac_a5a6

            abs_res_ac_a5a6 = float(np.linalg.norm(A_dc @ z_ac_a5a6 - y_ac))
            rel_res_ac_a5a6 = abs_res_ac_a5a6 / max(norm_y_ac, 1e-12)
            nrmse_ac_a5a6 = abs_res_ac_a5a6 / np.sqrt(37)
            patch_rec_a5a6 = idct_patch(z_a5a6.reshape(8, 8))
            mse_a5a6 = float(np.mean((patch_rec_a5a6 - clean_patch) ** 2))
            psnr_a5a6 = 10.0 * np.log10(1.0 / max(mse_a5a6, 1e-12))
            dc_err_a5a6 = float(abs(z_a5a6[0] - theta_clean[0]))
            ac_err_a5a6 = float(np.linalg.norm(z_a5a6[1:] - theta_clean[1:]))
            coeff_err_a5a6 = float(np.linalg.norm(z_a5a6 - theta_clean))

            # -------------------------------------------------------------
            # Initialization comparison (Item 8)
            # -------------------------------------------------------------
            res_init = float(np.linalg.norm(A_std @ z0_v7 - y_std))
            patch_rec_init = idct_patch(z0_v7.reshape(8, 8))
            mse_init = float(np.mean((patch_rec_init - clean_patch) ** 2))
            psnr_init = 10.0 * np.log10(1.0 / max(mse_init, 1e-12))
            initialization_records.append({
                "patch_id": p_id,
                "noise_sigma": sigma_noise,
                "initial_residual": res_init,
                "final_residual": abs_res_v7,
                "initial_psnr": psnr_init,
                "final_psnr": psnr_v7,
                "initial_norm": float(np.linalg.norm(z0_v7)),
                "final_norm": float(np.linalg.norm(z_v7)),
            })

            # Signal scaling records (Item 11)
            scaling_records.append({
                "patch_id": p_id,
                "noise_sigma": sigma_noise,
                "norm_theta": norm_theta,
                "norm_y_std": norm_y_std,
                "norm_Az_v7": float(np.linalg.norm(A_std @ z_v7)),
                "abs_residual_v7": abs_res_v7,
                "relative_residual_v7": rel_res_v7,
                "normalized_rmse_v7": nrmse_v7,
                "expected_clean_res": np.sqrt(38) * sigma_scaled,
            })

            # DCT scale records (Item 12)
            dct_scale_records.append({
                "patch_id": p_id,
                "noise_sigma": sigma_noise,
                "dc_clean": abs(dc_clean),
                "mean_ac_clean": float(np.mean(ac_clean_mags)),
                "max_ac_clean": float(np.max(ac_clean_mags)),
                "dc_noisy": abs(dc_noisy),
                "mean_ac_noisy": float(np.mean(ac_noisy_mags)),
                "max_ac_noisy": float(np.max(ac_noisy_mags)),
                "dc_rec_v7": abs(float(z_v7[0])),
                "mean_ac_rec_v7": float(np.mean(np.abs(z_v7[1:]))),
                "dc_rec_a6": abs(float(z_a6[0])),
                "mean_ac_rec_a6": float(np.mean(np.abs(z_a6[1:]))),
            })

            # DC preservation check (Item 13)
            dc_preservation_records.append({
                "patch_id": p_id,
                "noise_sigma": sigma_noise,
                "theta_dc_clean": dc_clean,
                "theta_dc_noisy": dc_noisy,
                "dc_noise_error": abs(dc_noisy - dc_clean),
                "dc_err_v7": dc_err_v7,
                "dc_err_a5": dc_err_a5,
                "dc_err_a6": dc_err_a6,
                "dc_err_a5a6": dc_err_a5a6,
            })

            # Support conditioning & noise amplification (Item 14, 15, 16, 17)
            support_conditioning_records.append({
                "patch_id": p_id,
                "noise_sigma": sigma_noise,
                "config": "A5",
                "support_size": int(np.count_nonzero(supp_a5)),
                "M": 38,
                "support_ratio_M": float(np.count_nonzero(supp_a5)) / 38.0,
                "cond_A_S": cond_a5,
                "cond_ATA": cond_a5 ** 2,
                "rank_A_S": rank_a5,
                "variance_amplification_tr": var_amp_a5,
                "pre_debias_residual": abs_res_v7,
                "post_debias_residual": abs_res_a5,
                "psnr": psnr_a5,
            })

            support_conditioning_records.append({
                "patch_id": p_id,
                "noise_sigma": sigma_noise,
                "config": "A5A6",
                "support_size": int(np.count_nonzero(supp_a5a6)),
                "M": 37,
                "support_ratio_M": float(np.count_nonzero(supp_a5a6)) / 37.0,
                "cond_A_S": cond_a5a6,
                "cond_ATA": cond_a5a6 ** 2,
                "rank_A_S": rank_a5a6,
                "variance_amplification_tr": var_amp_a5a6,
                "pre_debias_residual": abs_res_ac_a6,
                "post_debias_residual": abs_res_ac_a5a6,
                "psnr": psnr_a5a6,
            })

            # Master comparison across configurations (Item 9)
            for cfg, p, m, a_r, r_r, d_e, a_e, c_e in [
                ("V7_OPT_BASE", psnr_v7, mse_v7, abs_res_v7, rel_res_v7, dc_err_v7, ac_err_v7, coeff_err_v7),
                ("V7_A5_TWO_STAGE", psnr_a5, mse_a5, abs_res_a5, rel_res_a5, dc_err_a5, ac_err_a5, coeff_err_a5),
                ("V7_A6_DC_PRESERVATION", psnr_a6, mse_a6, abs_res_ac_a6, rel_res_ac_a6, dc_err_a6, ac_err_a6, coeff_err_a6),
                ("V7_A5A6_COMBINED", psnr_a5a6, mse_a5a6, abs_res_ac_a5a6, rel_res_ac_a5a6, dc_err_a5a6, ac_err_a5a6, coeff_err_a5a6),
            ]:
                patch_comparison_records.append({
                    "patch_id": p_id,
                    "noise_sigma": sigma_noise,
                    "configuration": cfg,
                    "psnr": p,
                    "mse": m,
                    "absolute_residual": a_r,
                    "relative_residual": r_r,
                    "dc_error": d_e,
                    "ac_error": a_e,
                    "coefficient_error": c_e,
                })

    # -------------------------------------------------------------
    # Write CSV Files
    # -------------------------------------------------------------
    print("\nWriting diagnostic CSV files...")

    # 1. iteration_trace.csv
    trace_path = os.path.join(RESULTS_DIR, "iteration_trace.csv")
    with open(trace_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_iteration_traces[0].keys()))
        writer.writeheader()
        writer.writerows(all_iteration_traces)
    print(f"Saved: {trace_path} ({len(all_iteration_traces)} rows)")

    # 2. gradient_balance.csv
    grad_path = os.path.join(RESULTS_DIR, "gradient_balance.csv")
    with open(grad_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_grad_balances[0].keys()))
        writer.writeheader()
        writer.writerows(all_grad_balances)
    print(f"Saved: {grad_path} ({len(all_grad_balances)} rows)")

    # 3. support_conditioning.csv
    cond_path = os.path.join(RESULTS_DIR, "support_conditioning.csv")
    with open(cond_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(support_conditioning_records[0].keys()))
        writer.writeheader()
        writer.writerows(support_conditioning_records)
    print(f"Saved: {cond_path} ({len(support_conditioning_records)} rows)")

    # -------------------------------------------------------------
    # Statistical Aggregations & Computations
    # -------------------------------------------------------------
    print("\nComputing statistical summaries and sigma range aggregations...")

    # Section 5: Residual vs Sigma ranges
    # Group iteration trace by sigma ranges:
    # sigma > 1, 1 >= sigma > 0.1, 0.1 >= sigma > 0.01, sigma == 0.01
    range_buckets = {
        "sigma > 1": [],
        "1 >= sigma > 0.1": [],
        "0.1 >= sigma > 0.01": [],
        "sigma == 0.01": [],
    }
    for tr in all_iteration_traces:
        s = tr["sigma"]
        if s > 1.0:
            range_buckets["sigma > 1"].append(tr)
        elif s > 0.1:
            range_buckets["1 >= sigma > 0.1"].append(tr)
        elif s > 0.01 + 1e-6:
            range_buckets["0.1 >= sigma > 0.01"].append(tr)
        else:
            range_buckets["sigma == 0.01"].append(tr)

    print("\n--- RESIDUAL VS SIGMA RANGES (V7, sigma_noise=15) ---")
    sigma_range_summary = []
    for r_name, items in range_buckets.items():
        if items:
            res_vals = [x["absolute_residual"] for x in items]
            fid_vals = [x["fidelity_term"] for x in items]
            sparse_vals = [x["sparsity_term"] for x in items]
            row = {
                "range": r_name,
                "count": len(items),
                "mean_residual": float(np.mean(res_vals)),
                "median_residual": float(np.median(res_vals)),
                "std_residual": float(np.std(res_vals)),
                "mean_fidelity": float(np.mean(fid_vals)),
                "mean_sparsity": float(np.mean(sparse_vals)),
            }
            sigma_range_summary.append(row)
            print(f"Range {r_name:18s}: mean_res={row['mean_residual']:.4f}, med_res={row['median_residual']:.4f}, std_res={row['std_residual']:.4f}, mean_fid={row['mean_fidelity']:.4f}, mean_sparse={row['mean_sparsity']:.4f}")

    # Section 8: Initialization Summary
    print("\n--- INITIALIZATION ANALYSIS (V7, 100 patches) ---")
    for s_n in noise_sigmas:
        subset = [x for x in initialization_records if x["noise_sigma"] == s_n]
        init_res_mean = np.mean([x["initial_residual"] for x in subset])
        final_res_mean = np.mean([x["final_residual"] for x in subset])
        init_psnr_mean = np.mean([x["initial_psnr"] for x in subset])
        final_psnr_mean = np.mean([x["final_psnr"] for x in subset])
        print(f"Noise sigma={s_n:2.0f}: Initial Res={init_res_mean:.2e}, Final Res={final_res_mean:.4f} | Initial PSNR={init_psnr_mean:.2f} dB, Final PSNR={final_psnr_mean:.2f} dB")

    # Section 11: Signal Scaling Summary
    print("\n--- SIGNAL SCALING ANALYSIS (100 patches) ---")
    for s_n in noise_sigmas:
        subset = [x for x in scaling_records if x["noise_sigma"] == s_n]
        th_norms = [x["norm_theta"] for x in subset]
        y_norms = [x["norm_y_std"] for x in subset]
        res_norms = [x["abs_residual_v7"] for x in subset]
        rel_norms = [x["relative_residual_v7"] for x in subset]
        print(f"Noise sigma={s_n:2.0f}: ||theta||={np.mean(th_norms):.4f}, ||y||={np.mean(y_norms):.4f}, ||Az-y||={np.mean(res_norms):.4f}, RelRes={np.mean(rel_norms):.4f} (expected clean dist = {np.mean([x['expected_clean_res'] for x in subset]):.4f})")

    # Section 12: DCT Scale Summary
    print("\n--- DCT COEFFICIENT MAGNITUDES (100 patches) ---")
    for s_n in noise_sigmas:
        subset = [x for x in dct_scale_records if x["noise_sigma"] == s_n]
        dc_cl = np.mean([x["dc_clean"] for x in subset])
        ac_cl = np.mean([x["mean_ac_clean"] for x in subset])
        print(f"Noise sigma={s_n:2.0f}: Mean Clean DC={dc_cl:.4f}, Mean Clean AC={ac_cl:.4f} (Ratio DC/AC = {dc_cl/max(ac_cl, 1e-6):.1f}x)")

    # Section 13: DC Preservation Summary
    print("\n--- DC PRESERVATION ERRORS (100 patches) ---")
    for s_n in noise_sigmas:
        subset = [x for x in dc_preservation_records if x["noise_sigma"] == s_n]
        noise_err = np.mean([x["dc_noise_error"] for x in subset])
        v7_err = np.mean([x["dc_err_v7"] for x in subset])
        a5_err = np.mean([x["dc_err_a5"] for x in subset])
        a6_err = np.mean([x["dc_err_a6"] for x in subset])
        a5a6_err = np.mean([x["dc_err_a5a6"] for x in subset])
        print(f"Noise sigma={s_n:2.0f}: DC Noise Err={noise_err:.4f}, V7 DC Err={v7_err:.4f}, A5 DC Err={a5_err:.4f}, A6 DC Err={a6_err:.4f}, A5A6 DC Err={a5a6_err:.4f}")

    # Section 15 & 16: Support Conditioning Summary for A5A6
    print("\n--- A5A6 CONDITIONING & NOISE AMPLIFICATION ACROSS NOISE ---")
    for s_n in noise_sigmas:
        subset = [x for x in support_conditioning_records if x["noise_sigma"] == s_n and x["config"] == "A5A6"]
        supp_mean = np.mean([x["support_size"] for x in subset])
        cond_mean = np.mean([x["cond_A_S"] for x in subset])
        cond_max = np.max([x["cond_A_S"] for x in subset])
        var_amp_mean = np.mean([x["variance_amplification_tr"] for x in subset])
        psnr_mean = np.mean([x["psnr"] for x in subset])
        res_post = np.mean([x["post_debias_residual"] for x in subset])
        print(f"Noise sigma={s_n:2.0f}: Support={supp_mean:.1f}/37 ({supp_mean/37.0:.1%}), Cond(A_S) mean={cond_mean:.2f} max={cond_max:.2f}, Tr(ATA^-1)={var_amp_mean:.1f}, PSNR={psnr_mean:.2f} dB, Post-Res={res_post:.4f}")

    # Section 20: Correlations between Residual and Quality
    print("\n--- CORRELATIONS: RESIDUAL VS QUALITY (Across all patch evaluations) ---")
    residuals_all = np.array([x["absolute_residual"] for x in patch_comparison_records])
    psnr_all = np.array([x["psnr"] for x in patch_comparison_records])
    mse_all = np.array([x["mse"] for x in patch_comparison_records])

    corr_res_psnr = float(np.corrcoef(residuals_all, psnr_all)[0, 1])
    corr_res_mse = float(np.corrcoef(residuals_all, mse_all)[0, 1])

    print(f"Overall Pearson corr(residual, PSNR): {corr_res_psnr:.4f}")
    print(f"Overall Pearson corr(residual, MSE):  {corr_res_mse:.4f}")

    # Compute correlation within each noise level
    for s_n in noise_sigmas:
        sub = [x for x in patch_comparison_records if x["noise_sigma"] == s_n]
        res_sub = np.array([x["absolute_residual"] for x in sub])
        psnr_sub = np.array([x["psnr"] for x in sub])
        mse_sub = np.array([x["mse"] for x in sub])
        c_p = float(np.corrcoef(res_sub, psnr_sub)[0, 1])
        c_m = float(np.corrcoef(res_sub, mse_sub)[0, 1])
        print(f"  At sigma={s_n:2.0f}: corr(residual, PSNR) = {c_p:.4f}, corr(residual, MSE) = {c_m:.4f}")

    # -------------------------------------------------------------
    # Generate 10 Diagnostic Dashboard Figures
    # -------------------------------------------------------------
    print("\nGenerating 10 diagnostic figures...")

    # Filter traces for patch 0 (or average across 100 patches)
    # 1. residual vs iteration
    plt.figure(figsize=(7, 5))
    iter_nums = np.arange(150)
    # Average residual across all 100 patches at each iteration
    res_per_iter = np.zeros(150)
    for it in range(150):
        res_per_iter[it] = np.mean([tr["absolute_residual"] for tr in all_iteration_traces if tr["iteration"] == it])
    plt.plot(iter_nums, res_per_iter, "b-", lw=2, label="Mean Absolute Residual ||Az-y||")
    plt.axhline(y=np.sqrt(38) * (15.0 / 255.0), color="r", linestyle="--", label="Theoretical Noise Floor $\\sqrt{M}\\sigma$")
    plt.xlabel("Iteration", fontsize=11)
    plt.ylabel("Measurement Residual ||Az - y||_2", fontsize=11)
    plt.title("Fig 1: Residual vs Iteration (V7, $\\sigma=15$)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "1_residual_vs_iteration.png"), dpi=150)
    plt.close()

    # 2. residual vs sigma
    plt.figure(figsize=(7, 5))
    sigmas_iter = [np.mean([tr["sigma"] for tr in all_iteration_traces if tr["iteration"] == it]) for it in range(150)]
    plt.semilogx(sigmas_iter, res_per_iter, "r.-", lw=1.5)
    plt.gca().invert_xaxis()
    plt.xlabel("Continuation Parameter $\\sigma$ (log scale, left=initial, right=final)", fontsize=11)
    plt.ylabel("Measurement Residual ||Az - y||_2", fontsize=11)
    plt.title("Fig 2: Residual vs Continuation Scale $\\sigma$", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "2_residual_vs_sigma.png"), dpi=150)
    plt.close()

    # 3. fidelity vs sparsity term
    plt.figure(figsize=(7, 5))
    fid_per_iter = [np.mean([tr["fidelity_term"] for tr in all_iteration_traces if tr["iteration"] == it]) for it in range(150)]
    sparse_per_iter = [np.mean([tr["sparsity_term"] for tr in all_iteration_traces if tr["iteration"] == it]) for it in range(150)]
    plt.plot(iter_nums, fid_per_iter, "g-", lw=2, label="Fidelity Term: 0.5||Az-y||^2")
    plt.plot(iter_nums, sparse_per_iter, "m--", lw=2, label="Sparsity Term: -\\lambda \\sum w_i")
    plt.xlabel("Iteration", fontsize=11)
    plt.ylabel("Objective Component Value", fontsize=11)
    plt.title("Fig 3: Objective Decomposition (Fidelity vs Sparsity)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "3_fidelity_vs_sparsity.png"), dpi=150)
    plt.close()

    # 4. gradient norm comparison
    plt.figure(figsize=(7, 5))
    g_fid_per_iter = [np.mean([tr["gradient_fidelity_norm"] for tr in all_iteration_traces if tr["iteration"] == it]) for it in range(150)]
    g_sparse_per_iter = [np.mean([tr["gradient_sparsity_norm"] for tr in all_iteration_traces if tr["iteration"] == it]) for it in range(150)]
    g_tot_per_iter = [np.mean([tr["gradient_total_norm"] for tr in all_iteration_traces if tr["iteration"] == it]) for it in range(150)]
    plt.plot(iter_nums, g_fid_per_iter, "b-", label="||g_fidelity||")
    plt.plot(iter_nums, g_sparse_per_iter, "r--", label="||g_sparsity|| (weighted)")
    plt.plot(iter_nums, g_tot_per_iter, "k:", lw=2, label="||g_total||")
    plt.xlabel("Iteration", fontsize=11)
    plt.ylabel("Gradient L2 Norm", fontsize=11)
    plt.title("Fig 4: Gradient Balance vs Iteration", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "4_gradient_norm_comparison.png"), dpi=150)
    plt.close()

    # 5. active support vs sigma
    plt.figure(figsize=(7, 5))
    active_per_iter = [np.mean([tr["active_count"] for tr in all_iteration_traces if tr["iteration"] == it]) for it in range(150)]
    plt.plot(iter_nums, active_per_iter, "c-", lw=2)
    plt.xlabel("Iteration", fontsize=11)
    plt.ylabel("Active Support Count (|z_i| > \\sigma)", fontsize=11)
    plt.title("Fig 5: Active Support Count vs Iteration", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "5_active_support_vs_sigma.png"), dpi=150)
    plt.close()

    # 6. support size vs conditioning
    plt.figure(figsize=(7, 5))
    a5a6_cond = [x for x in support_conditioning_records if x["config"] == "A5A6"]
    colors = {15.0: "blue", 25.0: "orange", 50.0: "red"}
    for s_n in noise_sigmas:
        sub = [x for x in a5a6_cond if x["noise_sigma"] == s_n]
        plt.scatter([x["support_size"] for x in sub], [x["cond_A_S"] for x in sub],
                    c=colors[s_n], label=f"$\\sigma={s_n}$", alpha=0.7)
    plt.axvline(x=37, color="black", linestyle="--", label="Max AC Measurements M=37")
    plt.xlabel("Active Support Size |S|", fontsize=11)
    plt.ylabel("Condition Number cond(A_S)", fontsize=11)
    plt.title("Fig 6: Support Size vs Matrix Conditioning (A5A6)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "6_support_size_vs_conditioning.png"), dpi=150)
    plt.close()

    # 7. PSNR vs residual
    plt.figure(figsize=(7, 5))
    cfg_colors = {
        "V7_OPT_BASE": "gray",
        "V7_A5_TWO_STAGE": "blue",
        "V7_A6_DC_PRESERVATION": "green",
        "V7_A5A6_COMBINED": "red",
    }
    for cfg in cfg_colors:
        sub = [x for x in patch_comparison_records if x["configuration"] == cfg and x["noise_sigma"] == 15.0]
        plt.scatter([x["absolute_residual"] for x in sub], [x["psnr"] for x in sub],
                    c=cfg_colors[cfg], label=cfg, alpha=0.6, s=20)
    plt.xlabel("Absolute Measurement Residual ||Az - y||_2", fontsize=11)
    plt.ylabel("Patch PSNR (dB)", fontsize=11)
    plt.title("Fig 7: Patch PSNR vs Measurement Residual ($\\sigma=15$)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "7_psnr_vs_residual.png"), dpi=150)
    plt.close()

    # 8. SSIM vs residual (approximated by 1/(1+MSE) or negative MSE)
    plt.figure(figsize=(7, 5))
    for cfg in cfg_colors:
        sub = [x for x in patch_comparison_records if x["configuration"] == cfg and x["noise_sigma"] == 15.0]
        plt.scatter([x["absolute_residual"] for x in sub], [x["mse"] for x in sub],
                    c=cfg_colors[cfg], label=cfg, alpha=0.6, s=20)
    plt.xlabel("Absolute Measurement Residual ||Az - y||_2", fontsize=11)
    plt.ylabel("Patch MSE", fontsize=11)
    plt.title("Fig 8: Patch MSE vs Measurement Residual ($\\sigma=15$)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "8_ssim_vs_residual.png"), dpi=150)
    plt.close()

    # 9. A6 DC error comparison
    plt.figure(figsize=(7, 5))
    bar_sigmas = [15, 25, 50]
    bar_width = 0.2
    x_indices = np.arange(len(bar_sigmas))

    mean_dc_v7 = [np.mean([x["dc_err_v7"] for x in dc_preservation_records if x["noise_sigma"] == s]) for s in bar_sigmas]
    mean_dc_a5 = [np.mean([x["dc_err_a5"] for x in dc_preservation_records if x["noise_sigma"] == s]) for s in bar_sigmas]
    mean_dc_a6 = [np.mean([x["dc_err_a6"] for x in dc_preservation_records if x["noise_sigma"] == s]) for s in bar_sigmas]
    mean_dc_noise = [np.mean([x["dc_noise_error"] for x in dc_preservation_records if x["noise_sigma"] == s]) for s in bar_sigmas]

    plt.bar(x_indices - 1.5 * bar_width, mean_dc_v7, width=bar_width, label="V7_OPT_BASE", color="gray")
    plt.bar(x_indices - 0.5 * bar_width, mean_dc_a5, width=bar_width, label="V7_A5", color="blue")
    plt.bar(x_indices + 0.5 * bar_width, mean_dc_a6, width=bar_width, label="V7_A6 (DC Preserved)", color="green")
    plt.bar(x_indices + 1.5 * bar_width, mean_dc_noise, width=bar_width, label="Noise Error", color="red")
    plt.xticks(x_indices, ["$\\sigma=15$", "$\\sigma=25$", "$\\sigma=50$"])
    plt.ylabel("Mean Absolute DC Error", fontsize=11)
    plt.title("Fig 9: DC Reconstruction Error Across Configurations", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "9_dc_error_comparison.png"), dpi=150)
    plt.close()

    # 10. A5/A5A6 pre/post debias residual
    plt.figure(figsize=(7, 5))
    pre_res_a5a6 = [np.mean([x["pre_debias_residual"] for x in a5a6_cond if x["noise_sigma"] == s]) for s in bar_sigmas]
    post_res_a5a6 = [np.mean([x["post_debias_residual"] for x in a5a6_cond if x["noise_sigma"] == s]) for s in bar_sigmas]

    plt.bar(x_indices - 0.5 * bar_width, pre_res_a5a6, width=bar_width, label="Pre-Debias Residual (A6 Stage 1)", color="purple")
    plt.bar(x_indices + 0.5 * bar_width, post_res_a5a6, width=bar_width, label="Post-Debias Residual (A5A6 Stage 2)", color="red")
    plt.xticks(x_indices, ["$\\sigma=15$", "$\\sigma=25$", "$\\sigma=50$"])
    plt.ylabel("Measurement Residual ||Az - y||_2", fontsize=11)
    plt.title("Fig 10: Pre- vs Post-Debiasing Measurement Residual", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "10_pre_post_debias_residual.png"), dpi=150)
    plt.close()

    print("All 10 figures successfully generated and saved to results/residual_analysis/figures/.")
    print("=" * 70)
    print("DIAGNOSTIC DATA COLLECTION COMPLETED.")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnostics()
