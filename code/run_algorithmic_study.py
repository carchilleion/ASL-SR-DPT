"""
Driver for ASL-SR-DPT Controlled Algorithmic Improvement Phase.
Executes Variants A1 through A6, collects full diagnostics, computes comparative metrics,
and saves all outputs in results/algorithmic_variants/.
"""

import os
import sys

# Strict single-thread enforcement
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import json
import csv
import numpy as np
from skimage.metrics import structural_similarity as compute_skimage_ssim

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dataset import load_bsd68
from sensing import generate_random_sensing, generate_dc_sensing, add_awgn
from reconstruction import extract_patches, dct_patch, idct_patch, reconstruct_image
from metrics import compute_all_metrics
from algorithmic_variants import (
    SolverV7OptBase,
    SolverV7A1AdaptiveExit,
    SolverV7A2ScaleCoupled,
    SolverV7A3SupportThreshold,
    SolverV7A4Continuation,
    SolverV7A5TwoStage,
    SolverV7A6DCPreservation,
)

RESULTS_DIR = os.path.join(current_dir, "..", "results", "algorithmic_variants")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_deterministic_validation_data(num_patches=500):
    """
    Loads fixed deterministic validation dataset:
    test001.png, sigma_noise=15, seeds matched.
    """
    data_dir = os.path.join(current_dir, "..", "data", "BSD68")
    images = load_bsd68(data_dir)
    clean_img = images["test001"]["image"]
    H, W = clean_img.shape

    base_seed = 20260908
    M, N = 38, 64

    # Matrices
    A_std = generate_random_sensing(M, N, seed=base_seed)
    A_dc = generate_dc_sensing(37, 63, seed=base_seed)

    # AWGN noise
    noisy_img = add_awgn(clean_img, sigma_noise=15.0, seed=base_seed)

    clean_records = extract_patches(clean_img, patch_size=8, stride=2)[:num_patches]
    noisy_records = extract_patches(noisy_img, patch_size=8, stride=2)[:num_patches]

    clean_patches = [r["patch"] for r in clean_records]
    noisy_patches = [r["patch"] for r in noisy_records]

    clean_thetas = np.array([dct_patch(p).reshape(-1) for p in clean_patches])  # (500, 64)
    noisy_thetas = np.array([dct_patch(p).reshape(-1) for p in noisy_patches])  # (500, 64)

    # Standard measurements: y = A @ theta
    Y_std = (A_std @ noisy_thetas.T).T  # (500, 38)

    # DC measurements: y_ac = A_dc @ theta[1:], y_dc = theta[0]
    Y_ac = (A_dc @ noisy_thetas[:, 1:].T).T  # (500, 37)
    Y_dc = noisy_thetas[:, 0]  # (500,)

    return {
        "clean_img": clean_img,
        "noisy_img": noisy_img,
        "clean_records": clean_records,
        "clean_patches": clean_patches,
        "clean_thetas": clean_thetas,
        "noisy_thetas": noisy_thetas,
        "A_std": A_std,
        "A_dc": A_dc,
        "Y_std": Y_std,
        "Y_ac": Y_ac,
        "Y_dc": Y_dc,
        "H": H,
        "W": W,
    }


def evaluate_solver_on_patches(solver_fn, Y_data, clean_thetas, clean_patches, clean_records, A_matrix, H, W):
    """
    Evaluates a solver across the 500 patches with precise timing and metric extraction.
    """
    num_patches = len(Y_data)
    recovered_thetas = []
    diagnostics_list = []

    # Strictly time the solver execution
    t0 = time.perf_counter()
    for i in range(num_patches):
        y = Y_data[i]
        z_rec, diag = solver_fn(y, i)
        recovered_thetas.append(z_rec)
        diagnostics_list.append(diag)
    total_time = time.perf_counter() - t0

    recovered_thetas = np.array(recovered_thetas)  # (500, 64)
    ms_per_patch = (total_time / num_patches) * 1000.0

    # 1. Coefficient Domain Errors
    coeff_errors = np.linalg.norm(recovered_thetas - clean_thetas, axis=1)
    mean_coeff_error = float(np.mean(coeff_errors))

    dc_errors = np.abs(recovered_thetas[:, 0] - clean_thetas[:, 0])
    mean_dc_error = float(np.mean(dc_errors))

    ac_errors = np.linalg.norm(recovered_thetas[:, 1:] - clean_thetas[:, 1:], axis=1)
    mean_ac_error = float(np.mean(ac_errors))

    # 2. Measurement Residuals
    if A_matrix.shape[1] == 64:
        residuals = np.linalg.norm(recovered_thetas @ A_matrix.T - Y_data, axis=1)
    else:  # AC only
        residuals = np.linalg.norm(recovered_thetas[:, 1:] @ A_matrix.T - Y_data, axis=1)
    mean_residual = float(np.mean(residuals))

    # 3. Patch-level Quality Metrics
    patch_mses = []
    patch_ssims = []
    for i in range(num_patches):
        p_clean = clean_patches[i]
        p_rec = idct_patch(recovered_thetas[i].reshape(8, 8))
        mse_i = float(np.mean((p_clean - p_rec) ** 2))
        ssim_i = float(compute_skimage_ssim(p_clean, p_rec, data_range=1.0))
        patch_mses.append(mse_i)
        patch_ssims.append(ssim_i)

    mean_patch_mse = float(np.mean(patch_mses))
    mean_patch_ssim = float(np.mean(patch_ssims))
    overall_patch_psnr = float(10.0 * np.log10(1.0 / max(mean_patch_mse, 1e-15)))

    # 4. Iteration Diagnostics
    iterations = [d["iterations"] for d in diagnostics_list]
    mean_iters = float(np.mean(iterations))
    median_iters = float(np.median(iterations))
    max_iters = int(np.max(iterations))
    pct_max_iter = float(np.mean([it == 150 for it in iterations]) * 100.0)

    final_sigmas = [d["final_sigma"] for d in diagnostics_list]
    mean_final_sigma = float(np.mean(final_sigmas))

    active_ratios = [d.get("active_support_ratio", 1.0) for d in diagnostics_list]
    mean_active_ratio = float(np.mean(active_ratios))

    active_counts = [d.get("mean_active_count", 64.0) for d in diagnostics_list]
    mean_active_count = float(np.mean(active_counts))

    min_active_counts = [d.get("min_active_count", 64) for d in diagnostics_list]
    min_active_count = int(np.min(min_active_counts))

    mean_mus = [d.get("mean_mu", 0.2) for d in diagnostics_list]
    avg_mu = float(np.mean(mean_mus))

    failed_ls = int(np.sum([d.get("failed_line_searches", 0) for d in diagnostics_list]))
    backtracks = int(np.sum([d.get("backtracking_steps", 0) for d in diagnostics_list]))

    objectives = [d["objective"][-1] if d["objective"] else 0.0 for d in diagnostics_list]
    mean_objective = float(np.mean(objectives))

    # Gradient audits aggregate
    audit_summary = {}
    for tag in ["init", "sigma_1", "sigma_01", "sigma_001"]:
        tag_ratios = [d["gradient_audits"][tag]["gradient_ratio"] for d in diagnostics_list if tag in d.get("gradient_audits", {})]
        tag_fids = [d["gradient_audits"][tag]["norm_fidelity"] for d in diagnostics_list if tag in d.get("gradient_audits", {})]
        tag_spars = [d["gradient_audits"][tag]["norm_sparsity"] for d in diagnostics_list if tag in d.get("gradient_audits", {})]
        if tag_ratios:
            audit_summary[tag] = {
                "ratio": float(np.mean(tag_ratios)),
                "fidelity": float(np.mean(tag_fids)),
                "sparsity": float(np.mean(tag_spars)),
            }

    return {
        "total_time_sec": total_time,
        "runtime_ms_patch": ms_per_patch,
        "psnr": overall_patch_psnr,
        "ssim": mean_patch_ssim,
        "mse": mean_patch_mse,
        "measurement_residual": mean_residual,
        "coefficient_error": mean_coeff_error,
        "dc_error": mean_dc_error,
        "ac_error": mean_ac_error,
        "mean_iterations": mean_iters,
        "median_iterations": median_iters,
        "max_iterations": max_iters,
        "pct_max_iter": pct_max_iter,
        "final_sigma": mean_final_sigma,
        "active_support_ratio": mean_active_ratio,
        "mean_active_count": mean_active_count,
        "min_active_count": min_active_count,
        "mean_mu": avg_mu,
        "failed_line_searches": failed_ls,
        "backtracking_steps": backtracks,
        "mean_objective": mean_objective,
        "audit_summary": audit_summary,
        "recovered_thetas": recovered_thetas,
    }


def run_algorithmic_study():
    print("==================================================")
    print("ASL-SR-DPT CONTROLLED ALGORITHMIC IMPROVEMENT STUDY")
    print("==================================================")

    data = load_deterministic_validation_data(num_patches=500)
    A_std = data["A_std"]
    A_dc = data["A_dc"]
    Y_std = data["Y_std"]
    Y_ac = data["Y_ac"]
    Y_dc = data["Y_dc"]
    clean_thetas = data["clean_thetas"]
    clean_patches = data["clean_patches"]
    clean_records = data["clean_records"]
    H, W = data["H"], data["W"]

    master_records = []

    # =========================================================================
    # STEP 1: ESTABLISH V7_OPT_BASE REFERENCE
    # =========================================================================
    print("\n--- Phase 1: Establishing V7_OPT_BASE Reference ---")
    base_solver = SolverV7OptBase(A_std, lambda_reg=0.1, tol=1e-5)
    res_base = evaluate_solver_on_patches(
        lambda y, i: base_solver.denoise_patch(y, return_diagnostics=True),
        Y_std, clean_thetas, clean_patches, clean_records, A_std, H, W
    )

    base_dir = os.path.join(RESULTS_DIR, "V7_OPT_BASE")
    os.makedirs(base_dir, exist_ok=True)
    ref_csv = os.path.join(base_dir, "reference.csv")
    with open(ref_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in res_base.items():
            if k not in ["recovered_thetas", "audit_summary"]:
                w.writerow([k, v])
    print(f"V7_OPT_BASE: {res_base['runtime_ms_patch']:.3f} ms/patch, PSNR: {res_base['psnr']:.2f} dB, Residual: {res_base['measurement_residual']:.4f}, Mean Iters: {res_base['mean_iterations']:.1f}")
    print(f"Saved: {ref_csv}")

    master_records.append({
        "variant": "V7_OPT_BASE",
        "description": "Approved V7 fixed mathematics with Category-A optimizations",
        "runtime_ms_patch": res_base["runtime_ms_patch"],
        "speedup_vs_V7": 1.0,
        "runtime_reduction_pct": 0.0,
        "psnr": res_base["psnr"],
        "ssim": res_base["ssim"],
        "mse": res_base["mse"],
        "measurement_residual": res_base["measurement_residual"],
        "coefficient_error": res_base["coefficient_error"],
        "dc_error": res_base["dc_error"],
        "ac_error": res_base["ac_error"],
        "iterations": res_base["mean_iterations"],
        "final_sigma": res_base["final_sigma"],
        "active_support_ratio": res_base["active_support_ratio"],
        "mean_active_count": res_base["mean_active_count"],
        "failed_line_searches": res_base["failed_line_searches"],
        "gradient_ratio_001": res_base["audit_summary"].get("sigma_001", {}).get("ratio", 0.0),
        "decision": "REFERENCE_BASELINE",
    })

    # =========================================================================
    # STEP 2: VARIANT A1 (ADAPTIVE EARLY EXIT)
    # =========================================================================
    print("\n--- Phase 2: Evaluating Variant A1 (Adaptive Early Exit) ---")
    a1_dir = os.path.join(RESULTS_DIR, "V7_A1")
    os.makedirs(a1_dir, exist_ok=True)
    a1_csv = os.path.join(a1_dir, "a1_results.csv")

    patience_grid = [1, 2, 3, 5]
    best_a1 = None

    with open(a1_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patience", "runtime_ms_patch", "speedup", "psnr", "ssim", "mse", "measurement_residual", "coefficient_error", "iterations", "final_sigma", "decision"])

        for pat in patience_grid:
            solver_a1 = SolverV7A1AdaptiveExit(A_std, lambda_reg=0.1, tol=1e-5, patience=pat, min_iter=5)
            res_a1 = evaluate_solver_on_patches(
                lambda y, i: solver_a1.denoise_patch(y, return_diagnostics=True),
                Y_std, clean_thetas, clean_patches, clean_records, A_std, H, W
            )
            speedup = res_base["runtime_ms_patch"] / res_a1["runtime_ms_patch"]
            reduct_pct = (res_base["runtime_ms_patch"] - res_a1["runtime_ms_patch"]) / res_base["runtime_ms_patch"] * 100.0

            # Quality trade-off check: PSNR change
            d_psnr = res_a1["psnr"] - res_base["psnr"]
            decision = "PROMISING" if (speedup >= 1.10 and d_psnr >= -0.10) else ("TRADE_OFF" if speedup >= 1.10 else "REJECT")

            w.writerow([pat, f"{res_a1['runtime_ms_patch']:.3f}", f"{speedup:.3f}x", f"{res_a1['psnr']:.4f}", f"{res_a1['ssim']:.4f}", f"{res_a1['mse']:.6f}", f"{res_a1['measurement_residual']:.4f}", f"{res_a1['coefficient_error']:.4f}", f"{res_a1['mean_iterations']:.1f}", f"{res_a1['final_sigma']:.4f}", decision])
            print(f"  Patience K={pat}: {res_a1['runtime_ms_patch']:.3f} ms ({speedup:.2f}x), PSNR: {res_a1['psnr']:.2f} dB (dPSNR: {d_psnr:+.2f} dB), Mean Iters: {res_a1['mean_iterations']:.1f}, Final Sigma: {res_a1['final_sigma']:.4f} -> {decision}")

            if pat == 2:  # Default recommended patience
                best_a1 = res_a1
                master_records.append({
                    "variant": "V7_A1_ADAPTIVE_EXIT (K=2)",
                    "description": "Consecutive stability early exit (patience=2 iterations)",
                    "runtime_ms_patch": res_a1["runtime_ms_patch"],
                    "speedup_vs_V7": speedup,
                    "runtime_reduction_pct": reduct_pct,
                    "psnr": res_a1["psnr"],
                    "ssim": res_a1["ssim"],
                    "mse": res_a1["mse"],
                    "measurement_residual": res_a1["measurement_residual"],
                    "coefficient_error": res_a1["coefficient_error"],
                    "dc_error": res_a1["dc_error"],
                    "ac_error": res_a1["ac_error"],
                    "iterations": res_a1["mean_iterations"],
                    "final_sigma": res_a1["final_sigma"],
                    "active_support_ratio": res_a1["active_support_ratio"],
                    "mean_active_count": res_a1["mean_active_count"],
                    "failed_line_searches": res_a1["failed_line_searches"],
                    "gradient_ratio_001": res_a1["audit_summary"].get("sigma_001", {}).get("ratio", 0.0),
                    "decision": decision,
                })

    # =========================================================================
    # STEP 3: VARIANT A2 (SCALE-COUPLED REGULARIZATION)
    # =========================================================================
    print("\n--- Phase 3: Evaluating Variant A2 (Scale-Coupled Regularization) ---")
    a2_dir = os.path.join(RESULTS_DIR, "V7_A2")
    os.makedirs(a2_dir, exist_ok=True)
    a2_csv = os.path.join(a2_dir, "a2_results.csv")
    grad_csv = os.path.join(a2_dir, "gradient_balance.csv")

    solver_a2 = SolverV7A2ScaleCoupled(A_std, lambda_0=0.1, tol=1e-5)
    res_a2 = evaluate_solver_on_patches(
        lambda y, i: solver_a2.denoise_patch(y, return_diagnostics=True),
        Y_std, clean_thetas, clean_patches, clean_records, A_std, H, W
    )
    speedup_a2 = res_base["runtime_ms_patch"] / res_a2["runtime_ms_patch"]
    reduct_a2 = (res_base["runtime_ms_patch"] - res_a2["runtime_ms_patch"]) / res_base["runtime_ms_patch"] * 100.0
    d_psnr_a2 = res_a2["psnr"] - res_base["psnr"]
    d_res_a2 = res_a2["measurement_residual"] - res_base["measurement_residual"]

    decision_a2 = "PROMISING" if (d_psnr_a2 >= 0.5 and d_res_a2 <= -0.1) else ("REJECT_QUALITY_COLLAPSE" if d_psnr_a2 < -1.0 else "REJECT")

    with open(a2_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "V7_OPT_BASE", "V7_A2_SCALE_COUPLED", "difference", "decision"])
        w.writerow(["runtime_ms_patch", f"{res_base['runtime_ms_patch']:.3f}", f"{res_a2['runtime_ms_patch']:.3f}", f"{res_a2['runtime_ms_patch'] - res_base['runtime_ms_patch']:.3f}", "—"])
        w.writerow(["speedup", "1.000x", f"{speedup_a2:.3f}x", f"+{speedup_a2 - 1.0:.3f}x", "—"])
        w.writerow(["PSNR", f"{res_base['psnr']:.4f}", f"{res_a2['psnr']:.4f}", f"{d_psnr_a2:+.4f}", "—"])
        w.writerow(["SSIM", f"{res_base['ssim']:.4f}", f"{res_a2['ssim']:.4f}", f"{res_a2['ssim'] - res_base['ssim']:+.4f}", "—"])
        w.writerow(["MSE", f"{res_base['mse']:.6f}", f"{res_a2['mse']:.6f}", f"{res_a2['mse'] - res_base['mse']:+.6f}", "—"])
        w.writerow(["measurement_residual", f"{res_base['measurement_residual']:.4f}", f"{res_a2['measurement_residual']:.4f}", f"{d_res_a2:+.4f}", "—"])
        w.writerow(["coefficient_error", f"{res_base['coefficient_error']:.4f}", f"{res_a2['coefficient_error']:.4f}", f"{res_a2['coefficient_error'] - res_base['coefficient_error']:+.4f}", "—"])
        w.writerow(["dc_error", f"{res_base['dc_error']:.4f}", f"{res_a2['dc_error']:.4f}", f"{res_a2['dc_error'] - res_base['dc_error']:+.4f}", "—"])
        w.writerow(["ac_error", f"{res_base['ac_error']:.4f}", f"{res_a2['ac_error']:.4f}", f"{res_a2['ac_error'] - res_base['ac_error']:+.4f}", "—"])
        w.writerow(["iterations", f"{res_base['mean_iterations']:.1f}", f"{res_a2['mean_iterations']:.1f}", f"{res_a2['mean_iterations'] - res_base['mean_iterations']:+.1f}", decision_a2])

    # Record gradient balance comparison table
    with open(grad_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["milestone", "V7_BASE_ratio", "V7_BASE_g_spar", "V7_BASE_g_fid", "V7_A2_ratio", "V7_A2_g_spar", "V7_A2_g_fid"])
        for tag in ["init", "sigma_1", "sigma_01", "sigma_001"]:
            b_aud = res_base["audit_summary"].get(tag, {"ratio": 0, "sparsity": 0, "fidelity": 0})
            a_aud = res_a2["audit_summary"].get(tag, {"ratio": 0, "sparsity": 0, "fidelity": 0})
            w.writerow([tag, f"{b_aud['ratio']:.4f}", f"{b_aud['sparsity']:.4f}", f"{b_aud['fidelity']:.4f}", f"{a_aud['ratio']:.4f}", f"{a_aud['sparsity']:.4f}", f"{a_aud['fidelity']:.4f}"])

    print(f"V7_A2: {res_a2['runtime_ms_patch']:.3f} ms/patch, PSNR: {res_a2['psnr']:.2f} dB (dPSNR: {d_psnr_a2:+.2f} dB), Residual: {res_a2['measurement_residual']:.4f} (dResidual: {d_res_a2:+.4f}), DC Error: {res_a2['dc_error']:.4f} vs Base {res_base['dc_error']:.4f}")
    print(f"  Gradient Ratio at sigma=0.01: Base = {res_base['audit_summary'].get('sigma_001', {}).get('ratio', 0.0):.2f} vs A2 = {res_a2['audit_summary'].get('sigma_001', {}).get('ratio', 0.0):.2f}")

    master_records.append({
        "variant": "V7_A2_SCALE_COUPLED_LAMBDA",
        "description": "Scale-coupled lambda(sigma) = lambda_0 * sigma^2",
        "runtime_ms_patch": res_a2["runtime_ms_patch"],
        "speedup_vs_V7": speedup_a2,
        "runtime_reduction_pct": reduct_a2,
        "psnr": res_a2["psnr"],
        "ssim": res_a2["ssim"],
        "mse": res_a2["mse"],
        "measurement_residual": res_a2["measurement_residual"],
        "coefficient_error": res_a2["coefficient_error"],
        "dc_error": res_a2["dc_error"],
        "ac_error": res_a2["ac_error"],
        "iterations": res_a2["mean_iterations"],
        "final_sigma": res_a2["final_sigma"],
        "active_support_ratio": res_a2["active_support_ratio"],
        "mean_active_count": res_a2["mean_active_count"],
        "failed_line_searches": res_a2["failed_line_searches"],
        "gradient_ratio_001": res_a2["audit_summary"].get("sigma_001", {}).get("ratio", 0.0),
        "decision": decision_a2,
    })

    # =========================================================================
    # STEP 4: VARIANT A3 (ACTIVE SUPPORT THRESHOLD SENSITIVITY)
    # =========================================================================
    print("\n--- Phase 4: Evaluating Variant A3 (Active-Support Threshold Study) ---")
    a3_dir = os.path.join(RESULTS_DIR, "V7_A3")
    os.makedirs(a3_dir, exist_ok=True)
    a3_csv = os.path.join(a3_dir, "a3_results.csv")

    threshold_grid = [1e-5, 1e-4, 1e-3, 1e-2]
    best_a3 = None

    with open(a3_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["multiplier", "active_support_ratio", "mean_active_count", "min_active_count", "runtime_ms_patch", "speedup", "psnr", "ssim", "mse", "measurement_residual", "iterations", "decision"])

        for mult in threshold_grid:
            solver_a3 = SolverV7A3SupportThreshold(A_std, lambda_reg=0.1, tol=1e-5, threshold_multiplier=mult)
            res_a3 = evaluate_solver_on_patches(
                lambda y, i: solver_a3.denoise_patch(y, return_diagnostics=True),
                Y_std, clean_thetas, clean_patches, clean_records, A_std, H, W
            )
            speedup = res_base["runtime_ms_patch"] / res_a3["runtime_ms_patch"]
            reduct_pct = (res_base["runtime_ms_patch"] - res_a3["runtime_ms_patch"]) / res_base["runtime_ms_patch"] * 100.0
            d_psnr = res_a3["psnr"] - res_base["psnr"]

            dec = "KEEP" if (speedup >= 1.03 and d_psnr >= -0.05) else ("REJECT" if d_psnr < -0.10 else "NO_SPEEDUP_REJECT")

            w.writerow([f"{mult:.0e}", f"{res_a3['active_support_ratio']:.4f}", f"{res_a3['mean_active_count']:.2f}", res_a3['min_active_count'], f"{res_a3['runtime_ms_patch']:.3f}", f"{speedup:.3f}x", f"{res_a3['psnr']:.4f}", f"{res_a3['ssim']:.4f}", f"{res_a3['mse']:.6f}", f"{res_a3['measurement_residual']:.4f}", f"{res_a3['mean_iterations']:.1f}", dec])
            print(f"  Threshold {mult:.0e}*sigma: Active Ratio: {res_a3['active_support_ratio']:.4f}, Mean Active: {res_a3['mean_active_count']:.1f}, Min: {res_a3['min_active_count']}, Time: {res_a3['runtime_ms_patch']:.3f} ms ({speedup:.2f}x), PSNR: {res_a3['psnr']:.2f} dB -> {dec}")

            if mult == 1e-3:
                best_a3 = res_a3
                master_records.append({
                    "variant": "V7_A3_SUPPORT_THRESHOLD (1e-3)",
                    "description": "Active support threshold multiplier tau = 1e-3 * sigma",
                    "runtime_ms_patch": res_a3["runtime_ms_patch"],
                    "speedup_vs_V7": speedup,
                    "runtime_reduction_pct": reduct_pct,
                    "psnr": res_a3["psnr"],
                    "ssim": res_a3["ssim"],
                    "mse": res_a3["mse"],
                    "measurement_residual": res_a3["measurement_residual"],
                    "coefficient_error": res_a3["coefficient_error"],
                    "dc_error": res_a3["dc_error"],
                    "ac_error": res_a3["ac_error"],
                    "iterations": res_a3["mean_iterations"],
                    "final_sigma": res_a3["final_sigma"],
                    "active_support_ratio": res_a3["active_support_ratio"],
                    "mean_active_count": res_a3["mean_active_count"],
                    "failed_line_searches": res_a3["failed_line_searches"],
                    "gradient_ratio_001": res_a3["audit_summary"].get("sigma_001", {}).get("ratio", 0.0),
                    "decision": dec,
                })

    # =========================================================================
    # STEP 5: VARIANT A4 (CONTINUATION SCHEDULE STUDY)
    # =========================================================================
    print("\n--- Phase 5: Evaluating Variant A4 (Continuation Schedule Study) ---")
    a4_dir = os.path.join(RESULTS_DIR, "V7_A4")
    os.makedirs(a4_dir, exist_ok=True)
    a4_csv = os.path.join(a4_dir, "a4_results.csv")

    decay_grid = [0.90, 0.95, 0.98]
    best_a4 = None

    with open(a4_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sigma_decay", "iterations", "runtime_ms_patch", "speedup", "final_sigma", "psnr", "ssim", "mse", "measurement_residual", "decision"])

        for decay in decay_grid:
            solver_a4 = SolverV7A4Continuation(A_std, lambda_reg=0.1, tol=1e-5, decrease_factor=decay)
            res_a4 = evaluate_solver_on_patches(
                lambda y, i: solver_a4.denoise_patch(y, return_diagnostics=True),
                Y_std, clean_thetas, clean_patches, clean_records, A_std, H, W
            )
            speedup = res_base["runtime_ms_patch"] / res_a4["runtime_ms_patch"]
            reduct_pct = (res_base["runtime_ms_patch"] - res_a4["runtime_ms_patch"]) / res_base["runtime_ms_patch"] * 100.0
            d_psnr = res_a4["psnr"] - res_base["psnr"]

            dec = "PROMISING" if (speedup >= 1.05 and d_psnr >= -0.05) else ("REJECT" if d_psnr < -0.10 else "NEUTRAL")

            w.writerow([decay, f"{res_a4['mean_iterations']:.1f}", f"{res_a4['runtime_ms_patch']:.3f}", f"{speedup:.3f}x", f"{res_a4['final_sigma']:.4f}", f"{res_a4['psnr']:.4f}", f"{res_a4['ssim']:.4f}", f"{res_a4['mse']:.6f}", f"{res_a4['measurement_residual']:.4f}", dec])
            print(f"  Decay {decay}: Iters: {res_a4['mean_iterations']:.1f}, Final Sigma: {res_a4['final_sigma']:.4f}, Time: {res_a4['runtime_ms_patch']:.3f} ms ({speedup:.2f}x), PSNR: {res_a4['psnr']:.2f} dB -> {dec}")

            if decay == 0.90:
                best_a4 = res_a4
                master_records.append({
                    "variant": "V7_A4_CONTINUATION (decay=0.90)",
                    "description": "Faster continuation decay sigma_decay = 0.90",
                    "runtime_ms_patch": res_a4["runtime_ms_patch"],
                    "speedup_vs_V7": speedup,
                    "runtime_reduction_pct": reduct_pct,
                    "psnr": res_a4["psnr"],
                    "ssim": res_a4["ssim"],
                    "mse": res_a4["mse"],
                    "measurement_residual": res_a4["measurement_residual"],
                    "coefficient_error": res_a4["coefficient_error"],
                    "dc_error": res_a4["dc_error"],
                    "ac_error": res_a4["ac_error"],
                    "iterations": res_a4["mean_iterations"],
                    "final_sigma": res_a4["final_sigma"],
                    "active_support_ratio": res_a4["active_support_ratio"],
                    "mean_active_count": res_a4["mean_active_count"],
                    "failed_line_searches": res_a4["failed_line_searches"],
                    "gradient_ratio_001": res_a4["audit_summary"].get("sigma_001", {}).get("ratio", 0.0),
                    "decision": dec,
                })

    # =========================================================================
    # STEP 6: VARIANT A5 (TWO-STAGE SPARSE RECOVERY)
    # =========================================================================
    print("\n--- Phase 6: Evaluating Variant A5 (Two-Stage Recovery) ---")
    a5_dir = os.path.join(RESULTS_DIR, "V7_A5")
    os.makedirs(a5_dir, exist_ok=True)
    a5_csv = os.path.join(a5_dir, "a5_results.csv")

    prune_grid = [1e-4, 1e-3, 1e-2]
    best_a5 = None

    with open(a5_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prune_thresh", "mean_support_size", "runtime_ms_patch", "psnr", "ssim", "mse", "measurement_residual", "coefficient_error", "dc_error", "ac_error", "decision"])

        for thresh in prune_grid:
            solver_a5 = SolverV7A5TwoStage(A_std, lambda_reg=0.1, tol=1e-5, support_prune_threshold=thresh)
            res_a5 = evaluate_solver_on_patches(
                lambda y, i: solver_a5.denoise_patch(y, return_diagnostics=True),
                Y_std, clean_thetas, clean_patches, clean_records, A_std, H, W
            )
            speedup = res_base["runtime_ms_patch"] / res_a5["runtime_ms_patch"]
            reduct_pct = (res_base["runtime_ms_patch"] - res_a5["runtime_ms_patch"]) / res_base["runtime_ms_patch"] * 100.0
            d_psnr = res_a5["psnr"] - res_base["psnr"]
            d_res = res_a5["measurement_residual"] - res_base["measurement_residual"]

            dec = "STRONG_CANDIDATE" if (d_psnr >= 0.5 and d_res <= -0.2) else ("PROMISING" if d_psnr > 0 else "REJECT")

            w.writerow([thresh, f"{res_a5['mean_active_count']:.1f}", f"{res_a5['runtime_ms_patch']:.3f}", f"{res_a5['psnr']:.4f}", f"{res_a5['ssim']:.4f}", f"{res_a5['mse']:.6f}", f"{res_a5['measurement_residual']:.4f}", f"{res_a5['coefficient_error']:.4f}", f"{res_a5['dc_error']:.4f}", f"{res_a5['ac_error']:.4f}", dec])
            print(f"  Prune Thresh {thresh:.0e}: Support: {res_a5['mean_active_count']:.1f}, PSNR: {res_a5['psnr']:.2f} dB (dPSNR: {d_psnr:+.2f} dB), Residual: {res_a5['measurement_residual']:.4f} (dRes: {d_res:+.4f}), DC Error: {res_a5['dc_error']:.4f} vs Base {res_base['dc_error']:.4f}, Time: {res_a5['runtime_ms_patch']:.3f} ms -> {dec}")

            if thresh == 1e-3:
                best_a5 = res_a5
                master_records.append({
                    "variant": "V7_A5_TWO_STAGE (thresh=1e-3)",
                    "description": "Two-stage: ASL-SR-DPT support ID + Least Squares debiasing",
                    "runtime_ms_patch": res_a5["runtime_ms_patch"],
                    "speedup_vs_V7": speedup,
                    "runtime_reduction_pct": reduct_pct,
                    "psnr": res_a5["psnr"],
                    "ssim": res_a5["ssim"],
                    "mse": res_a5["mse"],
                    "measurement_residual": res_a5["measurement_residual"],
                    "coefficient_error": res_a5["coefficient_error"],
                    "dc_error": res_a5["dc_error"],
                    "ac_error": res_a5["ac_error"],
                    "iterations": res_a5["mean_iterations"],
                    "final_sigma": res_a5["final_sigma"],
                    "active_support_ratio": res_a5["active_support_ratio"],
                    "mean_active_count": res_a5["mean_active_count"],
                    "failed_line_searches": res_a5["failed_line_searches"],
                    "gradient_ratio_001": res_a5["audit_summary"].get("sigma_001", {}).get("ratio", 0.0),
                    "decision": dec,
                })

    # =========================================================================
    # STEP 7: VARIANT A6 (DC-PRESERVING ARCHITECTURE)
    # =========================================================================
    print("\n--- Phase 7: Evaluating Variant A6 (DC-Preserving Architecture) ---")
    a6_dir = os.path.join(RESULTS_DIR, "V7_A6")
    os.makedirs(a6_dir, exist_ok=True)
    a6_csv = os.path.join(a6_dir, "a6_results.csv")

    solver_a6 = SolverV7A6DCPreservation(A_dc, lambda_reg=0.1, tol=1e-5)
    res_a6 = evaluate_solver_on_patches(
        lambda y, i: solver_a6.denoise_patch_dc(y, Y_dc[i], return_diagnostics=True),
        Y_ac, clean_thetas, clean_patches, clean_records, A_dc, H, W
    )
    speedup_a6 = res_base["runtime_ms_patch"] / res_a6["runtime_ms_patch"]
    reduct_a6 = (res_base["runtime_ms_patch"] - res_a6["runtime_ms_patch"]) / res_base["runtime_ms_patch"] * 100.0
    d_psnr_a6 = res_a6["psnr"] - res_base["psnr"]
    d_dc_err = res_a6["dc_error"] - res_base["dc_error"]

    dec_a6 = "PROMISING_ARCH" if d_psnr_a6 >= 0.5 else "NEUTRAL"

    with open(a6_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "Standard_Sensing", "DC_Preserving_Sensing", "difference", "decision"])
        w.writerow(["runtime_ms_patch", f"{res_base['runtime_ms_patch']:.3f}", f"{res_a6['runtime_ms_patch']:.3f}", f"{res_a6['runtime_ms_patch'] - res_base['runtime_ms_patch']:.3f}", "—"])
        w.writerow(["speedup", "1.000x", f"{speedup_a6:.3f}x", f"+{speedup_a6 - 1.0:.3f}x", "—"])
        w.writerow(["PSNR", f"{res_base['psnr']:.4f}", f"{res_a6['psnr']:.4f}", f"{d_psnr_a6:+.4f}", "—"])
        w.writerow(["SSIM", f"{res_base['ssim']:.4f}", f"{res_a6['ssim']:.4f}", f"{res_a6['ssim'] - res_base['ssim']:+.4f}", "—"])
        w.writerow(["MSE", f"{res_base['mse']:.6f}", f"{res_a6['mse']:.6f}", f"{res_a6['mse'] - res_base['mse']:+.6f}", "—"])
        w.writerow(["measurement_residual", f"{res_base['measurement_residual']:.4f}", f"{res_a6['measurement_residual']:.4f}", f"{res_a6['measurement_residual'] - res_base['measurement_residual']:+.4f}", "—"])
        w.writerow(["coefficient_error", f"{res_base['coefficient_error']:.4f}", f"{res_a6['coefficient_error']:.4f}", f"{res_a6['coefficient_error'] - res_base['coefficient_error']:+.4f}", "—"])
        w.writerow(["dc_error", f"{res_base['dc_error']:.4f}", f"{res_a6['dc_error']:.4f}", f"{d_dc_err:+.4f}", "—"])
        w.writerow(["ac_error", f"{res_base['ac_error']:.4f}", f"{res_a6['ac_error']:.4f}", f"{res_a6['ac_error'] - res_base['ac_error']:+.4f}", "—"])
        w.writerow(["iterations", f"{res_base['mean_iterations']:.1f}", f"{res_a6['mean_iterations']:.1f}", f"{res_a6['mean_iterations'] - res_base['mean_iterations']:+.1f}", dec_a6])

    print(f"V7_A6 (DC-Preserving): {res_a6['runtime_ms_patch']:.3f} ms/patch, PSNR: {res_a6['psnr']:.2f} dB (dPSNR: {d_psnr_a6:+.2f} dB), DC Error: {res_a6['dc_error']:.4f} vs Base {res_base['dc_error']:.4f} (dDC: {d_dc_err:+.4f}) -> {dec_a6}")

    master_records.append({
        "variant": "V7_A6_DC_PRESERVATION",
        "description": "DC-preserving architecture (1 DC exact, 37 AC measurements of 63 AC)",
        "runtime_ms_patch": res_a6["runtime_ms_patch"],
        "speedup_vs_V7": speedup_a6,
        "runtime_reduction_pct": reduct_a6,
        "psnr": res_a6["psnr"],
        "ssim": res_a6["ssim"],
        "mse": res_a6["mse"],
        "measurement_residual": res_a6["measurement_residual"],
        "coefficient_error": res_a6["coefficient_error"],
        "dc_error": res_a6["dc_error"],
        "ac_error": res_a6["ac_error"],
        "iterations": res_a6["mean_iterations"],
        "final_sigma": res_a6["final_sigma"],
        "active_support_ratio": res_a6["active_support_ratio"],
        "mean_active_count": res_a6["mean_active_count"],
        "failed_line_searches": res_a6["failed_line_searches"],
        "gradient_ratio_001": res_a6["audit_summary"].get("sigma_001", {}).get("ratio", 0.0),
        "decision": dec_a6,
    })

    # =========================================================================
    # STEP 8: MASTER COMPARISON TABLE
    # =========================================================================
    print("\n--- Phase 8: Generating Master Comparison Table ---")
    master_csv = os.path.join(RESULTS_DIR, "variant_comparison.csv")
    cols = [
        "variant",
        "runtime_ms_patch",
        "speedup_vs_V7",
        "runtime_reduction_pct",
        "psnr",
        "ssim",
        "mse",
        "measurement_residual",
        "coefficient_error",
        "dc_error",
        "ac_error",
        "iterations",
        "final_sigma",
        "active_support_ratio",
        "mean_active_count",
        "failed_line_searches",
        "gradient_ratio",
        "decision",
    ]
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in master_records:
            w.writerow([
                r["variant"],
                f"{r['runtime_ms_patch']:.3f}",
                f"{r['speedup_vs_V7']:.3f}x",
                f"{r['runtime_reduction_pct']:.2f}%",
                f"{r['psnr']:.4f}",
                f"{r['ssim']:.4f}",
                f"{r['mse']:.6f}",
                f"{r['measurement_residual']:.4f}",
                f"{r['coefficient_error']:.4f}",
                f"{r['dc_error']:.4f}",
                f"{r['ac_error']:.4f}",
                f"{r['iterations']:.1f}",
                f"{r['final_sigma']:.4f}",
                f"{r['active_support_ratio']:.4f}",
                f"{r['mean_active_count']:.1f}",
                r["failed_line_searches"],
                f"{r['gradient_ratio_001']:.2f}",
                r["decision"],
            ])
    print(f"Saved: {master_csv}")

    # =========================================================================
    # STEP 9: STATISTICAL TIMING BENCHMARK (10 REPEATED TRIALS)
    # =========================================================================
    print("\n--- Phase 9: Statistical Timing Benchmark (10 Repetitions on 500 Patches) ---")
    promising_solvers = [
        ("V7_OPT_BASE", lambda y: base_solver.denoise_patch(y, return_diagnostics=False), Y_std),
        ("V7_A1_ADAPTIVE_EXIT", lambda y: SolverV7A1AdaptiveExit(A_std, 0.1, 1e-5, patience=2).denoise_patch(y, return_diagnostics=False), Y_std),
        ("V7_A4_CONTINUATION", lambda y: SolverV7A4Continuation(A_std, 0.1, 1e-5, decrease_factor=0.90).denoise_patch(y, return_diagnostics=False), Y_std),
        ("V7_A5_TWO_STAGE", lambda y: SolverV7A5TwoStage(A_std, 0.1, 1e-5, support_prune_threshold=1e-3).denoise_patch(y, return_diagnostics=False), Y_std),
        ("V7_A6_DC_PRESERVATION", lambda y_idx: solver_a6.denoise_patch_dc(Y_ac[y_idx], Y_dc[y_idx], return_diagnostics=False), None),
    ]

    timing_csv = os.path.join(RESULTS_DIR, "timing_benchmarks.csv")
    with open(timing_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "mean_ms_patch", "median_ms_patch", "std_ms_patch", "min_ms_patch", "max_ms_patch", "speedup_vs_V7"])

        for name, s_call, y_src in promising_solvers:
            times = []
            for trial in range(10):
                t_start = time.perf_counter()
                if y_src is not None:
                    for i in range(500):
                        s_call(y_src[i])
                else:
                    for i in range(500):
                        s_call(i)
                t_elapsed = time.perf_counter() - t_start
                times.append((t_elapsed / 500.0) * 1000.0)

            t_mean = float(np.mean(times))
            t_med = float(np.median(times))
            t_std = float(np.std(times))
            t_min = float(np.min(times))
            t_max = float(np.max(times))
            spd = res_base["runtime_ms_patch"] / t_mean

            w.writerow([name, f"{t_mean:.3f}", f"{t_med:.3f}", f"{t_std:.3f}", f"{t_min:.3f}", f"{t_max:.3f}", f"{spd:.3f}x"])
            print(f"  {name:25s}: Mean={t_mean:.3f} ms/patch (std={t_std:.3f}), Median={t_med:.3f}, Min={t_min:.3f}, Max={t_max:.3f} -> Speedup: {spd:.2f}x")

    print(f"Saved: {timing_csv}")

    # Return top promising candidates for full-image pilot
    return master_records


if __name__ == "__main__":
    run_algorithmic_study()
