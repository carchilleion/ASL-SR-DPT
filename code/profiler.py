"""
ASL-SR-DPT Profiling Suite
Investigates runtime bottlenecks, active-support behavior, line-search dynamics,
objective decomposition, and quality loss across controlled experiments.
Outputs all profiling data directly to results/profiling/.
"""

import os
import sys
import time
import json
import csv
import cProfile
import pstats
import io
import tracemalloc
import numpy as np
from PIL import Image

# Ensure code/ is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_sparse_solver_v7_fixed import (
    HybridSparseSolverV7,
    run_omp,
    precompute_omp,
    precompute_lasso_admm,
    run_lasso_admm,
)
from sensing import (
    generate_random_sensing,
    generate_dc_sensing,
    generate_standard_measurement,
    generate_dc_preserving_measurement,
    restore_dc_component,
    add_awgn,
)
from reconstruction import extract_patches, dct_patch, idct_patch, reconstruct_image
from metrics import compute_all_metrics, compute_psnr


def run_profiling_suite(
    image_path="data/BSD68/test001.png",
    num_patches=500,
    noise_sigma=15.0,
    output_dir="results/profiling",
    seed=20260908,
):
    """
    Executes Parts 4 through 20 of the root-cause profiling study.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("ASL-SR-DPT ROOT-CAUSE & PERFORMANCE PROFILING STUDY")
    print(f"Target Image: {image_path}")
    print(f"Controlled Patches: {num_patches}")
    print(f"Noise Sigma: {noise_sigma}")
    print(f"Output Directory: {output_dir}")
    print("=" * 70)

    # 1. Load clean image and create noisy image
    clean_img = np.array(Image.open(image_path), dtype=float) / 255.0
    H, W = clean_img.shape
    noise_seed = seed + 1000
    sensing_seed = seed + 1

    noisy_img = add_awgn(clean_img, sigma_noise=noise_sigma, seed=noise_seed)

    # Extract all patches and slice to requested count
    all_patch_records = extract_patches(noisy_img, patch_size=8, stride=2)
    clean_patch_records = extract_patches(clean_img, patch_size=8, stride=2)
    total_image_patches = len(all_patch_records)
    patch_records = all_patch_records[:num_patches]
    clean_patches = clean_patch_records[:num_patches]

    print(f"Extracted {len(patch_records)} patches for profiling (out of {total_image_patches} total).")

    # -------------------------------------------------------------
    # PART 4 & 5 & 6 & 7 & 8 & 11 & 12: Detailed ASL-SR-DPT Run
    # -------------------------------------------------------------
    print("\n[PART 4-8] Running Instrumented ASL-SR-DPT on 500 Patches...")

    # Timer tracking for pipeline components
    t_pinv_start = time.perf_counter()
    A_std = generate_random_sensing(M=38, N=64, seed=sensing_seed)
    solver_asl = HybridSparseSolverV7(A_std, lambda_reg=0.1, tol=1e-5, seed=seed)
    t_pinv = time.perf_counter() - t_pinv_start

    # DCT time
    t_dct_start = time.perf_counter()
    dct_noisy = [dct_patch(p["patch"]) for p in patch_records]
    dct_clean = [dct_patch(p["patch"]) for p in clean_patches]
    t_dct = time.perf_counter() - t_dct_start

    # Measurement generation time
    t_meas_start = time.perf_counter()
    measurements = [generate_standard_measurement(theta, A_std) for theta in dct_noisy]
    t_meas = time.perf_counter() - t_meas_start

    # Solver execution with fine-grained internal instrumentation
    asl_patch_diagnostics = []
    asl_recovered_thetas = []
    t_solver_start = time.perf_counter()

    for i in range(num_patches):
        y_val = measurements[i]
        z_rec, diag = solver_asl.denoise_patch(
            y_val,
            sigma_min=0.01,
            decrease_factor=0.95,
            max_iter=150,
            initial_mu=0.2,
            armijo_c=1e-4,
            beta_decay=0.5,
            max_backtracks=10,
            support_threshold_multiplier=1e-5,
            support_reopen_interval=3,
            use_midpoint=True,
            return_diagnostics=True,
            detailed_profile=True,
            init_method="pinv",
        )
        asl_recovered_thetas.append(z_rec)
        asl_patch_diagnostics.append(diag)

    t_solver = time.perf_counter() - t_solver_start

    # IDCT time
    t_idct_start = time.perf_counter()
    spatial_patches = [{"patch": idct_patch(z.reshape((8, 8))), "x": p["x"], "y": p["y"]} 
                       for z, p in zip(asl_recovered_thetas, patch_records)]
    t_idct = time.perf_counter() - t_idct_start

    # Hamming reconstruction time (reconstructing the sub-image canvas)
    t_hamming_start = time.perf_counter()
    sub_recon = reconstruct_image(spatial_patches, image_shape=(H, W), patch_size=8)
    t_hamming = time.perf_counter() - t_hamming_start

    # Metric calculation time
    t_metrics_start = time.perf_counter()
    metrics_asl = compute_all_metrics(clean_img, sub_recon)
    t_metrics = time.perf_counter() - t_metrics_start

    # Internal solver time breakdown aggregation
    t_obj = sum(d["time_breakdown"]["objective_time"] for d in asl_patch_diagnostics)
    t_grad = sum(d["time_breakdown"]["gradient_time"] for d in asl_patch_diagnostics)
    t_cand = sum(d["time_breakdown"]["candidate_time"] for d in asl_patch_diagnostics)
    t_mid = sum(d["time_breakdown"]["midpoint_time"] for d in asl_patch_diagnostics)
    t_mask = sum(d["time_breakdown"]["support_mask_time"] for d in asl_patch_diagnostics)
    t_res = sum(d["time_breakdown"]["residual_time"] for d in asl_patch_diagnostics)
    t_other_solver = max(t_solver - (t_obj + t_grad + t_cand + t_mid + t_mask + t_res), 0.0)

    # I/O & CSV writing time
    t_io_start = time.perf_counter()

    # -------------------------------------------------------------
    # Output 1: results/profiling/time_breakdown.csv
    # -------------------------------------------------------------
    t_total_pipeline = t_pinv + t_dct + t_meas + t_solver + t_idct + t_hamming + t_metrics

    breakdown_rows = [
        ("pseudoinverse_setup", t_pinv, t_pinv / num_patches, 0.0, (t_pinv / t_total_pipeline) * 100),
        ("forward_dct", t_dct, t_dct / num_patches, 0.0, (t_dct / t_total_pipeline) * 100),
        ("measurement_generation", t_meas, t_meas / num_patches, 0.0, (t_meas / t_total_pipeline) * 100),
        ("asl_solver_total", t_solver, t_solver / num_patches, 100.0, (t_solver / t_total_pipeline) * 100),
        ("  solver_objective_evaluation", t_obj, t_obj / num_patches, (t_obj / t_solver) * 100, (t_obj / t_total_pipeline) * 100),
        ("  solver_gradient_evaluation", t_grad, t_grad / num_patches, (t_grad / t_solver) * 100, (t_grad / t_total_pipeline) * 100),
        ("  solver_armijo_candidate_checks", t_cand, t_cand / num_patches, (t_cand / t_solver) * 100, (t_cand / t_total_pipeline) * 100),
        ("  solver_midpoint_evaluation", t_mid, t_mid / num_patches, (t_mid / t_solver) * 100, (t_mid / t_total_pipeline) * 100),
        ("  solver_support_mask_calculation", t_mask, t_mask / num_patches, (t_mask / t_solver) * 100, (t_mask / t_total_pipeline) * 100),
        ("  solver_final_residual_calc", t_res, t_res / num_patches, (t_res / t_solver) * 100, (t_res / t_total_pipeline) * 100),
        ("  solver_internal_overhead", t_other_solver, t_other_solver / num_patches, (t_other_solver / t_solver) * 100, (t_other_solver / t_total_pipeline) * 100),
        ("inverse_idct", t_idct, t_idct / num_patches, 0.0, (t_idct / t_total_pipeline) * 100),
        ("hamming_aggregation", t_hamming, t_hamming / num_patches, 0.0, (t_hamming / t_total_pipeline) * 100),
        ("metric_calculation", t_metrics, t_metrics / num_patches, 0.0, (t_metrics / t_total_pipeline) * 100),
    ]

    with open(os.path.join(output_dir, "time_breakdown.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "total_time", "mean_time_per_patch", "percentage_of_solver_time", "percentage_of_total_time", "patch_count"])
        for comp, tot, mean_p, pct_s, pct_t in breakdown_rows:
            writer.writerow([comp, f"{tot:.6f}", f"{mean_p:.8f}", f"{pct_s:.2f}", f"{pct_t:.2f}", num_patches])

    # -------------------------------------------------------------
    # Output 2: results/profiling/asl_operation_counts.csv (Part 5)
    # -------------------------------------------------------------
    with open(os.path.join(output_dir, "asl_operation_counts.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patch_id", "iterations", "objective_evaluations", "gradient_evaluations",
            "candidate_evaluations", "midpoint_evaluations", "backtracking_attempts",
            "accepted_steps", "failed_line_searches", "support_mask_calculations"
        ])
        for p_idx, d in enumerate(asl_patch_diagnostics):
            op = d["operation_counts"]
            writer.writerow([
                p_idx, d["iterations"], op["objective_evaluations"], op["gradient_evaluations"],
                op["candidate_evaluations"], op["midpoint_evaluations"], op["backtracking_attempts"],
                op["accepted_steps"], op["failed_line_searches"], op["support_mask_calculations"]
            ])

    # -------------------------------------------------------------
    # Output 3: results/profiling/active_support.csv (Part 6)
    # -------------------------------------------------------------
    with open(os.path.join(output_dir, "active_support.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patch_id", "iteration", "active_count", "active_ratio", "sigma", "mu",
            "relative_change", "objective", "residual", "accepted_or_failed"
        ])
        for p_idx, d in enumerate(asl_patch_diagnostics):
            for row in d["iteration_history"]:
                writer.writerow([
                    p_idx, row["iteration"], row["active_count"], f"{row['active_ratio']:.4f}",
                    f"{row['sigma']:.6f}", f"{row.get('mu', row.get('accepted_mu', 0.0)):.6f}", f"{row['relative_change']:.6e}",
                    f"{row['objective']:.6f}", f"{row['residual']:.6f}", row["accepted_or_failed"]
                ])

    # -------------------------------------------------------------
    # Output 4: results/profiling/sigma_behavior.csv (Part 7)
    # -------------------------------------------------------------
    with open(os.path.join(output_dir, "sigma_behavior.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patch_id", "initial_sigma", "final_sigma", "number_of_sigma_updates",
            "number_of_iterations_at_sigma_min", "max_iter_hit", "converged_at_sigma_min"
        ])
        for p_idx, d in enumerate(asl_patch_diagnostics):
            sig = d["sigma_history"]
            writer.writerow([
                p_idx, f"{sig['initial_sigma']:.6f}", f"{sig['final_sigma']:.6f}",
                sig["number_of_sigma_updates"], sig["number_of_iterations_at_sigma_min"],
                sig["max_iter_hit"], sig["converged_at_sigma_min"]
            ])

    # -------------------------------------------------------------
    # Output 5: results/profiling/line_search.csv (Part 8)
    # -------------------------------------------------------------
    with open(os.path.join(output_dir, "line_search.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patch_id", "iteration", "initial_mu", "accepted_mu", "number_of_backtracks",
            "accepted_step", "failed_step", "midpoint_accepted"
        ])
        for p_idx, d in enumerate(asl_patch_diagnostics):
            for row in d["iteration_history"]:
                is_acc = 1 if row["accepted_or_failed"] in ["accepted", "accepted_midpoint"] else 0
                is_fail = 1 if row["accepted_or_failed"] == "failed" else 0
                writer.writerow([
                    p_idx, row["iteration"], f"{row['initial_mu']:.6f}", f"{row['accepted_mu']:.6f}",
                    row["backtracks"], is_acc, is_fail, 1 if row["midpoint_accepted"] else 0
                ])

    # -------------------------------------------------------------
    # Output 6: results/profiling/objective_components.csv (Part 12)
    # -------------------------------------------------------------
    with open(os.path.join(output_dir, "objective_components.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patch_id", "iteration", "sigma", "fidelity_term", "sparsity_term",
            "total_objective", "ratio_sparsity_to_fidelity"
        ])
        for p_idx, d in enumerate(asl_patch_diagnostics):
            for row in d["iteration_history"]:
                fid = row["fidelity"]
                sparse = abs(row["sparsity"])
                ratio = (sparse / fid) if fid > 1e-12 else 0.0
                writer.writerow([
                    p_idx, row["iteration"], f"{row['sigma']:.6f}", f"{fid:.6f}",
                    f"{row['sparsity']:.6f}", f"{row['objective']:.6f}", f"{ratio:.4f}"
                ])

    t_io = time.perf_counter() - t_io_start

    print(f"  ASL-SR-DPT Solver time: {t_solver:.3f} s ({t_solver / num_patches * 1000:.2f} ms/patch)")
    print(f"  Solver represents {(t_solver / t_total_pipeline) * 100:.2f}% of total image pipeline time.")

    # -------------------------------------------------------------
    # PART 9: Midpoint Analysis (ON vs OFF on exact same 500 patches)
    # -------------------------------------------------------------
    print("\n[PART 9] Running Midpoint ON vs OFF Comparison...")
    # Run Midpoint OFF
    asl_off_thetas = []
    asl_off_diagnostics = []
    t_off_start = time.perf_counter()
    for i in range(num_patches):
        z_rec, diag = solver_asl.denoise_patch(
            measurements[i],
            use_midpoint=False,
            return_diagnostics=True,
            detailed_profile=True,
            init_method="pinv",
        )
        asl_off_thetas.append(z_rec)
        asl_off_diagnostics.append(diag)
    t_off = time.perf_counter() - t_off_start

    # Reconstruct OFF image for metric comparison
    sp_off = [{"patch": idct_patch(z.reshape((8, 8))), "x": p["x"], "y": p["y"]} 
              for z, p in zip(asl_off_thetas, patch_records)]
    sub_recon_off = reconstruct_image(sp_off, image_shape=(H, W), patch_size=8)
    metrics_off = compute_all_metrics(clean_img, sub_recon_off)

    # Compute comparative midpoint metrics
    tot_obj_on = sum(d["operation_counts"]["objective_evaluations"] for d in asl_patch_diagnostics)
    tot_mid_on = sum(d["operation_counts"]["midpoint_evaluations"] for d in asl_patch_diagnostics)
    tot_bt_on = sum(d["operation_counts"]["backtracking_attempts"] for d in asl_patch_diagnostics)
    tot_acc_on = sum(d["operation_counts"]["accepted_steps"] for d in asl_patch_diagnostics)
    tot_fail_on = sum(d["operation_counts"]["failed_line_searches"] for d in asl_patch_diagnostics)
    tot_iters_on = sum(d["iterations"] for d in asl_patch_diagnostics)
    mean_res_on = float(np.mean([d["final_residual"] for d in asl_patch_diagnostics]))

    tot_obj_off = sum(d["operation_counts"]["objective_evaluations"] for d in asl_off_diagnostics)
    tot_mid_off = sum(d["operation_counts"]["midpoint_evaluations"] for d in asl_off_diagnostics)
    tot_bt_off = sum(d["operation_counts"]["backtracking_attempts"] for d in asl_off_diagnostics)
    tot_acc_off = sum(d["operation_counts"]["accepted_steps"] for d in asl_off_diagnostics)
    tot_fail_off = sum(d["operation_counts"]["failed_line_searches"] for d in asl_off_diagnostics)
    tot_iters_off = sum(d["iterations"] for d in asl_off_diagnostics)
    mean_res_off = float(np.mean([d["final_residual"] for d in asl_off_diagnostics]))

    mid_accepted_count = sum(
        sum(1 for row in d["iteration_history"] if row.get("midpoint_accepted", False))
        for d in asl_patch_diagnostics
    )

    overhead_pct = ((t_solver - t_off) / t_off) * 100.0 if t_off > 0 else 0.0
    eval_fraction = (tot_mid_on / tot_obj_on) * 100.0 if tot_obj_on > 0 else 0.0
    acceptance_rate = (mid_accepted_count / tot_acc_on) * 100.0 if tot_acc_on > 0 else 0.0

    with open(os.path.join(output_dir, "midpoint_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "configuration", "runtime_seconds", "objective_evaluations", "midpoint_evaluations",
            "backtracking_attempts", "accepted_steps", "failed_line_searches", "iterations",
            "mean_residual", "psnr", "ssim", "mse", "midpoint_overhead_pct",
            "midpoint_eval_fraction", "midpoint_acceptance_rate"
        ])
        writer.writerow([
            "Midpoint_ON", f"{t_solver:.4f}", tot_obj_on, tot_mid_on, tot_bt_on, tot_acc_on,
            tot_fail_on, tot_iters_on, f"{mean_res_on:.6f}", f"{metrics_asl['psnr']:.4f}",
            f"{metrics_asl['ssim']:.4f}", f"{metrics_asl['mse']:.6f}", f"{overhead_pct:.2f}",
            f"{eval_fraction:.2f}", f"{acceptance_rate:.2f}"
        ])
        writer.writerow([
            "Midpoint_OFF", f"{t_off:.4f}", tot_obj_off, tot_mid_off, tot_bt_off, tot_acc_off,
            tot_fail_off, tot_iters_off, f"{mean_res_off:.6f}", f"{metrics_off['psnr']:.4f}",
            f"{metrics_off['ssim']:.4f}", f"{metrics_off['mse']:.6f}", "0.00", "0.00", "0.00"
        ])

    print(f"  Midpoint ON:  Time={t_solver:.3f} s | ObjEvals={tot_obj_on} | MidEvals={tot_mid_on} | PSNR={metrics_asl['psnr']:.2f} dB")
    print(f"  Midpoint OFF: Time={t_off:.3f} s | ObjEvals={tot_obj_off} | MidEvals=0 | PSNR={metrics_off['psnr']:.2f} dB")
    print(f"  Midpoint overhead: {overhead_pct:+.2f}% | Acceptance rate: {acceptance_rate:.2f}%")

    # -------------------------------------------------------------
    # PART 10 & 11: Three-Solver Patch Comparison & Quality Loss
    # -------------------------------------------------------------
    print("\n[PART 10-11] Running 3-Solver Patch Comparison & Quality Decomposition...")

    # OMP Run
    omp_col_norms = precompute_omp(A_std)
    omp_thetas = []
    omp_times = []
    omp_iters = []
    omp_res = []

    for y_val in measurements:
        t0 = time.perf_counter()
        th, diag = run_omp(y_val, A_std, relative_residual_tol=1e-5, max_coefficients=38, col_norms=omp_col_norms, return_diagnostics=True)
        t1 = time.perf_counter()
        omp_thetas.append(th)
        omp_times.append(t1 - t0)
        omp_iters.append(diag["iterations"])
        omp_res.append(diag["final_residual"])

    t_omp_total = sum(omp_times)

    # LASSO-ADMM Run
    lasso_L = precompute_lasso_admm(A_std, rho=1.0)
    lasso_thetas = []
    lasso_times = []
    lasso_iters = []
    lasso_res = []

    for y_val in measurements:
        t0 = time.perf_counter()
        th, diag = run_lasso_admm(y_val, A_std, lambda_lasso=0.01, rho=1.0, tol=1e-4, max_iter=100, L=lasso_L, return_diagnostics=True)
        t1 = time.perf_counter()
        lasso_thetas.append(th)
        lasso_times.append(t1 - t0)
        lasso_iters.append(diag["iterations"])
        lasso_res.append(diag["final_primal_residual"])

    t_lasso_total = sum(lasso_times)

    # Per-patch solver comparison CSV with diagnostic quality decomposition
    with open(os.path.join(output_dir, "solver_patch_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patch_id", "solver", "runtime", "iterations", "residual", "recovered_coefficient_norm",
            "support_size", "patch_psnr", "measurement_error", "coefficient_error", "DC_error", "AC_error"
        ])
        for p_idx in range(num_patches):
            clean_th_2d = dct_clean[p_idx]
            clean_th = clean_th_2d.reshape(-1)
            y_val = measurements[p_idx]

            # ASL
            th_asl = asl_recovered_thetas[p_idx]
            meas_err_asl = float(np.linalg.norm(A_std @ th_asl - y_val))
            coeff_err_asl = float(np.linalg.norm(th_asl - clean_th))
            dc_err_asl = abs(float(th_asl[0] - clean_th[0]))
            ac_err_asl = float(np.linalg.norm(th_asl[1:] - clean_th[1:]))
            sp_asl = int(np.count_nonzero(np.abs(th_asl) > 1e-5))
            p_psnr_asl = compute_psnr(idct_patch(clean_th_2d), idct_patch(th_asl.reshape((8, 8))))
            t_asl_patch = sum(asl_patch_diagnostics[p_idx]["time_breakdown"].values())
            writer.writerow([
                p_idx, "ASL-SR-DPT", f"{t_asl_patch:.6f}",
                asl_patch_diagnostics[p_idx]["iterations"], f"{asl_patch_diagnostics[p_idx]['final_residual']:.6f}",
                f"{np.linalg.norm(th_asl):.6f}", sp_asl, f"{p_psnr_asl:.2f}", f"{meas_err_asl:.6f}",
                f"{coeff_err_asl:.6f}", f"{dc_err_asl:.6f}", f"{ac_err_asl:.6f}"
            ])

            # OMP
            th_omp = omp_thetas[p_idx]
            meas_err_omp = float(np.linalg.norm(A_std @ th_omp - y_val))
            coeff_err_omp = float(np.linalg.norm(th_omp - clean_th))
            dc_err_omp = abs(float(th_omp[0] - clean_th[0]))
            ac_err_omp = float(np.linalg.norm(th_omp[1:] - clean_th[1:]))
            sp_omp = int(np.count_nonzero(np.abs(th_omp) > 1e-5))
            p_psnr_omp = compute_psnr(idct_patch(clean_th_2d), idct_patch(th_omp.reshape((8, 8))))
            writer.writerow([
                p_idx, "OMP", f"{omp_times[p_idx]:.6f}", omp_iters[p_idx], f"{omp_res[p_idx]:.6f}",
                f"{np.linalg.norm(th_omp):.6f}", sp_omp, f"{p_psnr_omp:.2f}", f"{meas_err_omp:.6f}",
                f"{coeff_err_omp:.6f}", f"{dc_err_omp:.6f}", f"{ac_err_omp:.6f}"
            ])

            # LASSO-ADMM
            th_lasso = lasso_thetas[p_idx]
            meas_err_lasso = float(np.linalg.norm(A_std @ th_lasso - y_val))
            coeff_err_lasso = float(np.linalg.norm(th_lasso - clean_th))
            dc_err_lasso = abs(float(th_lasso[0] - clean_th[0]))
            ac_err_lasso = float(np.linalg.norm(th_lasso[1:] - clean_th[1:]))
            sp_lasso = int(np.count_nonzero(np.abs(th_lasso) > 1e-5))
            p_psnr_lasso = compute_psnr(idct_patch(clean_th_2d), idct_patch(th_lasso.reshape((8, 8))))
            writer.writerow([
                p_idx, "LASSO-ADMM", f"{lasso_times[p_idx]:.6f}", lasso_iters[p_idx], f"{lasso_res[p_idx]:.6f}",
                f"{np.linalg.norm(th_lasso):.6f}", sp_lasso, f"{p_psnr_lasso:.2f}", f"{meas_err_lasso:.6f}",
                f"{coeff_err_lasso:.6f}", f"{dc_err_lasso:.6f}", f"{ac_err_lasso:.6f}"
            ])

    print(f"  ASL-SR-DPT Total Solve: {t_solver:.3f} s (Mean: {t_solver / num_patches * 1000:.2f} ms/patch)")
    print(f"  OMP Total Solve:        {t_omp_total:.3f} s (Mean: {t_omp_total / num_patches * 1000:.2f} ms/patch)")
    print(f"  LASSO-ADMM Total Solve: {t_lasso_total:.3f} s (Mean: {t_lasso_total / num_patches * 1000:.2f} ms/patch)")

    # -------------------------------------------------------------
    # PART 13: Initialization Comparison (Pinv vs Zeros)
    # -------------------------------------------------------------
    print("\n[PART 13] Running Initialization Comparison (Pinv vs Zeros)...")
    zeros_thetas = []
    zeros_diagnostics = []
    t_zeros_start = time.perf_counter()
    for i in range(num_patches):
        z_rec, diag = solver_asl.denoise_patch(
            measurements[i],
            use_midpoint=True,
            return_diagnostics=True,
            detailed_profile=True,
            init_method="zeros",
        )
        zeros_thetas.append(z_rec)
        zeros_diagnostics.append(diag)
    t_zeros = time.perf_counter() - t_zeros_start

    sp_zeros = [{"patch": idct_patch(z.reshape((8, 8))), "x": p["x"], "y": p["y"]} 
                for z, p in zip(zeros_thetas, patch_records)]
    sub_recon_zeros = reconstruct_image(sp_zeros, image_shape=(H, W), patch_size=8)
    metrics_zeros = compute_all_metrics(clean_img, sub_recon_zeros)

    tot_iters_zeros = sum(d["iterations"] for d in zeros_diagnostics)
    mean_res_zeros = float(np.mean([d["final_residual"] for d in zeros_diagnostics]))
    mean_obj_zeros = float(np.mean([d["objective"][-1] for d in zeros_diagnostics]))
    mean_obj_pinv = float(np.mean([d["objective"][-1] for d in asl_patch_diagnostics]))

    with open(os.path.join(output_dir, "initialization_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["init_method", "iterations", "runtime_seconds", "final_objective", "residual", "psnr", "ssim", "mse"])
        writer.writerow(["pinv_warm_start", tot_iters_on, f"{t_solver:.4f}", f"{mean_obj_pinv:.6f}", f"{mean_res_on:.6f}", f"{metrics_asl['psnr']:.4f}", f"{metrics_asl['ssim']:.4f}", f"{metrics_asl['mse']:.6f}"])
        writer.writerow(["zeros_cold_start", tot_iters_zeros, f"{t_zeros:.4f}", f"{mean_obj_zeros:.6f}", f"{mean_res_zeros:.6f}", f"{metrics_zeros['psnr']:.4f}", f"{metrics_zeros['ssim']:.4f}", f"{metrics_zeros['mse']:.6f}"])

    print(f"  Pinv Init:  Time={t_solver:.3f} s | MeanIters={tot_iters_on / num_patches:.1f} | PSNR={metrics_asl['psnr']:.2f} dB")
    print(f"  Zeros Init: Time={t_zeros:.3f} s | MeanIters={tot_iters_zeros / num_patches:.1f} | PSNR={metrics_zeros['psnr']:.2f} dB")

    # -------------------------------------------------------------
    # PART 16: Python Hotspots Profiling (cProfile)
    # -------------------------------------------------------------
    print("\n[PART 16] Profiling Python-Level Hotspots with cProfile on 500 Patches...")
    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(min(num_patches, 500)):
        solver_asl.denoise_patch(measurements[i], use_midpoint=True)

    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(25)
    ps.sort_stats("time")
    ps.print_stats(25)

    with open(os.path.join(output_dir, "cprofile.txt"), "w", encoding="utf-8") as f:
        f.write(s.getvalue())

    print("  Saved cProfile top functions to results/profiling/cprofile.txt")

    # -------------------------------------------------------------
    # PART 17: Memory Profiling
    # -------------------------------------------------------------
    print("\n[PART 17] Profiling Memory Usage on 500 Patches...")
    tracemalloc.start()
    mem_start, _ = tracemalloc.get_traced_memory()

    for i in range(min(num_patches, 500)):
        solver_asl.denoise_patch(measurements[i], use_midpoint=True)

    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    with open(os.path.join(output_dir, "memory_profile.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["process_start_memory_mb", "peak_memory_mb", "process_end_memory_mb", "delta_memory_mb"])
        writer.writerow([f"{mem_start / (1024 * 1024):.4f}", f"{mem_peak / (1024 * 1024):.4f}", f"{mem_current / (1024 * 1024):.4f}", f"{(mem_current - mem_start) / (1024 * 1024):.4f}"])

    print(f"  Memory: Start={mem_start / (1024 * 1024):.2f} MB, Peak={mem_peak / (1024 * 1024):.2f} MB, Current={mem_current / (1024 * 1024):.2f} MB")

    # -------------------------------------------------------------
    # PART 18: Patch Scaling Test (100, 250, 500, 1000 patches)
    # -------------------------------------------------------------
    print("\n[PART 18] Running Scaling Test (100, 250, 500, 1000 patches)...")
    scaling_counts = [100, 250, 500, 1000]
    scaling_results = []

    for count in scaling_counts:
        sub_meas = [generate_standard_measurement(dct_patch(p["patch"]), A_std) for p in all_patch_records[:count]]
        t0 = time.perf_counter()
        iters_list = []
        active_ratios_list = []
        for y_val in sub_meas:
            _, diag = solver_asl.denoise_patch(y_val, return_diagnostics=True)
            iters_list.append(diag["iterations"])
            active_ratios_list.append(diag["active_support_ratio"])
        t_dur = time.perf_counter() - t0
        scaling_results.append((count, t_dur, t_dur / count, float(np.mean(iters_list)), float(np.mean(active_ratios_list))))
        print(f"  {count:4d} patches: TotalTime={t_dur:.3f} s | MeanPerPatch={t_dur / count * 1000:.2f} ms | Iters={np.mean(iters_list):.1f}")

    with open(os.path.join(output_dir, "scaling.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patch_count", "total_solver_time", "mean_solver_time_per_patch", "mean_iterations", "mean_active_ratio"])
        for cnt, tot_t, m_t, m_it, m_act in scaling_results:
            writer.writerow([cnt, f"{tot_t:.4f}", f"{m_t:.6f}", f"{m_it:.2f}", f"{m_act:.4f}"])

    # -------------------------------------------------------------
    # PART 20: Standard vs DC-Preserving Sensing Profile
    # -------------------------------------------------------------
    print("\n[PART 20] Running Standard vs DC-Preserving Profile on 500 Patches...")
    A_ac = generate_dc_sensing(M_ac=37, N_ac=63, seed=sensing_seed)
    solver_dc = HybridSparseSolverV7(A_ac, lambda_reg=0.1, tol=1e-5, seed=seed)

    dc_meas = []
    dc_scalars = []
    for theta in dct_noisy[:num_patches]:
        theta_dc, y_ac = generate_dc_preserving_measurement(theta, A_ac)
        dc_scalars.append(theta_dc)
        dc_meas.append(y_ac)

    t_dc_start = time.perf_counter()
    dc_thetas = []
    dc_diagnostics = []
    for i in range(num_patches):
        z_ac_rec, diag = solver_dc.denoise_patch(
            dc_meas[i],
            use_midpoint=True,
            return_diagnostics=True,
            detailed_profile=True,
            init_method="pinv",
        )
        z_full = restore_dc_component(dc_scalars[i], z_ac_rec)
        dc_thetas.append(z_full)
        dc_diagnostics.append(diag)
    t_dc_solver = time.perf_counter() - t_dc_start

    sp_dc = [{"patch": idct_patch(z.reshape((8, 8))), "x": p["x"], "y": p["y"]} 
             for z, p in zip(dc_thetas, patch_records)]
    sub_recon_dc = reconstruct_image(sp_dc, image_shape=(H, W), patch_size=8)
    metrics_dc = compute_all_metrics(clean_img, sub_recon_dc)

    mean_iters_dc = float(np.mean([d["iterations"] for d in dc_diagnostics]))
    mean_act_dc = float(np.mean([d["active_support_ratio"] for d in dc_diagnostics]))
    mean_res_dc = float(np.mean([d["final_residual"] for d in dc_diagnostics]))

    mean_iters_std = float(np.mean([d["iterations"] for d in asl_patch_diagnostics]))
    mean_act_std = float(np.mean([d["active_support_ratio"] for d in asl_patch_diagnostics]))
    mean_res_std = float(np.mean([d["final_residual"] for d in asl_patch_diagnostics]))

    with open(os.path.join(output_dir, "standard_vs_dc_profile.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sensing_mode", "runtime_seconds", "mean_time_per_patch", "mean_iterations", "active_ratio", "residual", "psnr", "ssim", "mse"])
        writer.writerow(["standard", f"{t_solver:.4f}", f"{t_solver / num_patches:.6f}", f"{mean_iters_std:.2f}", f"{mean_act_std:.4f}", f"{mean_res_std:.6f}", f"{metrics_asl['psnr']:.4f}", f"{metrics_asl['ssim']:.4f}", f"{metrics_asl['mse']:.6f}"])
        writer.writerow(["dc_preserving", f"{t_dc_solver:.4f}", f"{t_dc_solver / num_patches:.6f}", f"{mean_iters_dc:.2f}", f"{mean_act_dc:.4f}", f"{mean_res_dc:.6f}", f"{metrics_dc['psnr']:.4f}", f"{metrics_dc['ssim']:.4f}", f"{metrics_dc['mse']:.6f}"])

    print(f"  Standard Sensing: Time={t_solver:.3f} s | Iters={mean_iters_std:.1f} | PSNR={metrics_asl['psnr']:.2f} dB | SSIM={metrics_asl['ssim']:.4f}")
    print(f"  DC-Preserving:    Time={t_dc_solver:.3f} s | Iters={mean_iters_dc:.1f} | PSNR={metrics_dc['psnr']:.2f} dB | SSIM={metrics_dc['ssim']:.4f}")

    print("\n[DONE] Profiling data collection completed successfully.")
    return True


if __name__ == "__main__":
    run_profiling_suite()
