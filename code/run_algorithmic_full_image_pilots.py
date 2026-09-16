"""
Full-Image Pilot Runner for Promising Algorithmic Candidates (Parts 21 & 22).
Evaluates full test001.png (37,604 patches, sigma_noise=15) for:
1. V7_OPT_BASE (Standard Sensing)
2. V7_A5_TWO_STAGE (Standard Sensing)
3. V7_OPT_BASE (DC-Preserving Sensing)
4. V7_A5_TWO_STAGE (DC-Preserving Sensing)

Saves results to:
results/algorithmic_variants/full_image_pilots.csv
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
import csv
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dataset import load_bsd68
from sensing import generate_random_sensing, generate_dc_sensing, add_awgn
from reconstruction import extract_patches, dct_patch, idct_patch, reconstruct_image
from metrics import compute_all_metrics
from hybrid_sparse_solver_v7_optimized import HybridSparseSolverV7Optimized

RESULTS_DIR = os.path.join(current_dir, "..", "results", "algorithmic_variants")
os.makedirs(RESULTS_DIR, exist_ok=True)


def solve_two_stage_batch(solver, A, Y_batch, prune_thresh=1e-3):
    """
    Batched two-stage solver:
    Stage 1: Accelerated Batched GEMM recovery
    Stage 2: Least-squares debiasing on active support
    """
    # Stage 1: GEMM micro-batch
    Z_stage1 = solver.denoise_batch(Y_batch)  # (N, B)
    N, B = Z_stage1.shape
    M = Y_batch.shape[0]

    Z_stage2 = np.zeros((N, B), dtype=float)

    # Stage 2: Independent fast least squares debiasing
    for b in range(B):
        z_b = Z_stage1[:, b]
        supp = np.abs(z_b) > prune_thresh
        cnt = int(np.count_nonzero(supp))

        if cnt == 0 or cnt >= M:
            if cnt == 0:
                supp = np.ones(N, dtype=bool)
            else:
                top_idx = np.argsort(np.abs(z_b))[-M:]
                supp = np.zeros(N, dtype=bool)
                supp[top_idx] = True

        A_supp = A[:, supp]
        y_b = Y_batch[:, b]
        z_supp = np.linalg.pinv(A_supp) @ y_b
        Z_stage2[supp, b] = z_supp

    return Z_stage2


def run_full_image_pilots():
    print("\n==================================================")
    print("RUNNING FULL-IMAGE PILOTS (37,604 PATCHES, sigma=15)")
    print("==================================================")

    data_dir = os.path.join(current_dir, "..", "data", "BSD68")
    images = load_bsd68(data_dir)
    test_img = images["test001"]["image"]
    H, W = test_img.shape
    patch_records = extract_patches(test_img, patch_size=8, stride=2)
    total_patches = len(patch_records)  # 37,604

    base_seed = 20260908
    M_std, N = 38, 64
    M_ac, N_ac = 37, 63

    A_std = generate_random_sensing(M_std, N, seed=base_seed)
    A_dc = generate_dc_sensing(M_ac, N_ac, seed=base_seed)

    noisy_img = add_awgn(test_img, sigma_noise=15.0, seed=base_seed)
    noisy_records = extract_patches(noisy_img, patch_size=8, stride=2)
    noisy_patches = [r["patch"] for r in noisy_records]

    print("Pre-transforming measurements for all 37,604 patches...")
    noisy_thetas = np.array([dct_patch(p).reshape(-1) for p in noisy_patches])  # (37604, 64)
    Y_std = (A_std @ noisy_thetas.T).T  # (37604, 38)

    Y_ac = (A_dc @ noisy_thetas[:, 1:].T).T  # (37604, 37)
    Y_dc = noisy_thetas[:, 0]  # (37604,)

    solver_std = HybridSparseSolverV7Optimized(A_std, lambda_reg=0.1, tol=1e-5)
    solver_dc = HybridSparseSolverV7Optimized(A_dc, lambda_reg=0.1, tol=1e-5)

    batch_size = 50
    results = []

    # -------------------------------------------------------------------------
    # 1. V7_OPT_BASE (Standard Sensing)
    # -------------------------------------------------------------------------
    print("\n[1/4] Running V7_OPT_BASE (Standard Sensing)...")
    t0 = time.perf_counter()
    z_list = []
    for start in range(0, total_patches, batch_size):
        end = min(start + batch_size, total_patches)
        Y_sub = Y_std[start:end].T
        Z_sub = solver_std.denoise_batch(Y_sub)
        z_list.append(Z_sub.T)
    t_std_base = time.perf_counter() - t0
    Z_all = np.vstack(z_list)

    rec_records = [
        {"patch": idct_patch(Z_all[i].reshape(8, 8)), "x": patch_records[i]["x"], "y": patch_records[i]["y"]}
        for i in range(total_patches)
    ]
    img_rec = reconstruct_image(rec_records, (H, W), patch_size=8)
    metrics_std_base = compute_all_metrics(test_img, img_rec)
    res_std_base = float(np.mean(np.linalg.norm(Z_all @ A_std.T - Y_std, axis=1)))

    print(f"  V7_OPT_BASE (Std): Solve Time: {t_std_base:.2f} s ({(t_std_base/total_patches)*1000:.3f} ms/patch), PSNR: {metrics_std_base['psnr']:.2f} dB, SSIM: {metrics_std_base['ssim']:.4f}, Res: {res_std_base:.4f}")
    results.append({
        "configuration": "V7_OPT_BASE (Standard)",
        "sensing_type": "standard",
        "solve_time_sec": t_std_base,
        "ms_per_patch": (t_std_base / total_patches) * 1000.0,
        "psnr": metrics_std_base["psnr"],
        "ssim": metrics_std_base["ssim"],
        "mse": metrics_std_base["mse"],
        "residual": res_std_base,
    })

    # -------------------------------------------------------------------------
    # 2. V7_A5_TWO_STAGE (Standard Sensing)
    # -------------------------------------------------------------------------
    print("\n[2/4] Running V7_A5_TWO_STAGE (Standard Sensing)...")
    t0 = time.perf_counter()
    z_list_a5 = []
    for start in range(0, total_patches, batch_size):
        end = min(start + batch_size, total_patches)
        Y_sub = Y_std[start:end].T
        Z_sub = solve_two_stage_batch(solver_std, A_std, Y_sub, prune_thresh=1e-3)
        z_list_a5.append(Z_sub.T)
    t_std_a5 = time.perf_counter() - t0
    Z_all_a5 = np.vstack(z_list_a5)

    rec_records_a5 = [
        {"patch": idct_patch(Z_all_a5[i].reshape(8, 8)), "x": patch_records[i]["x"], "y": patch_records[i]["y"]}
        for i in range(total_patches)
    ]
    img_rec_a5 = reconstruct_image(rec_records_a5, (H, W), patch_size=8)
    metrics_std_a5 = compute_all_metrics(test_img, img_rec_a5)
    res_std_a5 = float(np.mean(np.linalg.norm(Z_all_a5 @ A_std.T - Y_std, axis=1)))

    d_psnr_a5 = metrics_std_a5["psnr"] - metrics_std_base["psnr"]
    print(f"  V7_A5_TWO_STAGE (Std): Solve Time: {t_std_a5:.2f} s ({(t_std_a5/total_patches)*1000:.3f} ms/patch), PSNR: {metrics_std_a5['psnr']:.2f} dB (dPSNR: {d_psnr_a5:+.2f} dB), SSIM: {metrics_std_a5['ssim']:.4f}, Res: {res_std_a5:.4f}")
    results.append({
        "configuration": "V7_A5_TWO_STAGE (Standard)",
        "sensing_type": "standard",
        "solve_time_sec": t_std_a5,
        "ms_per_patch": (t_std_a5 / total_patches) * 1000.0,
        "psnr": metrics_std_a5["psnr"],
        "ssim": metrics_std_a5["ssim"],
        "mse": metrics_std_a5["mse"],
        "residual": res_std_a5,
    })

    # -------------------------------------------------------------------------
    # 3. V7_OPT_BASE (DC-Preserving Sensing)
    # -------------------------------------------------------------------------
    print("\n[3/4] Running V7_OPT_BASE (DC-Preserving Sensing)...")
    t0 = time.perf_counter()
    z_ac_list = []
    for start in range(0, total_patches, batch_size):
        end = min(start + batch_size, total_patches)
        Y_sub = Y_ac[start:end].T
        Z_sub = solver_dc.denoise_batch(Y_sub)
        z_ac_list.append(Z_sub.T)
    t_dc_base = time.perf_counter() - t0
    Z_ac_all = np.vstack(z_ac_list)

    # Prepend DC
    Z_full_dc = np.empty((total_patches, 64), dtype=float)
    Z_full_dc[:, 0] = Y_dc
    Z_full_dc[:, 1:] = Z_ac_all

    rec_records_dc = [
        {"patch": idct_patch(Z_full_dc[i].reshape(8, 8)), "x": patch_records[i]["x"], "y": patch_records[i]["y"]}
        for i in range(total_patches)
    ]
    img_rec_dc = reconstruct_image(rec_records_dc, (H, W), patch_size=8)
    metrics_dc_base = compute_all_metrics(test_img, img_rec_dc)
    res_dc_base = float(np.mean(np.linalg.norm(Z_ac_all @ A_dc.T - Y_ac, axis=1)))

    print(f"  V7_OPT_BASE (DC): Solve Time: {t_dc_base:.2f} s ({(t_dc_base/total_patches)*1000:.3f} ms/patch), PSNR: {metrics_dc_base['psnr']:.2f} dB, SSIM: {metrics_dc_base['ssim']:.4f}, AC Res: {res_dc_base:.4f}")
    results.append({
        "configuration": "V7_OPT_BASE (DC-Preserving)",
        "sensing_type": "dc_preserving",
        "solve_time_sec": t_dc_base,
        "ms_per_patch": (t_dc_base / total_patches) * 1000.0,
        "psnr": metrics_dc_base["psnr"],
        "ssim": metrics_dc_base["ssim"],
        "mse": metrics_dc_base["mse"],
        "residual": res_dc_base,
    })

    # -------------------------------------------------------------------------
    # 4. V7_A5_TWO_STAGE (DC-Preserving Sensing)
    # -------------------------------------------------------------------------
    print("\n[4/4] Running V7_A5_TWO_STAGE (DC-Preserving Sensing)...")
    t0 = time.perf_counter()
    z_ac_list_a5 = []
    for start in range(0, total_patches, batch_size):
        end = min(start + batch_size, total_patches)
        Y_sub = Y_ac[start:end].T
        Z_sub = solve_two_stage_batch(solver_dc, A_dc, Y_sub, prune_thresh=1e-3)
        z_ac_list_a5.append(Z_sub.T)
    t_dc_a5 = time.perf_counter() - t0
    Z_ac_all_a5 = np.vstack(z_ac_list_a5)

    # Prepend DC
    Z_full_dc_a5 = np.empty((total_patches, 64), dtype=float)
    Z_full_dc_a5[:, 0] = Y_dc
    Z_full_dc_a5[:, 1:] = Z_ac_all_a5

    rec_records_dc_a5 = [
        {"patch": idct_patch(Z_full_dc_a5[i].reshape(8, 8)), "x": patch_records[i]["x"], "y": patch_records[i]["y"]}
        for i in range(total_patches)
    ]
    img_rec_dc_a5 = reconstruct_image(rec_records_dc_a5, (H, W), patch_size=8)
    metrics_dc_a5 = compute_all_metrics(test_img, img_rec_dc_a5)
    res_dc_a5 = float(np.mean(np.linalg.norm(Z_ac_all_a5 @ A_dc.T - Y_ac, axis=1)))

    d_psnr_dc_a5 = metrics_dc_a5["psnr"] - metrics_dc_base["psnr"]
    print(f"  V7_A5_TWO_STAGE (DC): Solve Time: {t_dc_a5:.2f} s ({(t_dc_a5/total_patches)*1000:.3f} ms/patch), PSNR: {metrics_dc_a5['psnr']:.2f} dB (dPSNR: {d_psnr_dc_a5:+.2f} dB), SSIM: {metrics_dc_a5['ssim']:.4f}, AC Res: {res_dc_a5:.4f}")
    results.append({
        "configuration": "V7_A5_TWO_STAGE (DC-Preserving)",
        "sensing_type": "dc_preserving",
        "solve_time_sec": t_dc_a5,
        "ms_per_patch": (t_dc_a5 / total_patches) * 1000.0,
        "psnr": metrics_dc_a5["psnr"],
        "ssim": metrics_dc_a5["ssim"],
        "mse": metrics_dc_a5["mse"],
        "residual": res_dc_a5,
    })

    # Save full image pilot results
    pilot_csv = os.path.join(RESULTS_DIR, "full_image_pilots.csv")
    with open(pilot_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["configuration", "sensing_type", "solve_time_sec", "ms_per_patch", "psnr", "ssim", "mse", "residual"])
        for r in results:
            w.writerow([
                r["configuration"],
                r["sensing_type"],
                f"{r['solve_time_sec']:.3f}",
                f"{r['ms_per_patch']:.3f}",
                f"{r['psnr']:.4f}",
                f"{r['ssim']:.4f}",
                f"{r['mse']:.6f}",
                f"{r['residual']:.4f}",
            ])
    print(f"\nSaved: {pilot_csv}")


if __name__ == "__main__":
    run_full_image_pilots()
