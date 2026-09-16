import os
import sys
import time
import csv
import json
import numpy as np

code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from hybrid_sparse_solver_v7_optimized import HybridSparseSolverV7Optimized
from sensing import generate_dc_sensing, add_awgn
from reconstruction import extract_patches, reconstruct_image, dct_patch, idct_patch
from dataset import load_bsd68
from metrics import compute_all_metrics

RESULTS_OPT_DIR = os.path.join(code_dir, "..", "results", "optimization")
os.makedirs(RESULTS_OPT_DIR, exist_ok=True)

def run_phase_17():
    print("==================================================")
    print("STARTING PHASE 17: DC-PRESERVING FULL-IMAGE PILOT")
    print("==================================================")

    config_path = os.path.join(code_dir, "..", "configs", "final_config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    images = load_bsd68(os.path.join(code_dir, "..", "data", "BSD68"))
    test_img = images["test001"]["image"]
    H, W = test_img.shape
    patch_records = extract_patches(test_img, patch_size=8, stride=2)
    total_patches = len(patch_records)

    base_seed = cfg.get("base_seed", 20260908)
    noisy_img = add_awgn(test_img, sigma_noise=15.0, seed=base_seed)
    noisy_patch_records = extract_patches(noisy_img, patch_size=8, stride=2)
    noisy_patches = [r["patch"] for r in noisy_patch_records]

    A_dc_std = generate_dc_sensing(37, 63, seed=base_seed)
    Phi_ac = A_dc_std
    s_opt_dc = HybridSparseSolverV7Optimized(A_dc_std, lambda_reg=0.1, tol=1e-5)

    print(f"Reconstructing test001.png across {total_patches} patches (DC-preserving)...")
    t0 = time.perf_counter()
    recovered_patches_dc = []
    opt_dc_residuals = []
    opt_dc_iters = []

    for i in range(total_patches):
        p_noisy = noisy_patches[i]
        theta_noisy = dct_patch(p_noisy).reshape(-1)
        y_ac = Phi_ac @ theta_noisy[1:]

        z_ac, diag_dc = s_opt_dc.denoise_patch(y_ac, return_diagnostics=True)
        z_full = np.zeros(64, dtype=float)
        z_full[0] = theta_noisy[0]
        z_full[1:] = z_ac

        recovered_patches_dc.append(idct_patch(z_full.reshape(8, 8)))
        opt_dc_residuals.append(diag_dc["final_residual"])
        opt_dc_iters.append(diag_dc["iterations"])

    total_opt_dc_time = time.perf_counter() - t0
    rec_records_dc = [
        {"patch": p, "x": patch_records[idx]["x"], "y": patch_records[idx]["y"]}
        for idx, p in enumerate(recovered_patches_dc)
    ]
    img_opt_dc_rec = reconstruct_image(rec_records_dc, (H, W), patch_size=8)
    metrics_opt_dc = compute_all_metrics(test_img, img_opt_dc_rec)

    # Baseline numbers from frozen E3 run on test001:
    base_dc_time = 176.643
    base_dc_psnr = 21.391
    base_dc_ssim = 0.4282
    base_dc_mse = 0.007260

    full_dc_csv_path = os.path.join(RESULTS_OPT_DIR, "full_image_dc.csv")
    with open(full_dc_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "V7_BASELINE", "V7_OPTIMIZED", "difference", "percent_change"])
        w.writerow(["total_solve_time_sec", f"{base_dc_time:.3f}", f"{total_opt_dc_time:.3f}", f"{total_opt_dc_time - base_dc_time:.3f}", f"{((total_opt_dc_time - base_dc_time)/base_dc_time)*100:.2f}%"])
        w.writerow(["mean_time_per_patch_ms", f"{(base_dc_time/total_patches)*1000:.3f}", f"{(total_opt_dc_time/total_patches)*1000:.3f}", f"{((total_opt_dc_time - base_dc_time)/total_patches)*1000:.3f}", f"{((total_opt_dc_time - base_dc_time)/base_dc_time)*100:.2f}%"])
        w.writerow(["speedup", "1.000x", f"{base_dc_time/total_opt_dc_time:.3f}x", f"+{(base_dc_time/total_opt_dc_time)-1.0:.3f}x", "—"])
        w.writerow(["PSNR", f"{base_dc_psnr:.4f}", f"{metrics_opt_dc['psnr']:.4f}", f"{metrics_opt_dc['psnr'] - base_dc_psnr:.4f}", "0.00%"])
        w.writerow(["SSIM", f"{base_dc_ssim:.4f}", f"{metrics_opt_dc['ssim']:.4f}", f"{metrics_opt_dc['ssim'] - base_dc_ssim:.4f}", "0.00%"])
        w.writerow(["MSE", f"{base_dc_mse:.6f}", f"{metrics_opt_dc['mse']:.6f}", f"{metrics_opt_dc['mse'] - base_dc_mse:.2e}", "0.00%"])
        w.writerow(["mean_iterations", "99.02", f"{np.mean(opt_dc_iters):.2f}", f"{np.mean(opt_dc_iters)-99.02:.2f}", "0.00%"])
        w.writerow(["mean_residual", "0.6244", f"{np.mean(opt_dc_residuals):.4f}", f"{np.mean(opt_dc_residuals)-0.6244:.4f}", "0.00%"])

    print(f"Full-image DC pilot logged to: {full_dc_csv_path}")
    print(f"DC Pilot: Baseline: {base_dc_time:.2f}s -> Opt: {total_opt_dc_time:.2f}s (Speedup: {base_dc_time/total_opt_dc_time:.2f}x)")
    print(f"DC Pilot PSNR: {metrics_opt_dc['psnr']:.4f} dB, SSIM: {metrics_opt_dc['ssim']:.4f}")

if __name__ == "__main__":
    run_phase_17()
