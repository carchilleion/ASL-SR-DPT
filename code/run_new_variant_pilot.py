"""
ASL-SR-DPT Stage 1: New-Variant Pilot Execution Engine (750 Evaluations)

Evaluates:
1. Frozen ASL-SR-DPT (lambda=0.10, sigma_min=0.01, sigma_decay=0.95, DC-preserving)
2. Fixed Candidate ASL-SR-DPT (lambda=0.01, sigma_min=0.10, sigma_decay=0.90, DC-preserving)
3. Noise-Adaptive ASL-SR-DPT (lambda(sigma)=0.01*(sigma/15), sigma_min=0.10, sigma_decay=0.90, DC-preserving)
4. OMP (Canonical baseline, M=38, N=64, column-normalized, rel_tol=1e-5, max_coeff=38)
5. LASSO-ADMM (Canonical baseline, M=38, N=64, lambda=0.01, rho=1.0, max_iter=100, tol=1e-4)

Dataset: 10 BSD68 images (test001 to test010)
Noise levels: sigma in {15.0, 25.0, 50.0}
Trials: 1 to 5
Total Workload: 10 * 3 * 5 * 5 = 750 full-image evaluations

Features:
- Configurable MAX_CPU_WORKERS with worker clamping and thread isolation.
- Deterministic seed formulas matching project invariants.
- Atomic checkpointing with composite-key deduplication (image_id, noise_sigma, trial, method, configuration).
- Graceful pause via sentinel flag file (.pause_requested).
- Live manifest tracking (benchmark_manifest.json).
- Explicit failure logging (no silent drops).
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 1. Enforce strict single-threading per process before NumPy/BLAS imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import json
import csv
import hashlib
import argparse
import datetime
import shutil
import multiprocessing as mp
import concurrent.futures
import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(REPO_ROOT, "code")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from dataset import load_bsd68, get_bsd68_filenames, load_image_grayscale
from sensing import generate_random_sensing, generate_dc_sensing
from reconstruction import extract_patches, dct_patch, idct_patch, reconstruct_image
from metrics import compute_psnr, compute_ssim, compute_mse
from hybrid_sparse_solver_v7_optimized import (
    HybridSparseSolverV7Optimized,
    precompute_omp,
    run_omp,
    precompute_lasso_admm,
    run_lasso_admm,
)
import app

OUTPUT_DIR = os.path.join(REPO_ROOT, "results", "new_variant_pilot")
RAW_CSV_PATH = os.path.join(OUTPUT_DIR, "raw_results.csv")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "benchmark_manifest.json")
RECON_DIR = os.path.join(OUTPUT_DIR, "reconstructions")
PAUSE_FILE_PATH = os.path.join(OUTPUT_DIR, ".pause_requested")
PID_FILE_PATH = os.path.join(OUTPUT_DIR, ".runner_pid")
LIVE_LOG_PATH = os.path.join(OUTPUT_DIR, "live_terminal.log")
LIVE_PREVIEW_RECON = os.path.join(OUTPUT_DIR, "live_preview_recon.png")
LIVE_PREVIEW_CLEAN = os.path.join(OUTPUT_DIR, "live_preview_clean.png")
LIVE_PREVIEW_NOISY = os.path.join(OUTPUT_DIR, "live_preview_noisy.png")
LIVE_PREVIEW_META = os.path.join(OUTPUT_DIR, "live_preview_meta.json")

BASE_SEED = 20260908
PILOT_IMAGES = [f"test{i:03d}" for i in range(1, 11)]
NOISE_LEVELS = [15.0, 25.0, 50.0]
PILOT_TRIALS = [1, 2, 3, 4, 5]
EXPECTED_TOTAL_EVALS = 10 * 3 * 5 * 5  # Exactly 750

METHODS_ORDER = [
    "Frozen ASL-SR-DPT",
    "Fixed Candidate ASL-SR-DPT",
    "Noise-Adaptive ASL-SR-DPT",
    "OMP",
    "LASSO-ADMM",
]

CSV_HEADER = [
    "image_id",
    "noise_sigma",
    "trial",
    "method",
    "configuration",
    "psnr",
    "ssim",
    "mse",
    "runtime_seconds",
    "sensing_seed",
    "noise_seed",
    "code_hash",
    "config_hash",
    "max_cpu_workers",
    "actual_cpu_workers",
    "timestamp",
    "solver_status",
    "reconstruction_status",
    "lambda_val",
    "sigma_min",
    "sigma_decay",
    "sigma_final",
    "iteration_count",
    "accepted_step_count",
    "failed_line_searches",
    "active_support_ratio",
    "mean_active_support",
    "measurement_residual",
    "fidelity_term",
    "sparsity_term",
    "total_objective",
    "low_band_retention",
    "mid_band_retention",
    "high_band_retention",
    "admm_primal_residual",
    "admm_dual_residual",
    "admm_iterations",
    "error_message",
]

# Frequency coordinates for 8x8 DCT bands
freq_coords = [divmod(k, 8) for k in range(1, 64)]
manhattan_freqs = np.array([u + v for u, v in freq_coords])
mask_low = (manhattan_freqs >= 1) & (manhattan_freqs <= 3)    # 9 coeffs
mask_mid = (manhattan_freqs >= 4) & (manhattan_freqs <= 7)    # 26 coeffs
mask_high = (manhattan_freqs >= 8) & (manhattan_freqs <= 14)  # 28 coeffs


def log_event(msg, also_stdout=True):
    """Write timestamped event to both stdout and live_terminal.log."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_stdout:
        try:
            print(line, flush=True)
        except Exception:
            try:
                print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
            except Exception:
                pass
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(LIVE_LOG_PATH, "a", encoding="utf-8", errors="replace") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:
        pass


def compute_file_sha256(filepath):
    if not os.path.exists(filepath):
        return "missing"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_seeds(image_id, noise_sigma, trial):
    clean_id = int(str(image_id).replace("test", "").lstrip("0") or "0")
    sensing_seed = BASE_SEED + clean_id * 1000000 + int(noise_sigma) * 1000 + int(trial)
    noise_seed = BASE_SEED + int(trial) * 100000 + int(noise_sigma) * 1000 + clean_id
    return sensing_seed, noise_seed


def get_worker_configuration(requested_workers=None):
    avail = os.cpu_count() or 1
    if requested_workers is None:
        env_val = os.environ.get("MAX_CPU_WORKERS")
        if env_val:
            try:
                requested_workers = int(env_val)
            except ValueError:
                requested_workers = max(1, avail - 1)
        else:
            requested_workers = max(1, avail - 1)
    else:
        requested_workers = int(requested_workers)

    actual = max(1, min(requested_workers, avail))
    return requested_workers, avail, actual


def load_completed_keys(raw_csv_path):
    keys = set()
    if os.path.exists(raw_csv_path) and os.path.getsize(raw_csv_path) > 0:
        with open(raw_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    k = (
                        str(r["image_id"]),
                        float(r["noise_sigma"]),
                        int(r["trial"]),
                        str(r["method"]),
                        str(r.get("configuration", "")),
                    )
                    keys.add(k)
                except (KeyError, ValueError):
                    pass
    return keys


def evaluate_single_evaluation(eval_spec):
    """
    Evaluates exactly ONE full image solver task: (image_id, noise_sigma, trial, method, configuration).
    Returns a metric record dict.
    Deterministic, isolated, and safe against failures.
    """
    image_id = eval_spec["image_id"]
    noise_sigma = float(eval_spec["noise_sigma"])
    trial = int(eval_spec["trial"])
    m_name = eval_spec["method"]
    cfg_name = eval_spec["configuration"]
    code_hash = eval_spec.get("code_hash", "")
    config_hash = eval_spec.get("config_hash", "")
    max_workers = eval_spec.get("max_cpu_workers", 1)
    actual_workers = eval_spec.get("actual_cpu_workers", 1)
    save_preview = eval_spec.get("save_preview", False)

    sensing_seed, noise_seed = compute_seeds(image_id, noise_sigma, trial)

    img_path = os.path.join(REPO_ROOT, "data", "BSD68", f"{image_id}.png")
    clean_img = load_image_grayscale(img_path)
    H, W = clean_img.shape

    rng = np.random.default_rng(noise_seed)
    noise_arr = (noise_sigma / 255.0) * rng.standard_normal((H, W))
    noisy_img = np.clip(clean_img + noise_arr, 0.0, 1.0)

    clean_patches = extract_patches(clean_img, patch_size=8, stride=2)
    noisy_patches = extract_patches(noisy_img, patch_size=8, stride=2)
    num_patches = len(clean_patches)

    clean_thetas = np.array([dct_patch(r["patch"]).reshape(-1) for r in clean_patches])
    noisy_thetas = np.array([dct_patch(r["patch"]).reshape(-1) for r in noisy_patches])

    clean_dc = clean_thetas[:, 0]
    clean_ac = clean_thetas[:, 1:]
    noisy_dc = noisy_thetas[:, 0]
    noisy_ac = noisy_thetas[:, 1:]

    E_clean_total = float(np.sum(clean_ac ** 2))
    E_clean_low = float(np.sum(clean_ac[:, mask_low] ** 2))
    E_clean_mid = float(np.sum(clean_ac[:, mask_mid] ** 2))
    E_clean_high = float(np.sum(clean_ac[:, mask_high] ** 2))

    os.makedirs(RECON_DIR, exist_ok=True)
    clean_out = os.path.join(RECON_DIR, f"{image_id}_clean.png")
    noisy_out = os.path.join(RECON_DIR, f"{image_id}_sigma{int(noise_sigma)}_trial{trial}_noisy.png")
    if not os.path.exists(clean_out):
        Image.fromarray(np.clip(clean_img * 255.0, 0, 255).astype(np.uint8), mode="L").save(clean_out)
    if not os.path.exists(noisy_out):
        Image.fromarray(np.clip(noisy_img * 255.0, 0, 255).astype(np.uint8), mode="L").save(noisy_out)

    t0 = time.perf_counter()

    try:
        if m_name in ("Frozen ASL-SR-DPT", "Fixed Candidate ASL-SR-DPT", "Noise-Adaptive ASL-SR-DPT"):
            if m_name == "Frozen ASL-SR-DPT":
                lambda_val = 0.10
                s_min = 0.01
                s_decay = 0.95
            elif m_name == "Fixed Candidate ASL-SR-DPT":
                lambda_val = 0.01
                s_min = 0.10
                s_decay = 0.90
            elif m_name == "Noise-Adaptive ASL-SR-DPT":
                lambda_val = 0.01 * (noise_sigma / 15.0)
                s_min = 0.10
                s_decay = 0.90

            A_ac = generate_dc_sensing(M_ac=37, N_ac=63, seed=sensing_seed)
            Y_ac = (A_ac @ noisy_ac.T).T

            solver = HybridSparseSolverV7Optimized(A=A_ac, lambda_reg=lambda_val, tol=1e-5)
            batch_size = 1000
            recovered_ac_list = []
            total_accepted_steps = 0
            total_failed_ls = 0
            total_iterations_sum = 0
            all_active_ratios = []
            all_final_sigmas = []

            for start in range(0, num_patches, batch_size):
                end = min(start + batch_size, num_patches)
                Y_sub = Y_ac[start:end].T
                Z_sub, diag_sub = solver.denoise_batch(
                    Y_sub,
                    sigma_min=s_min,
                    decrease_factor=s_decay,
                    max_iter=150,
                    initial_mu=0.20,
                    armijo_c=1e-4,
                    beta_decay=0.5,
                    max_backtracks=10,
                    support_threshold_multiplier=1e-5,
                    support_reopen_interval=3,
                    use_midpoint=True,
                    return_diagnostics=True,
                )
                recovered_ac_list.append(Z_sub.T)
                total_accepted_steps += int(np.sum(diag_sub["patch_accepted_steps"]))
                total_failed_ls += int(np.sum(diag_sub["patch_failed_line_searches"]))
                total_iterations_sum += int(np.sum(diag_sub["patch_iterations"]))
                all_active_ratios.append(float(diag_sub["active_support_ratio"]))
                all_final_sigmas.append(float(diag_sub["final_sigma"]))

            rec_ac = np.vstack(recovered_ac_list)
            rec_thetas = np.empty((num_patches, 64), dtype=float)
            rec_thetas[:, 0] = noisy_dc
            rec_thetas[:, 1:] = rec_ac

            rec_patch_records = []
            for i, r in enumerate(noisy_patches):
                p_spatial = idct_patch(rec_thetas[i].reshape((8, 8)))
                rec_patch_records.append({"x": r["x"], "y": r["y"], "patch": p_spatial})

            rec_img = reconstruct_image(rec_patch_records, (H, W), patch_size=8)
            solve_time = time.perf_counter() - t0

            psnr_val = float(compute_psnr(clean_img, rec_img))
            ssim_val = float(compute_ssim(clean_img, rec_img))
            mse_val = float(compute_mse(clean_img, rec_img))

            E_rec_total = float(np.sum(rec_ac ** 2))
            E_rec_low = float(np.sum(rec_ac[:, mask_low] ** 2))
            E_rec_mid = float(np.sum(rec_ac[:, mask_mid] ** 2))
            E_rec_high = float(np.sum(rec_ac[:, mask_high] ** 2))

            low_ret = float(E_rec_low / E_clean_low) if E_clean_low > 0 else 0.0
            mid_ret = float(E_rec_mid / E_clean_mid) if E_clean_mid > 0 else 0.0
            high_ret = float(E_rec_high / E_clean_high) if E_clean_high > 0 else 0.0

            R_meas = (A_ac @ rec_ac.T).T - Y_ac
            res_norms = np.linalg.norm(R_meas, axis=1)
            mean_meas_residual = float(np.mean(res_norms))
            mean_fidelity = float(np.mean(0.5 * (res_norms ** 2)))
            mean_final_sigma = float(np.mean(all_final_sigmas))

            exp_arg = -0.5 * (rec_ac ** 2) / (mean_final_sigma ** 2)
            w_mat = np.exp(exp_arg)
            mean_sparsity = float(np.mean(-lambda_val * np.sum(w_mat, axis=1)))
            total_obj = mean_fidelity + mean_sparsity

            mean_iter = float(total_iterations_sum / num_patches)
            active_ratio = float(np.mean(all_active_ratios))
            mean_active_count = float(active_ratio * 63.0)

            slug = m_name.replace(" ", "_").lower()
            out_path = os.path.join(RECON_DIR, f"{image_id}_sigma{int(noise_sigma)}_trial{trial}_{slug}.png")
            if save_preview or trial <= 3:
                Image.fromarray(np.clip(rec_img * 255.0, 0, 255).astype(np.uint8), mode="L").save(out_path)
                preview_path = out_path
            else:
                temp_path = os.path.join(RECON_DIR, f"temp_{image_id}_{slug}.png")
                Image.fromarray(np.clip(rec_img * 255.0, 0, 255).astype(np.uint8), mode="L").save(temp_path)
                preview_path = temp_path

            return {
                "_preview_recon": preview_path,
                "_preview_clean": clean_out,
                "_preview_noisy": noisy_out,
                "image_id": image_id,
                "noise_sigma": noise_sigma,
                "trial": trial,
                "method": m_name,
                "configuration": cfg_name,
                "psnr": psnr_val,
                "ssim": ssim_val,
                "mse": mse_val,
                "runtime_seconds": float(solve_time),
                "sensing_seed": sensing_seed,
                "noise_seed": noise_seed,
                "code_hash": code_hash,
                "config_hash": config_hash,
                "max_cpu_workers": max_workers,
                "actual_cpu_workers": actual_workers,
                "timestamp": datetime.datetime.now().isoformat(),
                "solver_status": "SUCCESS",
                "reconstruction_status": "SUCCESS",
                "lambda_val": float(lambda_val),
                "sigma_min": float(s_min),
                "sigma_decay": float(s_decay),
                "sigma_final": float(mean_final_sigma),
                "iteration_count": mean_iter,
                "accepted_step_count": int(total_accepted_steps),
                "failed_line_searches": int(total_failed_ls),
                "active_support_ratio": active_ratio,
                "mean_active_support": mean_active_count,
                "measurement_residual": mean_meas_residual,
                "fidelity_term": mean_fidelity,
                "sparsity_term": mean_sparsity,
                "total_objective": total_obj,
                "low_band_retention": low_ret,
                "mid_band_retention": mid_ret,
                "high_band_retention": high_ret,
                "admm_primal_residual": "",
                "admm_dual_residual": "",
                "admm_iterations": "",
                "error_message": "",
            }

        elif m_name == "OMP":
            A_std = generate_random_sensing(M=38, N=64, seed=sensing_seed)
            Y_std = (A_std @ noisy_thetas.T).T
            pre_omp = precompute_omp(A_std)

            rec_thetas_b = np.empty((num_patches, 64), dtype=float)
            for p_idx in range(num_patches):
                y_p = Y_std[p_idx]
                sol_p = run_omp(A_std, y_p, max_coefficients=38, relative_residual_tol=1e-5, precomputed=pre_omp)
                rec_thetas_b[p_idx] = sol_p

            rec_patch_records = []
            for i, r in enumerate(noisy_patches):
                p_spatial = idct_patch(rec_thetas_b[i].reshape((8, 8)))
                rec_patch_records.append({"x": r["x"], "y": r["y"], "patch": p_spatial})

            rec_img = reconstruct_image(rec_patch_records, (H, W), patch_size=8)
            solve_time = time.perf_counter() - t0

            psnr_val = float(compute_psnr(clean_img, rec_img))
            ssim_val = float(compute_ssim(clean_img, rec_img))
            mse_val = float(compute_mse(clean_img, rec_img))

            R_meas = (A_std @ rec_thetas_b.T).T - Y_std
            mean_meas_residual = float(np.mean(np.linalg.norm(R_meas, axis=1)))

            slug = m_name.replace(" ", "_").lower()
            out_path = os.path.join(RECON_DIR, f"{image_id}_sigma{int(noise_sigma)}_trial{trial}_{slug}.png")
            if save_preview or trial <= 3:
                Image.fromarray(np.clip(rec_img * 255.0, 0, 255).astype(np.uint8), mode="L").save(out_path)
                preview_path = out_path
            else:
                temp_path = os.path.join(RECON_DIR, f"temp_{image_id}_{slug}.png")
                Image.fromarray(np.clip(rec_img * 255.0, 0, 255).astype(np.uint8), mode="L").save(temp_path)
                preview_path = temp_path

            return {
                "_preview_recon": preview_path,
                "_preview_clean": clean_out,
                "_preview_noisy": noisy_out,
                "image_id": image_id,
                "noise_sigma": noise_sigma,
                "trial": trial,
                "method": m_name,
                "configuration": cfg_name,
                "psnr": psnr_val,
                "ssim": ssim_val,
                "mse": mse_val,
                "runtime_seconds": float(solve_time),
                "sensing_seed": sensing_seed,
                "noise_seed": noise_seed,
                "code_hash": code_hash,
                "config_hash": config_hash,
                "max_cpu_workers": max_workers,
                "actual_cpu_workers": actual_workers,
                "timestamp": datetime.datetime.now().isoformat(),
                "solver_status": "SUCCESS",
                "reconstruction_status": "SUCCESS",
                "lambda_val": "",
                "sigma_min": "",
                "sigma_decay": "",
                "sigma_final": "",
                "iteration_count": "",
                "accepted_step_count": "",
                "failed_line_searches": "",
                "active_support_ratio": "",
                "mean_active_support": "",
                "measurement_residual": mean_meas_residual,
                "fidelity_term": "",
                "sparsity_term": "",
                "total_objective": "",
                "low_band_retention": "",
                "mid_band_retention": "",
                "high_band_retention": "",
                "admm_primal_residual": "",
                "admm_dual_residual": "",
                "admm_iterations": "",
                "error_message": "",
            }

        elif m_name == "LASSO-ADMM":
            A_std = generate_random_sensing(M=38, N=64, seed=sensing_seed)
            Y_std = (A_std @ noisy_thetas.T).T
            pre_admm = precompute_lasso_admm(A_std, rho=1.0)

            rec_thetas_b = np.empty((num_patches, 64), dtype=float)
            p_res_list, d_res_list, iter_list = [], [], []
            for p_idx in range(num_patches):
                y_p = Y_std[p_idx]
                sol_p, diag_p = run_lasso_admm(
                    A_std, y_p, lambda_reg=0.01, rho=1.0, max_iter=100, tol=1e-4, precomputed=pre_admm, return_diagnostics=True
                )
                rec_thetas_b[p_idx] = sol_p
                p_res_list.append(float(diag_p.get("primal_residual", 0.0)))
                d_res_list.append(float(diag_p.get("dual_residual", 0.0)))
                iter_list.append(int(diag_p.get("iterations", 100)))

            admm_p_res = float(np.mean(p_res_list))
            admm_d_res = float(np.mean(d_res_list))
            admm_iter = float(np.mean(iter_list))

            rec_patch_records = []
            for i, r in enumerate(noisy_patches):
                p_spatial = idct_patch(rec_thetas_b[i].reshape((8, 8)))
                rec_patch_records.append({"x": r["x"], "y": r["y"], "patch": p_spatial})

            rec_img = reconstruct_image(rec_patch_records, (H, W), patch_size=8)
            solve_time = time.perf_counter() - t0

            psnr_val = float(compute_psnr(clean_img, rec_img))
            ssim_val = float(compute_ssim(clean_img, rec_img))
            mse_val = float(compute_mse(clean_img, rec_img))

            R_meas = (A_std @ rec_thetas_b.T).T - Y_std
            mean_meas_residual = float(np.mean(np.linalg.norm(R_meas, axis=1)))

            slug = m_name.replace(" ", "_").lower()
            out_path = os.path.join(RECON_DIR, f"{image_id}_sigma{int(noise_sigma)}_trial{trial}_{slug}.png")
            if save_preview or trial <= 3:
                Image.fromarray(np.clip(rec_img * 255.0, 0, 255).astype(np.uint8), mode="L").save(out_path)
                preview_path = out_path
            else:
                temp_path = os.path.join(RECON_DIR, f"temp_{image_id}_{slug}.png")
                Image.fromarray(np.clip(rec_img * 255.0, 0, 255).astype(np.uint8), mode="L").save(temp_path)
                preview_path = temp_path

            return {
                "_preview_recon": preview_path,
                "_preview_clean": clean_out,
                "_preview_noisy": noisy_out,
                "image_id": image_id,
                "noise_sigma": noise_sigma,
                "trial": trial,
                "method": m_name,
                "configuration": cfg_name,
                "psnr": psnr_val,
                "ssim": ssim_val,
                "mse": mse_val,
                "runtime_seconds": float(solve_time),
                "sensing_seed": sensing_seed,
                "noise_seed": noise_seed,
                "code_hash": code_hash,
                "config_hash": config_hash,
                "max_cpu_workers": max_workers,
                "actual_cpu_workers": actual_workers,
                "timestamp": datetime.datetime.now().isoformat(),
                "solver_status": "SUCCESS",
                "reconstruction_status": "SUCCESS",
                "lambda_val": "",
                "sigma_min": "",
                "sigma_decay": "",
                "sigma_final": "",
                "iteration_count": "",
                "accepted_step_count": "",
                "failed_line_searches": "",
                "active_support_ratio": "",
                "mean_active_support": "",
                "measurement_residual": mean_meas_residual,
                "fidelity_term": "",
                "sparsity_term": "",
                "total_objective": "",
                "low_band_retention": "",
                "mid_band_retention": "",
                "high_band_retention": "",
                "admm_primal_residual": admm_p_res,
                "admm_dual_residual": admm_d_res,
                "admm_iterations": admm_iter,
                "error_message": "",
            }
        else:
            raise ValueError(f"Unknown method: {m_name}")

    except Exception as exc:
        err_msg = str(exc)
        log_event(f"[ERROR] Evaluation failed for {image_id} sigma={noise_sigma} trial={trial} method={m_name}: {err_msg}")
        return {
            "_preview_recon": "",
            "_preview_clean": clean_out,
            "_preview_noisy": noisy_out,
            "image_id": image_id,
            "noise_sigma": noise_sigma,
            "trial": trial,
            "method": m_name,
            "configuration": cfg_name,
            "psnr": "",
            "ssim": "",
            "mse": "",
            "runtime_seconds": float(time.perf_counter() - t0),
            "sensing_seed": sensing_seed,
            "noise_seed": noise_seed,
            "code_hash": code_hash,
            "config_hash": config_hash,
            "max_cpu_workers": max_workers,
            "actual_cpu_workers": actual_workers,
            "timestamp": datetime.datetime.now().isoformat(),
            "solver_status": "FAILED",
            "reconstruction_status": "FAILED",
            "lambda_val": "",
            "sigma_min": "",
            "sigma_decay": "",
            "sigma_final": "",
            "iteration_count": "",
            "accepted_step_count": "",
            "failed_line_searches": "",
            "active_support_ratio": "",
            "mean_active_support": "",
            "measurement_residual": "",
            "fidelity_term": "",
            "sparsity_term": "",
            "total_objective": "",
            "low_band_retention": "",
            "mid_band_retention": "",
            "high_band_retention": "",
            "admm_primal_residual": "",
            "admm_dual_residual": "",
            "admm_iterations": "",
            "error_message": err_msg,
        }


def update_manifest(completed_count, requested_workers, actual_workers, elapsed_seconds=0.0, rate_per_min=0.0, eta_seconds=0.0):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest_data = {
        "benchmark_name": "ASL_SR_DPT_STAGE_1_NEW_VARIANT_PILOT",
        "dataset": "BSD68_PILOT_10",
        "total_images": len(PILOT_IMAGES),
        "noise_levels": NOISE_LEVELS,
        "trials_per_condition": len(PILOT_TRIALS),
        "total_methods": len(METHODS_ORDER),
        "expected_evaluations": EXPECTED_TOTAL_EVALS,
        "completed_evaluations": completed_count,
        "remaining_evaluations": max(0, EXPECTED_TOTAL_EVALS - completed_count),
        "percent_complete": round((completed_count / EXPECTED_TOTAL_EVALS) * 100.0, 2),
        "requested_cpu_workers": requested_workers,
        "available_cpu_workers": os.cpu_count(),
        "actual_cpu_workers": actual_workers,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "evaluations_per_minute": round(rate_per_min, 2),
        "estimated_remaining_seconds": round(eta_seconds, 2),
        "last_updated": datetime.datetime.now().isoformat(),
        "status": "COMPLETED" if completed_count >= EXPECTED_TOTAL_EVALS else "IN_PROGRESS",
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)


def run_pilot(max_cpu_workers=None, confirm_full=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RECON_DIR, exist_ok=True)

    code_hash = app.CODE_HASH
    config_hash = compute_file_sha256(os.path.join(REPO_ROOT, "configs", "pilot_config.json"))

    req_workers, avail_workers, actual_workers = get_worker_configuration(max_cpu_workers)
    print(f"\n[Worker Configuration]")
    print(f"  Requested workers: {req_workers}")
    print(f"  Available logical CPUs: {avail_workers}")
    print(f"  Actual workers used: {actual_workers}")

    # Record PID
    try:
        with open(PID_FILE_PATH, "w", encoding="utf-8") as pf:
            pf.write(str(os.getpid()))
    except Exception:
        pass

    # Initialize CSV if not exists
    if not os.path.exists(RAW_CSV_PATH) or os.path.getsize(RAW_CSV_PATH) == 0:
        with open(RAW_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()

    completed_keys = load_completed_keys(RAW_CSV_PATH)
    completed_count = len(completed_keys)

    # Build queue of individual evaluation tasks
    eval_queue = []
    for img_id in PILOT_IMAGES:
        for noise in NOISE_LEVELS:
            for trial in PILOT_TRIALS:
                for m in METHODS_ORDER:
                    cfg = "ASL_FROZEN" if m == "Frozen ASL-SR-DPT" else (
                        "ASL_FIXED_CANDIDATE" if m == "Fixed Candidate ASL-SR-DPT" else (
                            "ASL_NOISE_ADAPTIVE" if m == "Noise-Adaptive ASL-SR-DPT" else (
                                "OMP_BASELINE" if m == "OMP" else "LASSO_ADMM_BASELINE"
                            )
                        )
                    )
                    k = (img_id, float(noise), int(trial), m, cfg)
                    if k not in completed_keys:
                        is_preview = (img_id in ("test001", "test004", "test008") and trial == 1)
                        eval_queue.append({
                            "image_id": img_id,
                            "noise_sigma": noise,
                            "trial": trial,
                            "method": m,
                            "configuration": cfg,
                            "code_hash": code_hash,
                            "config_hash": config_hash,
                            "max_cpu_workers": req_workers,
                            "actual_cpu_workers": actual_workers,
                            "save_preview": is_preview,
                        })

    remaining_count = len(eval_queue)
    log_event("=" * 60)
    log_event("ASL-SR-DPT Stage 1: New-Variant Pilot Startup Audit")
    log_event("=" * 60)
    log_event(f"Expected evaluations: {EXPECTED_TOTAL_EVALS}")
    log_event(f"Completed evaluations: {completed_count}")
    log_event(f"Remaining evaluations: {remaining_count}")
    log_event(f"MAX_CPU_WORKERS: {req_workers} (Active: {actual_workers})")
    log_event(f"Target scope: 10 images, 3 noises, 5 trials, 5 methods")
    log_event("=" * 60)

    if remaining_count == 0:
        log_event("[Status] All 750 pilot evaluations are already completed. No pending tasks.")
        update_manifest(completed_count, req_workers, actual_workers)
        return

    update_manifest(completed_count, req_workers, actual_workers)

    start_time = time.perf_counter()
    session_completed = 0
    paused = False

    while completed_count < EXPECTED_TOTAL_EVALS and not paused:
        if os.path.exists(PAUSE_FILE_PATH):
            log_event("[PAUSE] Sentinel file .pause_requested detected. Pausing gracefully.")
            paused = True
            break

        completed_keys = load_completed_keys(RAW_CSV_PATH)
        completed_count = len(completed_keys)
        if completed_count >= EXPECTED_TOTAL_EVALS:
            break

        # Re-build queue of remaining tasks
        eval_queue = []
        for img_id in PILOT_IMAGES:
            for noise in NOISE_LEVELS:
                for trial in PILOT_TRIALS:
                    for m in METHODS_ORDER:
                        cfg = "ASL_FROZEN" if m == "Frozen ASL-SR-DPT" else (
                            "ASL_FIXED_CANDIDATE" if m == "Fixed Candidate ASL-SR-DPT" else (
                                "ASL_NOISE_ADAPTIVE" if m == "Noise-Adaptive ASL-SR-DPT" else (
                                    "OMP_BASELINE" if m == "OMP" else "LASSO_ADMM_BASELINE"
                                )
                            )
                        )
                        k = (img_id, float(noise), int(trial), m, cfg)
                        if k not in completed_keys:
                            is_preview = (img_id in ("test001", "test004", "test008") and trial == 1)
                            eval_queue.append({
                                "image_id": img_id,
                                "noise_sigma": noise,
                                "trial": trial,
                                "method": m,
                                "configuration": cfg,
                                "code_hash": code_hash,
                                "config_hash": config_hash,
                                "max_cpu_workers": req_workers,
                                "actual_cpu_workers": actual_workers,
                                "save_preview": is_preview,
                            })

        if not eval_queue:
            break

        # Process in batches of 28 evaluations (2 waves of 14 workers)
        batch_size = max(14, actual_workers * 2)
        chunk = eval_queue[:batch_size]

        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
                future_to_spec = {executor.submit(evaluate_single_evaluation, spec): spec for spec in chunk}

                for future in concurrent.futures.as_completed(future_to_spec):
                    if os.path.exists(PAUSE_FILE_PATH):
                        log_event("[PAUSE] Sentinel file .pause_requested detected. Pausing gracefully.")
                        paused = True
                        for f in future_to_spec:
                            f.cancel()
                        break

                    try:
                        record = future.result()
                    except Exception as exc:
                        spec = future_to_spec[future]
                        log_event(f"[FATAL] Future raised unhandled exception for {spec['image_id']} {spec['method']}: {exc}")
                        record = {
                            "image_id": spec["image_id"],
                            "noise_sigma": spec["noise_sigma"],
                            "trial": spec["trial"],
                            "method": spec["method"],
                            "configuration": spec["configuration"],
                            "psnr": "",
                            "ssim": "",
                            "mse": "",
                            "runtime_seconds": 0.0,
                            "sensing_seed": "",
                            "noise_seed": "",
                            "code_hash": spec.get("code_hash", ""),
                            "config_hash": spec.get("config_hash", ""),
                            "max_cpu_workers": spec.get("max_cpu_workers", 1),
                            "actual_cpu_workers": spec.get("actual_cpu_workers", 1),
                            "timestamp": datetime.datetime.now().isoformat(),
                            "solver_status": "FAILED",
                            "reconstruction_status": "FAILED",
                            "error_message": f"{type(exc).__name__}: {str(exc)}",
                        }
                        for col in CSV_HEADER:
                            if col not in record:
                                record[col] = ""

                    preview_recon = record.pop("_preview_recon", "")
                    preview_clean = record.pop("_preview_clean", "")
                    preview_noisy = record.pop("_preview_noisy", "")

                    with open(RAW_CSV_PATH, "a", newline="", encoding="utf-8") as csv_file:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
                        csv_writer.writerow(record)

                    completed_count += 1
                    session_completed += 1

                    elapsed = time.perf_counter() - start_time
                    rate = (session_completed / elapsed) * 60.0 if elapsed > 0 else 0.0
                    rem_tasks = EXPECTED_TOTAL_EVALS - completed_count
                    eta_s = (rem_tasks / (rate / 60.0)) if rate > 0 else 0.0
                    eta_str = f"{eta_s / 3600.0:.1f}h" if eta_s >= 3600 else f"{eta_s / 60.0:.1f}m"

                    img_id = record["image_id"]
                    noise_s = int(float(record["noise_sigma"]))
                    tr = record["trial"]
                    meth = record["method"]
                    status = record["solver_status"]

                    if status == "SUCCESS":
                        psnr_s = f"{float(record['psnr']):.2f} dB"
                        ssim_s = f"{float(record['ssim']):.4f}"
                        t_s = f"{float(record['runtime_seconds']):.2f}s"
                        log_event(f"[DONE] {img_id} | sigma={noise_s} | Trial {tr} | {meth} -> PSNR: {psnr_s}, SSIM: {ssim_s}, Time: {t_s} | Progress: {completed_count}/{EXPECTED_TOTAL_EVALS} ({completed_count / EXPECTED_TOTAL_EVALS * 100.0:.1f}%) | Rate: {rate:.1f} evals/min | ETA: {eta_str}")
                    else:
                        log_event(f"[FAIL] {img_id} | sigma={noise_s} | Trial {tr} | {meth} -> FAILED: {record.get('error_message', '')} | Progress: {completed_count}/{EXPECTED_TOTAL_EVALS}")

                    # Update live preview assets
                    try:
                        if preview_recon and os.path.exists(preview_recon):
                            shutil.copyfile(preview_recon, LIVE_PREVIEW_RECON)
                        if preview_clean and os.path.exists(preview_clean):
                            shutil.copyfile(preview_clean, LIVE_PREVIEW_CLEAN)
                        if preview_noisy and os.path.exists(preview_noisy):
                            shutil.copyfile(preview_noisy, LIVE_PREVIEW_NOISY)

                        meta_dict = {
                            "image_id": img_id,
                            "noise_sigma": float(noise_s),
                            "trial": int(tr),
                            "method": meth,
                            "configuration": record.get("configuration", ""),
                            "psnr": float(record["psnr"]) if record.get("psnr") != "" else 0.0,
                            "ssim": float(record["ssim"]) if record.get("ssim") != "" else 0.0,
                            "mse": float(record["mse"]) if record.get("mse") != "" else 0.0,
                            "runtime_seconds": float(record["runtime_seconds"]) if record.get("runtime_seconds") != "" else 0.0,
                            "measurement_residual": float(record["measurement_residual"]) if record.get("measurement_residual") != "" else 0.0,
                            "timestamp": record["timestamp"],
                            "completed_count": completed_count,
                            "expected_total": EXPECTED_TOTAL_EVALS,
                            "evals_per_minute": round(rate, 2),
                            "eta_string": eta_str,
                        }
                        with open(LIVE_PREVIEW_META, "w", encoding="utf-8") as pf:
                            json.dump(meta_dict, pf, indent=2)
                    except Exception:
                        pass

                    if session_completed % 5 == 0 or completed_count >= EXPECTED_TOTAL_EVALS:
                        update_manifest(completed_count, req_workers, actual_workers, elapsed, rate, eta_s)

        except Exception as pool_exc:
            log_event(f"[WARNING] Process pool exception encountered: {pool_exc}. Refreshing worker pool for remaining tasks.")
            time.sleep(1)

    total_elapsed = time.perf_counter() - start_time
    final_rate = (session_completed / total_elapsed) * 60.0 if total_elapsed > 0 else 0.0
    update_manifest(completed_count, req_workers, actual_workers, total_elapsed, final_rate, 0.0)

    log_event("")
    log_event("[Execution Summary]")
    log_event(f"  Total completed records: {completed_count} / {EXPECTED_TOTAL_EVALS}")
    log_event(f"  Session evaluations: {session_completed}")
    log_event(f"  Elapsed runtime: {total_elapsed:.2f} seconds")
    log_event(f"  Status: {'PAUSED' if paused else ('COMPLETE' if completed_count >= EXPECTED_TOTAL_EVALS else 'INCOMPLETE')}")

    try:
        if os.path.exists(PID_FILE_PATH):
            os.remove(PID_FILE_PATH)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="ASL-SR-DPT Stage 1 New-Variant Pilot Runner")
    parser.add_argument("--max-cpu-workers", type=int, default=None, help="Max CPU workers to use")
    parser.add_argument("--confirm-full", action="store_true", help="Confirm running pilot")
    args = parser.parse_args()

    run_pilot(max_cpu_workers=args.max_cpu_workers, confirm_full=args.confirm_full)


if __name__ == "__main__":
    main()
