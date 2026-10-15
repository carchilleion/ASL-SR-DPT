"""
ASL-SR-DPT Final Benchmark: Distributed PC Execution Engine (3-PC Distributed Architecture)

Executes exactly one designated partition (pc1, pc2, or pc3):
- 3,400 conditions per PC
- 3 methods per condition (Noise-Adaptive ASL-SR-DPT, OMP, LASSO-ADMM)
- Total evaluations per PC: 3,400 x 3 = 10,200 evaluations.

Key Operational Controls:
- Configurable MAX_CPU_WORKERS (default 10, clamped to system logical CPUs).
- Strict single-thread BLAS/OpenMP isolation per process.
- Loads noisy image and seeds deterministically once per condition; all 3 methods evaluate the exact identical noisy input.
- Atomic append to raw_results.csv; failures explicitly logged to error_log.csv.
- Dynamic resume support (reads completed keys from raw_results.csv).
- Graceful pause via .pause_requested in local PC output directory.
- Updates benchmark_manifest.json with real-time throughput and ETA.
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

from dataset import load_image_grayscale
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

CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "final_benchmark_3pc_config.json")
PARTITIONS_DIR = os.path.join(REPO_ROOT, "results", "final_benchmark", "partitions")

METHODS_ORDER = [
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
    "noisy_input_hash",
    "reconstruction_hash",
    "code_hash",
    "config_hash",
    "partition_id",
    "condition_id",
    "requested_max_cpu_workers",
    "actual_cpu_workers",
    "timestamp",
    "solver_status",
    "reconstruction_status",
    "effective_lambda",
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
mask_low = (manhattan_freqs >= 1) & (manhattan_freqs <= 3)    # 9 AC coefficients
mask_mid = (manhattan_freqs >= 4) & (manhattan_freqs <= 7)    # 26 AC coefficients
mask_high = (manhattan_freqs >= 8) & (manhattan_freqs <= 14)  # 28 AC coefficients


def compute_file_sha256(filepath):
    """Computes SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return "missing"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_worker_configuration(requested_workers=None):
    """
    Resolves worker count respecting configured limit (10) and available logical CPUs.
    """
    avail = os.cpu_count() or 1
    if requested_workers is None:
        env_val = os.environ.get("MAX_CPU_WORKERS")
        if env_val:
            try:
                requested_workers = int(env_val)
            except ValueError:
                requested_workers = 10
        else:
            requested_workers = 10
    else:
        requested_workers = int(requested_workers)

    # Clamping rule: strictly enforce max 10 workers regardless of requested value or CPU cores
    actual = max(1, min(requested_workers, 10, avail))
    return requested_workers, avail, actual


def load_partition_conditions(partition_id):
    """
    Loads condition list from results/final_benchmark/partitions/{partition_id}_conditions.csv.
    """
    csv_path = os.path.join(PARTITIONS_DIR, f"{partition_id}_conditions.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Partition condition file not found at: {csv_path}. Run generate_final_benchmark_partitions.py first.")

    conditions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            conditions.append({
                "condition_id": int(r["condition_id"]),
                "partition_id": str(r["partition_id"]),
                "image_id": str(r["image_id"]),
                "noise_sigma": float(r["noise_sigma"]),
                "trial": int(r["trial"]),
                "sensing_seed": int(r["sensing_seed"]),
                "noise_seed": int(r["noise_seed"]),
            })
    return conditions


def load_prior_failure_counts(error_csv_path):
    """
    Reads error_log.csv to determine how many times each (image_id, noise_sigma, trial, method) has failed.
    """
    failure_counts = {}
    if os.path.exists(error_csv_path) and os.path.getsize(error_csv_path) > 0:
        with open(error_csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # header
            for row in reader:
                if len(row) >= 5:
                    try:
                        k = (str(row[1]), float(row[2]), int(row[3]), str(row[4]))
                        failure_counts[k] = failure_counts.get(k, 0) + 1
                    except (ValueError, IndexError):
                        pass
    return failure_counts


def make_failed_record(spec, err_str, code_hash, config_hash, partition_id, req_workers, actual_workers):
    """Generates standard failed evaluation dictionary adhering strictly to CSV_HEADER."""
    return {
        "image_id": spec["image_id"],
        "noise_sigma": spec["noise_sigma"],
        "trial": spec["trial"],
        "method": spec["method"],
        "configuration": "UNKNOWN",
        "psnr": "",
        "ssim": "",
        "mse": "",
        "runtime_seconds": 0.0,
        "sensing_seed": spec["sensing_seed"],
        "noise_seed": spec["noise_seed"],
        "noisy_input_hash": "",
        "reconstruction_hash": "",
        "code_hash": code_hash,
        "config_hash": config_hash,
        "partition_id": partition_id,
        "condition_id": spec["condition_id"],
        "requested_max_cpu_workers": req_workers,
        "actual_cpu_workers": actual_workers,
        "timestamp": datetime.datetime.now().isoformat(),
        "solver_status": "FAILED",
        "reconstruction_status": "FAILED",
        "effective_lambda": "",
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
        "error_message": err_str,
    }


def load_and_audit_existing_results(raw_csv_path, expected_code_hash, expected_config_hash,
                                    expected_methods, clean_incompatible=False, live_log_path=None):
    """
    Audits existing raw_results.csv before resuming.
    Validates code_hash, config_hash, expected methods, and solver_status.
    Rejects and quarantines incompatible or invalid records instead of silently accepting them.

    Returns:
        successful_keys: set of valid (image_id, noise_sigma, trial, method) tuples
        incompatible_records: list of tuples (record, reason)
        failed_records: list of failed records
    """
    successful_keys = set()
    valid_rows = []
    incompatible_records = []
    duplicate_records = []
    failed_records = []

    if not os.path.exists(raw_csv_path) or os.path.getsize(raw_csv_path) == 0:
        return successful_keys, incompatible_records, failed_records

    with open(raw_csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                img_id = str(r.get("image_id", ""))
                noise_s = float(r.get("noise_sigma", 0))
                tr = int(r.get("trial", 0))
                meth = str(r.get("method", ""))
                k = (img_id, noise_s, tr, meth)

                status = r.get("solver_status", "")
                if status == "SUCCESS":
                    c_hash = str(r.get("code_hash", ""))
                    cfg_hash = str(r.get("config_hash", ""))

                    reasons = []
                    if expected_code_hash and c_hash != expected_code_hash:
                        reasons.append(f"code_hash mismatch ({c_hash[:8]} vs {expected_code_hash[:8]})")
                    if expected_config_hash and cfg_hash != expected_config_hash:
                        reasons.append(f"config_hash mismatch ({cfg_hash[:8]} vs {expected_config_hash[:8]})")
                    if meth not in expected_methods:
                        reasons.append(f"unknown method ({meth})")
                    try:
                        psnr_val = float(r.get("psnr", 0))
                        if psnr_val <= 0 or np.isnan(psnr_val) or np.isinf(psnr_val):
                            reasons.append(f"invalid PSNR: {psnr_val}")
                    except (ValueError, TypeError):
                        reasons.append("unparseable PSNR")

                    if reasons:
                        incompatible_records.append((r, "; ".join(reasons)))
                    elif k in successful_keys:
                        duplicate_records.append(r)
                    else:
                        successful_keys.add(k)
                        valid_rows.append(r)
                else:
                    failed_records.append(r)
            except Exception as e:
                incompatible_records.append((r, f"Parse error: {e}"))

    # Quarantine and clean raw_results.csv if incompatibilities or duplicates were found
    if clean_incompatible and (incompatible_records or duplicate_records or failed_records):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = raw_csv_path.replace(".csv", f"_pre_resume_backup_{ts}.csv")
        try:
            shutil.copyfile(raw_csv_path, backup_path)
            log_event(f"[RESUME AUDIT] Backed up previous results file with incompatibilities to: {os.path.basename(backup_path)}", live_log_path)
        except Exception:
            pass

        if incompatible_records:
            quarantine_path = os.path.join(os.path.dirname(raw_csv_path), "quarantined_records.csv")
            with open(quarantine_path, "a", newline="", encoding="utf-8") as qf:
                qwriter = csv.writer(qf)
                if not os.path.exists(quarantine_path) or os.path.getsize(quarantine_path) == 0:
                    qwriter.writerow(["quarantine_timestamp", "image_id", "noise_sigma", "trial", "method", "code_hash", "config_hash", "reasons"])
                for rec, reas in incompatible_records:
                    qwriter.writerow([ts, rec.get("image_id"), rec.get("noise_sigma"), rec.get("trial"), rec.get("method"), rec.get("code_hash"), rec.get("config_hash"), reas])
            log_event(f"[RESUME AUDIT] Quarantined {len(incompatible_records)} incompatible records. These conditions will be cleanly re-evaluated.", live_log_path)

        # Cleanly re-write raw_results.csv to retain ONLY valid successful rows
        with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()
            for row in valid_rows:
                writer.writerow(row)
        log_event(f"[RESUME AUDIT] Cleaned raw_results.csv: {len(valid_rows)} verified records retained.", live_log_path)

    return successful_keys, incompatible_records, failed_records


def load_completed_keys(raw_csv_path, expected_code_hash=None, expected_config_hash=None):
    """
    Loads set of completed valid keys: (image_id, noise_sigma, trial, method).
    Validates code and config hashes if provided (or defaults to current environment hashes).
    """
    if expected_code_hash is None:
        expected_code_hash = app.CODE_HASH
    if expected_config_hash is None:
        expected_config_hash = compute_file_sha256(CONFIG_PATH)

    keys, _, _ = load_and_audit_existing_results(
        raw_csv_path=raw_csv_path,
        expected_code_hash=expected_code_hash,
        expected_config_hash=expected_config_hash,
        expected_methods=METHODS_ORDER,
        clean_incompatible=False,
    )
    return keys


def evaluate_single_task(task_spec):
    """
    Executes one evaluation task: (condition, method).
    Generates noisy input once per condition deterministically via noise_seed.
    Evaluates requested method with bitwise consistency.
    """
    image_id = task_spec["image_id"]
    noise_sigma = float(task_spec["noise_sigma"])
    trial = int(task_spec["trial"])
    m_name = task_spec["method"]
    condition_id = task_spec["condition_id"]
    partition_id = task_spec["partition_id"]
    sensing_seed = int(task_spec["sensing_seed"])
    noise_seed = int(task_spec["noise_seed"])
    code_hash = task_spec.get("code_hash", "")
    config_hash = task_spec.get("config_hash", "")
    max_workers = task_spec.get("max_cpu_workers", 10)
    actual_workers = task_spec.get("actual_cpu_workers", 10)
    save_preview = task_spec.get("save_preview", False)
    recon_dir = task_spec.get("recon_dir", "")

    # 1. Load Clean Image
    img_path = os.path.join(REPO_ROOT, "data", "BSD68", f"{image_id}.png")
    clean_img = load_image_grayscale(img_path)
    H, W = clean_img.shape

    # 2. Generate Noisy Image Deterministically via noise_seed
    rng = np.random.default_rng(noise_seed)
    noise_arr = (noise_sigma / 255.0) * rng.standard_normal((H, W))
    noisy_img = np.clip(clean_img + noise_arr, 0.0, 1.0)
    noisy_input_hash = hashlib.sha256(noisy_img.tobytes()).hexdigest()

    # 3. Extract Patches & DCT Transform
    clean_patches = extract_patches(clean_img, patch_size=8, stride=2)
    noisy_patches = extract_patches(noisy_img, patch_size=8, stride=2)
    num_patches = len(clean_patches)

    clean_thetas = np.array([dct_patch(r["patch"]).reshape(-1) for r in clean_patches])
    noisy_thetas = np.array([dct_patch(r["patch"]).reshape(-1) for r in noisy_patches])

    clean_ac = clean_thetas[:, 1:]
    noisy_dc = noisy_thetas[:, 0]
    noisy_ac = noisy_thetas[:, 1:]

    E_clean_low = float(np.sum(clean_ac[:, mask_low] ** 2))
    E_clean_mid = float(np.sum(clean_ac[:, mask_mid] ** 2))
    E_clean_high = float(np.sum(clean_ac[:, mask_high] ** 2))

    if save_preview and recon_dir:
        clean_out = os.path.join(recon_dir, f"{image_id}_clean.png")
        noisy_out = os.path.join(recon_dir, f"{image_id}_sigma{int(noise_sigma)}_trial{trial}_noisy.png")
        if not os.path.exists(clean_out):
            try:
                Image.fromarray(np.clip(clean_img * 255.0, 0, 255).astype(np.uint8), mode="L").save(clean_out)
            except Exception:
                pass
        if not os.path.exists(noisy_out):
            try:
                Image.fromarray(np.clip(noisy_img * 255.0, 0, 255).astype(np.uint8), mode="L").save(noisy_out)
            except Exception:
                pass

    t0 = time.perf_counter()

    try:
        if m_name == "Noise-Adaptive ASL-SR-DPT":
            cfg_name = "ASL_NOISE_ADAPTIVE"
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
            rec_hash = hashlib.sha256(rec_img.tobytes()).hexdigest()

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
            mean_sparsity = float(np.mean(lambda_val * np.sum(np.abs(rec_ac), axis=1)))
            mean_total_obj = float(mean_fidelity + mean_sparsity)

            mean_iter = float(total_iterations_sum / num_patches)
            mean_accepted = float(total_accepted_steps / num_patches)
            mean_failed_ls = float(total_failed_ls / num_patches)
            mean_active_ratio = float(np.mean(all_active_ratios))
            mean_final_sigma = float(np.mean(all_final_sigmas))

            if save_preview and recon_dir:
                slug = m_name.replace(" ", "_").lower()
                out_path = os.path.join(recon_dir, f"{image_id}_sigma{int(noise_sigma)}_trial{trial}_{slug}.png")
                Image.fromarray(np.clip(rec_img * 255.0, 0, 255).astype(np.uint8), mode="L").save(out_path)

            return {
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
                "noisy_input_hash": noisy_input_hash,
                "reconstruction_hash": rec_hash,
                "code_hash": code_hash,
                "config_hash": config_hash,
                "partition_id": partition_id,
                "condition_id": condition_id,
                "requested_max_cpu_workers": max_workers,
                "actual_cpu_workers": actual_workers,
                "timestamp": datetime.datetime.now().isoformat(),
                "solver_status": "SUCCESS",
                "reconstruction_status": "SUCCESS",
                "effective_lambda": float(lambda_val),
                "sigma_min": float(s_min),
                "sigma_decay": float(s_decay),
                "sigma_final": mean_final_sigma,
                "iteration_count": mean_iter,
                "accepted_step_count": mean_accepted,
                "failed_line_searches": mean_failed_ls,
                "active_support_ratio": mean_active_ratio,
                "mean_active_support": float(mean_active_ratio * 63.0),
                "measurement_residual": mean_meas_residual,
                "fidelity_term": mean_fidelity,
                "sparsity_term": mean_sparsity,
                "total_objective": mean_total_obj,
                "low_band_retention": low_ret,
                "mid_band_retention": mid_ret,
                "high_band_retention": high_ret,
                "admm_primal_residual": "",
                "admm_dual_residual": "",
                "admm_iterations": "",
                "error_message": "",
            }

        elif m_name == "OMP":
            cfg_name = "OMP_BASELINE"
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
            rec_hash = hashlib.sha256(rec_img.tobytes()).hexdigest()

            R_meas = (A_std @ rec_thetas_b.T).T - Y_std
            mean_meas_residual = float(np.mean(np.linalg.norm(R_meas, axis=1)))

            if save_preview and recon_dir:
                slug = m_name.replace(" ", "_").lower()
                out_path = os.path.join(recon_dir, f"{image_id}_sigma{int(noise_sigma)}_trial{trial}_{slug}.png")
                Image.fromarray(np.clip(rec_img * 255.0, 0, 255).astype(np.uint8), mode="L").save(out_path)

            return {
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
                "noisy_input_hash": noisy_input_hash,
                "reconstruction_hash": rec_hash,
                "code_hash": code_hash,
                "config_hash": config_hash,
                "partition_id": partition_id,
                "condition_id": condition_id,
                "requested_max_cpu_workers": max_workers,
                "actual_cpu_workers": actual_workers,
                "timestamp": datetime.datetime.now().isoformat(),
                "solver_status": "SUCCESS",
                "reconstruction_status": "SUCCESS",
                "effective_lambda": "",
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
            cfg_name = "LASSO_ADMM_BASELINE"
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

            rec_patch_records = []
            for i, r in enumerate(noisy_patches):
                p_spatial = idct_patch(rec_thetas_b[i].reshape((8, 8)))
                rec_patch_records.append({"x": r["x"], "y": r["y"], "patch": p_spatial})

            rec_img = reconstruct_image(rec_patch_records, (H, W), patch_size=8)
            solve_time = time.perf_counter() - t0

            psnr_val = float(compute_psnr(clean_img, rec_img))
            ssim_val = float(compute_ssim(clean_img, rec_img))
            mse_val = float(compute_mse(clean_img, rec_img))
            rec_hash = hashlib.sha256(rec_img.tobytes()).hexdigest()

            R_meas = (A_std @ rec_thetas_b.T).T - Y_std
            mean_meas_residual = float(np.mean(np.linalg.norm(R_meas, axis=1)))

            if save_preview and recon_dir:
                slug = m_name.replace(" ", "_").lower()
                out_path = os.path.join(recon_dir, f"{image_id}_sigma{int(noise_sigma)}_trial{trial}_{slug}.png")
                Image.fromarray(np.clip(rec_img * 255.0, 0, 255).astype(np.uint8), mode="L").save(out_path)

            return {
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
                "noisy_input_hash": noisy_input_hash,
                "reconstruction_hash": rec_hash,
                "code_hash": code_hash,
                "config_hash": config_hash,
                "partition_id": partition_id,
                "condition_id": condition_id,
                "requested_max_cpu_workers": max_workers,
                "actual_cpu_workers": actual_workers,
                "timestamp": datetime.datetime.now().isoformat(),
                "solver_status": "SUCCESS",
                "reconstruction_status": "SUCCESS",
                "effective_lambda": 0.01,
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
                "admm_primal_residual": float(np.mean(p_res_list)),
                "admm_dual_residual": float(np.mean(d_res_list)),
                "admm_iterations": float(np.mean(iter_list)),
                "error_message": "",
            }

    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {str(exc)}"
        return {
            "image_id": image_id,
            "noise_sigma": noise_sigma,
            "trial": trial,
            "method": m_name,
            "configuration": "UNKNOWN",
            "psnr": "",
            "ssim": "",
            "mse": "",
            "runtime_seconds": float(time.perf_counter() - t0),
            "sensing_seed": sensing_seed,
            "noise_seed": noise_seed,
            "noisy_input_hash": noisy_input_hash if "noisy_input_hash" in locals() else "",
            "reconstruction_hash": "",
            "code_hash": code_hash,
            "config_hash": config_hash,
            "partition_id": partition_id,
            "condition_id": condition_id,
            "requested_max_cpu_workers": max_workers,
            "actual_cpu_workers": actual_workers,
            "timestamp": datetime.datetime.now().isoformat(),
            "solver_status": "FAILED",
            "reconstruction_status": "FAILED",
            "effective_lambda": "",
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


def log_event(msg, log_path, also_stdout=True):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_stdout:
        try:
            print(line, flush=True)
        except Exception:
            pass
    try:
        with open(log_path, "a", encoding="utf-8", errors="replace") as f:
            f.write(line + "\n")
    except Exception:
        pass


def update_pc_manifest(manifest_path, partition_id, successful_evals, total_expected_evals,
                       req_workers, actual_workers, elapsed_seconds=0.0, rate_per_min=0.0, eta_seconds=0.0,
                       failed_attempts=0, permanent_failures=0, status=None):
    """Updates benchmark_manifest.json with live telemetry and execution state."""
    if status is None:
        if successful_evals >= total_expected_evals:
            status = "COMPLETED"
        elif permanent_failures > 0 and (successful_evals + permanent_failures >= total_expected_evals):
            status = "COMPLETED_WITH_FAILURES"
        else:
            status = "IN_PROGRESS"

    remaining = max(0, total_expected_evals - successful_evals - permanent_failures)
    manifest_data = {
        "benchmark_name": "ASL_SR_DPT_FINAL_BSD68_BENCHMARK_3PC_DISTRIBUTED",
        "partition_id": partition_id,
        "expected_evaluations": total_expected_evals,
        "completed_evaluations": successful_evals,
        "successful_evaluations": successful_evals,
        "remaining_evaluations": remaining,
        "failed_attempts": failed_attempts,
        "permanent_failures": permanent_failures,
        "percent_complete": round((successful_evals / total_expected_evals) * 100.0, 2) if total_expected_evals > 0 else 0.0,
        "requested_cpu_workers": req_workers,
        "available_cpu_workers": os.cpu_count(),
        "actual_cpu_workers": actual_workers,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "evaluations_per_minute": round(rate_per_min, 2),
        "estimated_remaining_seconds": round(eta_seconds, 2),
        "last_updated": datetime.datetime.now().isoformat(),
        "status": status,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)


def run_partition(partition_id, max_cpu_workers=None, save_images="all",
                  max_retries=2, max_consecutive_failures=5, dry_run=False):
    """
    Executes the designated partition benchmark with persistent worker pool,
    resume validation, configurable retry limits, and clean pause draining.
    """
    output_dir = os.path.join(REPO_ROOT, "results", "final_benchmark", partition_id)
    os.makedirs(output_dir, exist_ok=True)

    raw_csv_path = os.path.join(output_dir, "raw_results.csv")
    error_csv_path = os.path.join(output_dir, "error_log.csv")
    manifest_path = os.path.join(output_dir, "benchmark_manifest.json")
    pause_file_path = os.path.join(output_dir, ".pause_requested")
    if os.path.exists(pause_file_path):
        try:
            os.remove(pause_file_path)
        except Exception:
            pass
    live_log_path = os.path.join(output_dir, "live_terminal.log")
    recon_dir = os.path.join(output_dir, "reconstructions")
    os.makedirs(recon_dir, exist_ok=True)
    pid_file_path = os.path.join(output_dir, ".runner_pid")

    try:
        with open(pid_file_path, "w", encoding="utf-8") as pf:
            pf.write(str(os.getpid()))
    except Exception:
        pass

    code_hash = app.CODE_HASH
    config_hash = compute_file_sha256(CONFIG_PATH)

    # Issue 1: Strictly enforce the 10-worker maximum: max(1, min(requested, 10, avail))
    req_workers, avail_workers, actual_workers = get_worker_configuration(max_cpu_workers)

    log_event("=" * 70, live_log_path)
    log_event(f"ASL-SR-DPT Final Benchmark: {partition_id.upper()} Distributed Runner", live_log_path)
    log_event("=" * 70, live_log_path)
    log_event(f"Requested CPU workers: {req_workers} (MAX_CPU_WORKERS hard limit 10 enforced)", live_log_path)
    log_event(f"Available system logical CPUs: {avail_workers}", live_log_path)
    log_event(f"Actual workers allocated: {actual_workers}", live_log_path)
    log_event(f"Code SHA-256 hash: {code_hash[:16]}...", live_log_path)
    log_event(f"Config SHA-256 hash: {config_hash[:16]}...", live_log_path)
    log_event(f"Max retries per task: {max_retries} | Max consecutive failures: {max_consecutive_failures}", live_log_path)

    # 1. Load Partition Conditions
    assigned_conditions = load_partition_conditions(partition_id)
    total_conditions = len(assigned_conditions)
    expected_evaluations = total_conditions * len(METHODS_ORDER)  # 3,400 x 3 = 10,200

    log_event(f"Assigned conditions: {total_conditions}", live_log_path)
    log_event(f"Expected solver evaluations: {expected_evaluations}", live_log_path)

    # 2. Initialize CSV headers if missing
    if not os.path.exists(raw_csv_path) or os.path.getsize(raw_csv_path) == 0:
        with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()

    if not os.path.exists(error_csv_path) or os.path.getsize(error_csv_path) == 0:
        with open(error_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["condition_id", "image_id", "noise_sigma", "trial", "method", "timestamp", "error_message"])

    # 3. Issue 3: Validate results before resuming (checks code_hash, config_hash, methods, metrics)
    successful_keys, incompatible_records, failed_records = load_and_audit_existing_results(
        raw_csv_path=raw_csv_path,
        expected_code_hash=code_hash,
        expected_config_hash=config_hash,
        expected_methods=METHODS_ORDER,
        clean_incompatible=True,
        live_log_path=live_log_path,
    )
    successful_count = len(successful_keys)

    # Issue 2: Load prior failure counts to prevent endless retries
    attempt_counts = load_prior_failure_counts(error_csv_path)
    failed_attempts_count = sum(attempt_counts.values())
    permanently_failed_keys = set()
    for k, cnt in attempt_counts.items():
        if cnt >= 1 + max_retries and k not in successful_keys:
            permanently_failed_keys.add(k)

    log_event(f"Already completed valid evaluations: {successful_count} / {expected_evaluations}", live_log_path)
    if incompatible_records:
        log_event(f"Rejected incompatible records quarantined: {len(incompatible_records)}", live_log_path)
    if failed_attempts_count > 0:
        log_event(f"Prior failed attempts logged: {failed_attempts_count}", live_log_path)
    if permanently_failed_keys:
        log_event(f"Permanently failed tasks (exhausted {1 + max_retries} attempts): {len(permanently_failed_keys)}", live_log_path)

    # Check for early completion
    if successful_count >= expected_evaluations:
        log_event(f"[COMPLETE] All {expected_evaluations} evaluations for {partition_id.upper()} are already finished.", live_log_path)
        update_pc_manifest(manifest_path, partition_id, successful_count, expected_evaluations,
                           req_workers, actual_workers, status="COMPLETED")
        return

    if (successful_count + len(permanently_failed_keys)) >= expected_evaluations:
        log_event(f"[HALTED] All remaining tasks have permanently failed ({len(permanently_failed_keys)}). Stopping.", live_log_path)
        update_pc_manifest(manifest_path, partition_id, successful_count, expected_evaluations,
                           req_workers, actual_workers, status="COMPLETED_WITH_FAILURES",
                           permanent_failures=len(permanently_failed_keys))
        return

    # 4. Build Evaluation Queue with only unresolved tasks
    unresolved_queue = []
    for cond in assigned_conditions:
        for m in METHODS_ORDER:
            k = (cond["image_id"], float(cond["noise_sigma"]), int(cond["trial"]), m)
            if k in successful_keys or k in permanently_failed_keys:
                continue

            if save_images == "all":
                is_save = True
            elif save_images == "trial1":
                is_save = (cond["trial"] == 1)
            elif save_images == "previews":
                is_save = (cond["image_id"] in ("test001", "test004", "test008") and cond["trial"] == 1)
            else:
                is_save = False

            unresolved_queue.append({
                "condition_id": cond["condition_id"],
                "partition_id": partition_id,
                "image_id": cond["image_id"],
                "noise_sigma": cond["noise_sigma"],
                "trial": cond["trial"],
                "sensing_seed": cond["sensing_seed"],
                "noise_seed": cond["noise_seed"],
                "method": m,
                "code_hash": code_hash,
                "config_hash": config_hash,
                "max_cpu_workers": req_workers,
                "actual_cpu_workers": actual_workers,
                "save_preview": is_save,
                "recon_dir": recon_dir,
            })

    log_event(f"Unresolved tasks in queue to execute: {len(unresolved_queue)}", live_log_path)

    if dry_run:
        log_event("[DRY RUN] Dry run requested. Exiting without solver execution.", live_log_path)
        update_pc_manifest(manifest_path, partition_id, successful_count, expected_evaluations,
                           req_workers, actual_workers, failed_attempts=failed_attempts_count,
                           permanent_failures=len(permanently_failed_keys))
        return

    update_pc_manifest(manifest_path, partition_id, successful_count, expected_evaluations,
                       req_workers, actual_workers, failed_attempts=failed_attempts_count,
                       permanent_failures=len(permanently_failed_keys))

    # 5. Issue 4: Reuse persistent process pool across all tasks
    start_time = time.perf_counter()
    session_completed = 0
    paused = False
    abort_requested = False
    consecutive_failures = 0

    pipeline_target = actual_workers * 2
    log_event(f"Initializing persistent ProcessPoolExecutor with {actual_workers} workers (pipeline capacity: {pipeline_target})...", live_log_path)

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
            in_flight = {}  # future -> task_spec

            def submit_next_task():
                nonlocal unresolved_queue
                if unresolved_queue and not paused and not abort_requested:
                    spec = unresolved_queue.pop(0)
                    fut = executor.submit(evaluate_single_task, spec)
                    in_flight[fut] = spec
                    return True
                return False

            # Pre-fill execution pipeline
            while len(in_flight) < pipeline_target and unresolved_queue:
                if not submit_next_task():
                    break

            # Main event loop: process completed tasks as they finish
            while in_flight:
                # Issue 6: Check pause sentinel. Stop submitting new tasks, drain in-flight tasks cleanly.
                if os.path.exists(pause_file_path) and not paused:
                    log_event(f"[PAUSE] Sentinel file .pause_requested detected. Draining {len(in_flight)} in-flight tasks...", live_log_path)
                    paused = True

                # Wait for next future(s) to finish
                done, _ = concurrent.futures.wait(in_flight.keys(), return_when=concurrent.futures.FIRST_COMPLETED)

                for future in done:
                    spec = in_flight.pop(future)
                    k = (spec["image_id"], float(spec["noise_sigma"]), int(spec["trial"]), spec["method"])

                    try:
                        record = future.result()
                    except Exception as exc:
                        err_str = f"{type(exc).__name__}: {str(exc)}"
                        log_event(f"[FATAL WORKER ERROR] Exception on {k}: {err_str}", live_log_path)
                        record = make_failed_record(spec, err_str, code_hash, config_hash, partition_id, req_workers, actual_workers)

                    status = record.get("solver_status", "FAILED")
                    # Issue 5: Progress accuracy - strictly increment successful_count only on SUCCESS
                    if status == "SUCCESS":
                        consecutive_failures = 0
                        successful_keys.add(k)
                        successful_count += 1
                        session_completed += 1

                        # Atomic append to raw_results.csv
                        with open(raw_csv_path, "a", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
                            writer.writerow(record)

                        elapsed = time.perf_counter() - start_time
                        rate = (session_completed / elapsed) * 60.0 if elapsed > 0 else 0.0
                        rem_tasks = max(0, expected_evaluations - successful_count - len(permanently_failed_keys))
                        eta_s = (rem_tasks / (rate / 60.0)) if rate > 0 else 0.0
                        eta_str = f"{eta_s / 3600.0:.1f}h" if eta_s >= 3600 else f"{eta_s / 60.0:.1f}m"

                        psnr_s = f"{float(record['psnr']):.2f} dB"
                        ssim_s = f"{float(record['ssim']):.4f}"
                        t_s = f"{float(record['runtime_seconds']):.2f}s"
                        pct = (successful_count / expected_evaluations) * 100.0

                        img_id = record["image_id"]
                        noise_s = int(float(record["noise_sigma"]))
                        tr = record["trial"]
                        meth = record["method"]

                        log_event(f"[{partition_id.upper()}] {img_id} | sigma={noise_s} | Trial {tr} | {meth} -> PSNR: {psnr_s}, SSIM: {ssim_s}, Time: {t_s} | Success: {successful_count}/{expected_evaluations} ({pct:.1f}%) | Pending: {rem_tasks} | Rate: {rate:.1f} evals/min | ETA: {eta_str}", live_log_path)

                    else:
                        # Issue 2: Prevent endless retries
                        failed_attempts_count += 1
                        consecutive_failures += 1
                        err_msg = record.get("error_message", "Unknown solver error")
                        log_event(f"[{partition_id.upper()} FAIL] {k} -> {err_msg}", live_log_path)

                        # Record failure in error_log.csv
                        with open(error_csv_path, "a", newline="", encoding="utf-8") as ef:
                            ewriter = csv.writer(ef)
                            ewriter.writerow([spec["condition_id"], spec["image_id"], spec["noise_sigma"], spec["trial"], spec["method"], datetime.datetime.now().isoformat(), err_msg])

                        attempt_counts[k] = attempt_counts.get(k, 0) + 1

                        if consecutive_failures >= max_consecutive_failures:
                            log_event(f"[ABORT] Exceeded maximum consecutive failures ({consecutive_failures} >= {max_consecutive_failures}). Halting benchmark execution.", live_log_path)
                            abort_requested = True
                        elif attempt_counts[k] <= 1 + max_retries:
                            log_event(f"[RETRY {attempt_counts[k]-1}/{max_retries}] Re-queueing {k} for retry (attempt {attempt_counts[k]}/{1+max_retries}).", live_log_path)
                            unresolved_queue.append(spec)
                        else:
                            log_event(f"[PERMANENT FAILURE] {k} failed all {attempt_counts[k]} attempts. Exhausted retry limit.", live_log_path)
                            permanently_failed_keys.add(k)

                    # Update manifest periodically or when drained
                    if session_completed % 10 == 0 or len(in_flight) == 0:
                        elapsed = time.perf_counter() - start_time
                        rate = (session_completed / elapsed) * 60.0 if elapsed > 0 else 0.0
                        rem_tasks = max(0, expected_evaluations - successful_count - len(permanently_failed_keys))
                        eta_s = (rem_tasks / (rate / 60.0)) if rate > 0 else 0.0
                        update_pc_manifest(manifest_path, partition_id, successful_count, expected_evaluations,
                                           req_workers, actual_workers, elapsed, rate, eta_s,
                                           failed_attempts=failed_attempts_count,
                                           permanent_failures=len(permanently_failed_keys))

                    # Replenish pipeline if not paused and not aborting
                    while len(in_flight) < pipeline_target and unresolved_queue and not paused and not abort_requested:
                        if not submit_next_task():
                            break

    except Exception as pool_exc:
        log_event(f"[FATAL POOL ERROR] Exception in process worker execution: {pool_exc}", live_log_path)

    total_elapsed = time.perf_counter() - start_time
    final_rate = (session_completed / total_elapsed) * 60.0 if total_elapsed > 0 else 0.0
    status_str = "PAUSED" if paused else ("ABORTED" if abort_requested else ("COMPLETED" if successful_count >= expected_evaluations else "INCOMPLETE"))

    update_pc_manifest(manifest_path, partition_id, successful_count, expected_evaluations,
                       req_workers, actual_workers, total_elapsed, final_rate, 0.0,
                       failed_attempts=failed_attempts_count,
                       permanent_failures=len(permanently_failed_keys),
                       status=status_str)

    log_event("", live_log_path)
    log_event(f"[{partition_id.upper()} Execution Summary]", live_log_path)
    log_event(f"  Valid Successful Evaluations: {successful_count} / {expected_evaluations}", live_log_path)
    log_event(f"  Failed Attempts Recorded: {failed_attempts_count}", live_log_path)
    log_event(f"  Permanently Failed Tasks: {len(permanently_failed_keys)}", live_log_path)
    log_event(f"  Pending Tasks: {max(0, expected_evaluations - successful_count - len(permanently_failed_keys))}", live_log_path)
    log_event(f"  Session Evaluations: {session_completed}", live_log_path)
    log_event(f"  Elapsed Runtime: {total_elapsed:.2f} seconds", live_log_path)
    log_event(f"  Final Status: {status_str}", live_log_path)

    try:
        if os.path.exists(pid_file_path):
            os.remove(pid_file_path)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="ASL-SR-DPT Distributed PC Final Benchmark Runner")
    parser.add_argument("--partition-id", choices=["pc1", "pc2", "pc3"], required=True,
                        help="Designated partition identifier for this machine (pc1, pc2, or pc3)")
    parser.add_argument("--max-cpu-workers", type=int, default=10,
                        help="Maximum CPU workers to allocate (default: 10, limit: 10)")
    parser.add_argument("--save-images", choices=["all", "trial1", "previews", "none"], default="all",
                        help="Image saving policy: 'all' (all 10,200 images, default), 'trial1' (saves all images for Trial 1), 'previews', 'none'")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Maximum retry attempts per failed evaluation before marking permanent failure (default: 2)")
    parser.add_argument("--max-consecutive-failures", type=int, default=5,
                        help="Maximum consecutive failures before aborting execution (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify partition queue without running solvers")
    args = parser.parse_args()

    run_partition(partition_id=args.partition_id, max_cpu_workers=args.max_cpu_workers,
                  save_images=args.save_images, max_retries=args.max_retries,
                  max_consecutive_failures=args.max_consecutive_failures, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
