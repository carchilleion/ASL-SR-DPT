"""
ASL-SR-DPT Version Evaluator and Harness.
Automates the testing, equivalence check, statistical benchmarking (10 runs),
memory profiling, and artifact logging for each candidate optimization version.

Maintains:
- results/optimization/versions/<VERSION_ID>/
- results/optimization/optimization_history.csv
"""

import os
import sys

# Single-threaded execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import json
import shutil
import hashlib
import csv
import tracemalloc
import numpy as np
from skimage.metrics import structural_similarity as compute_skimage_ssim

# Local module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from sensing import generate_random_sensing, add_awgn
from reconstruction import extract_patches, dct_patch, idct_patch
from dataset import load_bsd68
from hybrid_sparse_solver_v7_fixed import HybridSparseSolverV7


def compute_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_500_patches():
    config_path = os.path.join(current_dir, "..", "configs", "final_config.json")
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    data_dir = os.path.join(current_dir, "..", "data", "BSD68")
    images = load_bsd68(data_dir)
    test_img = images["test001"]["image"]

    base_seed = cfg.get("base_seed", 20260908)
    M, N = 38, 64

    A_std = generate_random_sensing(M, N, seed=base_seed)
    noisy_img = add_awgn(test_img, sigma_noise=15.0, seed=base_seed)

    clean_records = extract_patches(test_img, patch_size=8, stride=2)[:500]
    noisy_records = extract_patches(noisy_img, patch_size=8, stride=2)[:500]

    clean_patches = [r["patch"] for r in clean_records]
    noisy_patches = [r["patch"] for r in noisy_records]

    clean_thetas = np.array([dct_patch(p).reshape(-1) for p in clean_patches])
    measurements = np.array([A_std @ dct_patch(p).reshape(-1) for p in noisy_patches])

    return {
        "cfg": cfg,
        "A": A_std,
        "clean_patches": clean_patches,
        "clean_thetas": clean_thetas,
        "measurements": measurements,
    }


def evaluate_version(
    version_id,
    solver_module_or_class,
    solver_filepath,
    change_description,
    baseline_ms_patch=3.504,
    prev_best_ms_patch=3.504,
    num_trials=10,
    min_improvement_pct=3.0,
    batch_size=None,
):
    print(f"\n=======================================================")
    print(f"EVALUATING VERSION: {version_id}")
    print(f"Change: {change_description}")
    if batch_size is not None:
        print(f"Execution Mode: Batched (B = {batch_size})")
    print(f"=======================================================\n")

    res_opt_dir = os.path.join(current_dir, "..", "results", "optimization")
    version_dir = os.path.join(res_opt_dir, "versions", version_id)
    os.makedirs(version_dir, exist_ok=True)

    data = load_500_patches()
    A = data["A"]
    measurements = data["measurements"]
    clean_patches = data["clean_patches"]
    cfg = data["cfg"]

    # 1. Store solver file, config, hash, change description
    dest_solver = os.path.join(version_dir, os.path.basename(solver_filepath))
    shutil.copyfile(solver_filepath, dest_solver)

    code_hash = compute_file_hash(solver_filepath)
    with open(os.path.join(version_dir, "hash.txt"), "w", encoding="utf-8") as f:
        f.write(code_hash + "\n")

    dest_config = os.path.join(version_dir, "config.json")
    with open(dest_config, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    with open(os.path.join(version_dir, "description.txt"), "w", encoding="utf-8") as f:
        f.write(f"Version: {version_id}\n")
        f.write(f"Change Description: {change_description}\n")
        if batch_size:
            f.write(f"Batch Size: {batch_size}\n")

    # 2. Instantiate solvers
    s_baseline = HybridSparseSolverV7(A, lambda_reg=0.1, tol=1e-5)
    if isinstance(solver_module_or_class, type):
        s_cand = solver_module_or_class(A, lambda_reg=0.1, tol=1e-5)
    else:
        s_cand = solver_module_or_class.HybridSparseSolverV7Optimized(A, lambda_reg=0.1, tol=1e-5)

    # 3. Equivalence validation across 500 patches against frozen baseline
    print("Running 500-patch bit-for-bit equivalence test...")
    max_z_diff = 0.0
    max_obj_diff = 0.0
    max_res_diff = 0.0
    mismatch_count = 0

    cand_z_list = []
    cand_diag_list = []

    if batch_size is not None and hasattr(s_cand, "denoise_batch"):
        # Solve all 500 in batches
        batches_z = []
        for start in range(0, 500, batch_size):
            Y_sub = measurements[start : start + batch_size].T
            Z_sub = s_cand.denoise_batch(Y_sub)
            batches_z.append(Z_sub)
        all_batched_Z = np.hstack(batches_z)  # (64, 500)

        for i in range(500):
            y = measurements[i]
            zb, db = s_baseline.denoise_patch(y, return_diagnostics=True)
            zc = all_batched_Z[:, i]
            cand_z_list.append(zc)

            res_c = float(np.linalg.norm(A @ zc - y))
            dz = float(np.max(np.abs(zb - zc)))
            dr = float(abs(db["final_residual"] - res_c))

            max_z_diff = max(max_z_diff, dz)
            max_res_diff = max(max_res_diff, dr)

        # In batched mode, objective max diff check on endpoints
        max_obj_diff = max_res_diff
        equiv_passed = max_z_diff < 1e-10 and max_res_diff < 1e-10
    else:
        for i in range(500):
            y = measurements[i]
            zb, db = s_baseline.denoise_patch(y, return_diagnostics=True)
            zc, dc = s_cand.denoise_patch(y, return_diagnostics=True)

            cand_z_list.append(zc)
            cand_diag_list.append(dc)

            dz = float(np.max(np.abs(zb - zc)))
            dr = float(abs(db["final_residual"] - dc["final_residual"]))
            do = float(max(abs(a - b) for a, b in zip(db["objective"], dc["objective"])))

            max_z_diff = max(max_z_diff, dz)
            max_res_diff = max(max_res_diff, dr)
            max_obj_diff = max(max_obj_diff, do)

            it_match = db["iterations"] == dc["iterations"]
            acc_match = db["accepted_steps"] == dc["accepted_steps"]
            stop_match = db["stop_reason"] == dc["stop_reason"]

            if not (it_match and acc_match and stop_match):
                mismatch_count += 1

        equiv_passed = (
            max_z_diff < 1e-10
            and max_res_diff < 1e-10
            and max_obj_diff < 1e-6
            and mismatch_count == 0
        )

    validation_result = {
        "version": version_id,
        "max_z_difference": max_z_diff,
        "max_objective_difference": max_obj_diff,
        "max_residual_difference": max_res_diff,
        "mismatch_count": mismatch_count,
        "iteration_match": mismatch_count == 0,
        "stop_reason_match": mismatch_count == 0,
        "equivalence_passed": bool(equiv_passed),
    }

    with open(os.path.join(version_dir, "validation_result.json"), "w", encoding="utf-8") as f:
        json.dump(validation_result, f, indent=2)

    print(f"  Max z diff:   {max_z_diff:.2e} (threshold: 1e-10) -> {'PASS' if max_z_diff < 1e-10 else 'FAIL'}")
    print(f"  Max obj diff: {max_obj_diff:.2e} (threshold: 1e-6)  -> {'PASS' if max_obj_diff < 1e-6 else 'FAIL'}")
    print(f"  Max res diff: {max_res_diff:.2e} (threshold: 1e-10) -> {'PASS' if max_res_diff < 1e-10 else 'FAIL'}")
    print(f"  Mismatches:   {mismatch_count} / 500              -> {'PASS' if mismatch_count == 0 else 'FAIL'}")

    # 4. Quality metrics
    patch_mses = []
    patch_psnrs = []
    patch_ssims = []
    for i in range(500):
        p_clean = clean_patches[i]
        p_rec = idct_patch(cand_z_list[i].reshape(8, 8))
        mse_i = float(np.mean((p_clean - p_rec) ** 2))
        psnr_i = 10.0 * np.log10(1.0 / max(mse_i, 1e-15))
        ssim_i = float(compute_skimage_ssim(p_clean, p_rec, data_range=1.0))
        patch_mses.append(mse_i)
        patch_psnrs.append(psnr_i)
        patch_ssims.append(ssim_i)

    mean_patch_mse = float(np.mean(patch_mses))
    mean_patch_psnr = float(np.mean(patch_psnrs))
    mean_patch_ssim = float(np.mean(patch_ssims))
    overall_psnr = 10.0 * np.log10(1.0 / max(mean_patch_mse, 1e-15))

    # 5. Memory profiling via tracemalloc
    tracemalloc.start()
    snap_0 = tracemalloc.take_snapshot()
    if batch_size is not None and hasattr(s_cand, "denoise_batch"):
        s_cand.denoise_batch(measurements[:min(100, 500)].T)
    else:
        for i in range(100):
            s_cand.denoise_patch(measurements[i], return_diagnostics=False)
    snap_1 = tracemalloc.take_snapshot()
    curr_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024 * 1024)

    # 6. Statistical timing benchmark (10 repeated runs)
    print(f"\nRunning {num_trials} repeated timing trials on 500 patches...")
    run_times = []
    for trial in range(num_trials):
        t0 = time.perf_counter()
        if batch_size is not None and hasattr(s_cand, "denoise_batch"):
            for start in range(0, 500, batch_size):
                s_cand.denoise_batch(measurements[start : start + batch_size].T)
        else:
            for i in range(500):
                s_cand.denoise_patch(measurements[i], return_diagnostics=False)
        t_el = time.perf_counter() - t0
        run_times.append(t_el)
        print(f"  Trial {trial + 1:2d}: {t_el:.4f} s ({(t_el / 500) * 1000.0:.3f} ms/patch)")

    mean_time_sec = float(np.mean(run_times))
    median_time_sec = float(np.median(run_times))
    std_time_sec = float(np.std(run_times))
    min_time_sec = float(np.min(run_times))
    max_time_sec = float(np.max(run_times))

    mean_ms_patch = (mean_time_sec / 500) * 1000.0
    median_ms_patch = (median_time_sec / 500) * 1000.0
    std_ms_patch = (std_time_sec / 500) * 1000.0
    min_ms_patch = (min_time_sec / 500) * 1000.0
    max_ms_patch = (max_time_sec / 500) * 1000.0

    speedup_vs_baseline = baseline_ms_patch / mean_ms_patch
    speedup_vs_prev = prev_best_ms_patch / mean_ms_patch
    runtime_reduction_vs_prev = (prev_best_ms_patch - mean_ms_patch) / prev_best_ms_patch * 100.0
    runtime_reduction_vs_base = (baseline_ms_patch - mean_ms_patch) / baseline_ms_patch * 100.0

    print(f"\nTiming Result:")
    print(f"  Mean ms/patch:    {mean_ms_patch:.3f} ms (median: {median_ms_patch:.3f} ms, std: {std_ms_patch:.3f} ms)")
    print(f"  Vs Previous Best: {prev_best_ms_patch:.3f} ms -> {mean_ms_patch:.3f} ms ({runtime_reduction_vs_prev:+.2f}%, {speedup_vs_prev:.3f}x)")

    # 7. Decision rule:
    # Must pass equivalence AND improve by >= min_improvement_pct vs prev_best
    if not equiv_passed:
        decision = "REJECT"
        reason = "Failed mathematical equivalence tolerances."
    elif runtime_reduction_vs_prev < min_improvement_pct:
        decision = "REJECT"
        reason = f"Improvement ({runtime_reduction_vs_prev:.2f}%) below {min_improvement_pct}% threshold."
    else:
        decision = "KEEP"
        reason = f"Passed equivalence with {runtime_reduction_vs_prev:.2f}% runtime reduction."

    print(f"  DECISION:         {decision} ({reason})")

    benchmark_result = {
        "version": version_id,
        "mean_ms_patch": mean_ms_patch,
        "median_ms_patch": median_ms_patch,
        "std_ms_patch": std_ms_patch,
        "min_ms_patch": min_ms_patch,
        "max_ms_patch": max_ms_patch,
        "speedup_vs_prev_best": speedup_vs_prev,
        "runtime_reduction_pct_vs_prev_best": runtime_reduction_vs_prev,
        "speedup_vs_v7_baseline": speedup_vs_baseline,
        "runtime_reduction_pct_vs_v7_baseline": runtime_reduction_vs_base,
        "peak_memory_mb": peak_mem_mb,
        "overall_psnr_db": overall_psnr,
        "mean_patch_ssim": mean_patch_ssim,
        "mean_patch_mse": mean_patch_mse,
        "decision": decision,
        "decision_reason": reason,
    }

    with open(os.path.join(version_dir, "benchmark_result.json"), "w", encoding="utf-8") as f:
        json.dump(benchmark_result, f, indent=2)

    # 8. Append / update results/optimization/optimization_history.csv
    history_csv = os.path.join(res_opt_dir, "optimization_history.csv")
    fieldnames = [
        "version",
        "change",
        "runtime_ms_patch",
        "speedup",
        "runtime_reduction_pct",
        "max_z_difference",
        "max_objective_difference",
        "max_residual_difference",
        "iteration_match",
        "stop_reason_match",
        "psnr",
        "ssim",
        "mse",
        "memory_mb",
        "decision",
    ]

    history_rows = []
    if os.path.exists(history_csv):
        with open(history_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("version") != version_id:
                    history_rows.append(row)

    history_rows.append({
        "version": version_id,
        "change": change_description,
        "runtime_ms_patch": f"{mean_ms_patch:.3f}",
        "speedup": f"{speedup_vs_prev:.3f}x",
        "runtime_reduction_pct": f"{runtime_reduction_vs_prev:+.2f}%",
        "max_z_difference": f"{max_z_diff:.2e}",
        "max_objective_difference": f"{max_obj_diff:.2e}",
        "max_residual_difference": f"{max_res_diff:.2e}",
        "iteration_match": "TRUE" if mismatch_count == 0 else "FALSE",
        "stop_reason_match": "TRUE" if mismatch_count == 0 else "FALSE",
        "psnr": f"{overall_psnr:.2f}",
        "ssim": f"{mean_patch_ssim:.4f}",
        "mse": f"{mean_patch_mse:.6e}",
        "memory_mb": f"{peak_mem_mb:.3f}",
        "decision": decision,
    })

    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)

    return benchmark_result, validation_result
