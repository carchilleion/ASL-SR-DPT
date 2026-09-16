"""
Deep Profiling and Performance Instrumentation Engine for ASL-SR-DPT.
Implements Parts 1 and 2 of the Deep Optimization Loop.

Measures:
1. total solver time
2. time per iteration
3. objective time
4. gradient time
5. line-search time
6. midpoint time
7. support-mask time
8. residual computation time
9. allocation count / memory profile via tracemalloc
10. Python function call counts via cProfile
11. NumPy matrix operations
12. exponential evaluation count
13. active coefficient count
14. iteration count

Outputs:
- results/optimization/deep_baseline.csv
- results/optimization/deep_profile/cprofile_stats.txt
- results/optimization/deep_profile/solver.prof
- results/optimization/deep_profile/tracemalloc_stats.txt
- results/optimization/deep_profile/deep_profile_summary.json
- results/optimization/deep_profile/deep_profile_report.md
"""

import os
import sys

# Step 1: Enforce strict single-threaded execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import json
import cProfile
import pstats
import io
import tracemalloc
import hashlib
import csv
import numpy as np
from skimage.metrics import structural_similarity as compute_skimage_ssim

# Local module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from sensing import generate_random_sensing, add_awgn
from reconstruction import extract_patches, dct_patch, idct_patch
from dataset import load_bsd68
from hybrid_sparse_solver_v7_optimized import HybridSparseSolverV7Optimized
from hybrid_sparse_solver_v7_fixed import HybridSparseSolverV7


def compute_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_evaluation_data(num_patches=500):
    config_path = os.path.join(current_dir, "..", "configs", "final_config.json")
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    data_dir = os.path.join(current_dir, "..", "data", "BSD68")
    images = load_bsd68(data_dir)
    test_img = images["test001"]["image"]
    H, W = test_img.shape

    base_seed = cfg.get("base_seed", 20260908)
    M, N = 38, 64
    noise_sigma = 15.0 / 255.0

    A_std = generate_random_sensing(M, N, seed=base_seed)
    noisy_img = add_awgn(test_img, sigma_noise=15.0, seed=base_seed)

    clean_records = extract_patches(test_img, patch_size=8, stride=2)[:num_patches]
    noisy_records = extract_patches(noisy_img, patch_size=8, stride=2)[:num_patches]

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
        "M": M,
        "N": N,
    }


def run_deep_profiling(solver_cls=HybridSparseSolverV7Optimized, solver_label="V7_BEST_CURRENT", solver_filepath=None, num_patches=500, num_timing_runs=10):
    out_dir = os.path.join(current_dir, "..", "results", "optimization", "deep_profile")
    os.makedirs(out_dir, exist_ok=True)
    res_opt_dir = os.path.join(current_dir, "..", "results", "optimization")
    os.makedirs(res_opt_dir, exist_ok=True)

    if solver_filepath is None:
        solver_filepath = os.path.join(current_dir, "hybrid_sparse_solver_v7_optimized.py")
    code_hash = compute_file_hash(solver_filepath)

    data = load_evaluation_data(num_patches=num_patches)
    A = data["A"]
    measurements = data["measurements"]
    clean_patches = data["clean_patches"]
    clean_thetas = data["clean_thetas"]
    cfg = data["cfg"]

    solver = solver_cls(A, lambda_reg=0.1, tol=1e-5)

    print(f"\n=======================================================")
    print(f"DEEP PROFILING ENGINE: {solver_label}")
    print(f"File: {os.path.basename(solver_filepath)} | SHA-256: {code_hash[:12]}...")
    print(f"Patches: {num_patches} | Noise sigma: 15.0/255 | M={data['M']}, N={data['N']}")
    print(f"=======================================================\n")

    # -------------------------------------------------------------
    # 1. TRACEMALLOC PROFILING
    # -------------------------------------------------------------
    print("Running memory allocation profiling (tracemalloc)...")
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    sample_z = []
    sample_diag = []
    for i in range(min(100, num_patches)):
        z, diag = solver.denoise_patch(measurements[i], return_diagnostics=True, detailed_profile=False)
        sample_z.append(z)
        sample_diag.append(diag)

    snap_after = tracemalloc.take_snapshot()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    top_stats = snap_after.compare_to(snap_before, "lineno")
    tracemalloc_file = os.path.join(out_dir, "tracemalloc_stats.txt")
    with open(tracemalloc_file, "w", encoding="utf-8") as f:
        f.write(f"Tracemalloc Memory Profile for {solver_label}\n")
        f.write(f"Current Memory: {current_mem / 1024:.2f} KiB\n")
        f.write(f"Peak Memory:    {peak_mem / 1024:.2f} KiB\n\n")
        f.write("Top 20 Memory Allocation Differences:\n")
        for stat in top_stats[:20]:
            f.write(f"{stat}\n")
    print(f"Tracemalloc saved to: {tracemalloc_file} (Peak: {peak_mem / 1024:.1f} KiB)")

    # -------------------------------------------------------------
    # 2. CPROFILE & PSTATS PROFILING
    # -------------------------------------------------------------
    print("\nRunning call-graph & execution profiling (cProfile)...")
    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(num_patches):
        solver.denoise_patch(measurements[i], return_diagnostics=False, detailed_profile=False)

    profiler.disable()
    prof_binary = os.path.join(out_dir, "solver.prof")
    profiler.dump_stats(prof_binary)

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(35)
    cprofile_file = os.path.join(out_dir, "cprofile_stats.txt")
    with open(cprofile_file, "w", encoding="utf-8") as f:
        f.write(s.getvalue())
    print(f"cProfile saved to: {cprofile_file}")

    # -------------------------------------------------------------
    # 3. FINE-GRAINED INSTRUMENTED RUN (14 METRICS)
    # -------------------------------------------------------------
    print("\nRunning fine-grained telemetry pass across 500 patches...")
    all_z = []
    all_diag = []
    obj_times = []
    grad_times = []
    cand_times = []
    mid_times = []
    mask_times = []
    residual_times = []

    total_iters = 0
    total_accepted = 0
    total_matvec = 0
    total_exp = 0
    total_active_coords = 0
    total_backtracks = 0
    residuals = []
    final_objs = []
    active_ratios = []

    for i in range(num_patches):
        z, diag = solver.denoise_patch(measurements[i], return_diagnostics=True, detailed_profile=True)
        all_z.append(z)
        all_diag.append(diag)

        tb = diag.get("time_breakdown", {})
        op = diag.get("operation_counts", {})

        obj_times.append(tb.get("solver_objective_evaluation", 0.0))
        grad_times.append(tb.get("solver_gradient_evaluation", 0.0))
        cand_times.append(tb.get("solver_armijo_candidate_checks", 0.0))
        mid_times.append(tb.get("solver_midpoint_evaluation", 0.0))
        mask_times.append(tb.get("solver_support_mask_calculation", 0.0))
        residual_times.append(tb.get("solver_final_residual_calc", 0.0))

        total_iters += diag.get("iterations", 0)
        total_accepted += diag.get("accepted_steps", 0)
        total_matvec += op.get("matrix_vector_multiplications", 0)
        total_exp += op.get("exponential_evaluations", 0)
        total_backtracks += op.get("backtracking_attempts", 0)

        residuals.append(diag.get("final_residual", 0.0))
        final_objs.append(diag["objective"][-1] if diag["objective"] else 0.0)
        active_ratios.append(diag.get("active_support_ratio", 1.0))
        total_active_coords += sum(diag.get("active_counts", [64]))

    # Quality metrics across 500 patches
    patch_mses = []
    patch_psnrs = []
    patch_ssims = []
    for i in range(num_patches):
        p_clean = clean_patches[i]
        p_rec = idct_patch(all_z[i].reshape(8, 8))
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

    # -------------------------------------------------------------
    # 4. STATISTICAL TIMING BENCHMARK (10 REPEATED RUNS)
    # -------------------------------------------------------------
    print(f"\nRunning {num_timing_runs} repeated timing trials on 500 patches (clean benchmark mode)...")
    run_times = []
    batch_size = 50 if hasattr(solver, "denoise_batch") else None
    for trial in range(num_timing_runs):
        t0 = time.perf_counter()
        if batch_size is not None:
            for start in range(0, num_patches, batch_size):
                solver.denoise_batch(measurements[start : start + batch_size].T)
        else:
            for i in range(num_patches):
                solver.denoise_patch(measurements[i], return_diagnostics=False, detailed_profile=False)
        t_elapsed = time.perf_counter() - t0
        run_times.append(t_elapsed)
        print(f"  Trial {trial + 1:2d}: {t_elapsed:.4f} s ({(t_elapsed / num_patches) * 1000.0:.3f} ms/patch)")

    mean_time_total = float(np.mean(run_times))
    median_time_total = float(np.median(run_times))
    std_time_total = float(np.std(run_times))
    min_time_total = float(np.min(run_times))
    max_time_total = float(np.max(run_times))

    mean_ms_patch = (mean_time_total / num_patches) * 1000.0
    median_ms_patch = (median_time_total / num_patches) * 1000.0
    std_ms_patch = (std_time_total / num_patches) * 1000.0
    min_ms_patch = (min_time_total / num_patches) * 1000.0
    max_ms_patch = (max_time_total / num_patches) * 1000.0

    mean_time_per_iter = mean_ms_patch / (total_iters / num_patches)

    # Aggregated breakdown per patch in ms
    ms_obj = (np.sum(obj_times) / num_patches) * 1000.0
    ms_grad = (np.sum(grad_times) / num_patches) * 1000.0
    ms_cand = (np.sum(cand_times) / num_patches) * 1000.0
    ms_mid = (np.sum(mid_times) / num_patches) * 1000.0
    ms_mask = (np.sum(mask_times) / num_patches) * 1000.0
    ms_res = (np.sum(residual_times) / num_patches) * 1000.0

    summary_metrics = {
        "solver_label": solver_label,
        "solver_file": os.path.basename(solver_filepath),
        "code_hash": code_hash,
        "num_patches": num_patches,
        "num_timing_trials": num_timing_runs,
        "timing_stats": {
            "mean_total_seconds": mean_time_total,
            "median_total_seconds": median_time_total,
            "std_total_seconds": std_time_total,
            "min_total_seconds": min_time_total,
            "max_total_seconds": max_time_total,
            "mean_ms_patch": mean_ms_patch,
            "median_ms_patch": median_ms_patch,
            "std_ms_patch": std_ms_patch,
            "min_ms_patch": min_ms_patch,
            "max_ms_patch": max_ms_patch,
            "time_per_iteration_ms": mean_time_per_iter,
        },
        "time_breakdown_ms_per_patch": {
            "objective_time_ms": ms_obj,
            "gradient_time_ms": ms_grad,
            "line_search_candidate_time_ms": ms_cand,
            "midpoint_time_ms": ms_mid,
            "support_mask_time_ms": ms_mask,
            "residual_computation_time_ms": ms_res,
        },
        "operation_counts_500_patches": {
            "total_iterations": total_iters,
            "mean_iterations_per_patch": total_iters / num_patches,
            "total_accepted_steps": total_accepted,
            "total_backtracks": total_backtracks,
            "numpy_matvec_ops": total_matvec,
            "exponential_evaluations": total_exp,
            "total_active_coordinates": total_active_coords,
            "mean_active_ratio": float(np.mean(active_ratios)),
        },
        "memory_profile": {
            "peak_memory_kib": float(peak_mem / 1024),
            "current_memory_kib": float(current_mem / 1024),
        },
        "numerical_quality": {
            "mean_objective": float(np.mean(final_objs)),
            "mean_residual": float(np.mean(residuals)),
            "mean_patch_mse": mean_patch_mse,
            "mean_patch_psnr_db": mean_patch_psnr,
            "overall_psnr_db": overall_psnr,
            "mean_patch_ssim": mean_patch_ssim,
        }
    }

    # Save summary JSON
    summary_json_path = os.path.join(out_dir, "deep_profile_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    # Save Markdown report
    report_md_path = os.path.join(out_dir, "deep_profile_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Deep Profile Dashboard: {solver_label}\n\n")
        f.write(f"- **Source File:** `{os.path.basename(solver_filepath)}`\n")
        f.write(f"- **Code SHA-256:** `{code_hash}`\n")
        f.write(f"- **Sample Size:** {num_patches} patches (`test001.png`, $\\sigma=15/255$)\n")
        f.write(f"- **Timing Trials:** {num_timing_runs} independent runs\n\n")
        f.write("## 1. Timing Summary\n\n")
        f.write(f"| Metric | Total (s) | Per Patch (ms) |\n")
        f.write(f"| :--- | :---: | :---: |\n")
        f.write(f"| **Mean** | {mean_time_total:.4f} s | **{mean_ms_patch:.3f} ms** |\n")
        f.write(f"| **Median** | {median_time_total:.4f} s | {median_ms_patch:.3f} ms |\n")
        f.write(f"| **Std Dev** | {std_time_total:.4f} s | {std_ms_patch:.3f} ms |\n")
        f.write(f"| **Min** | {min_time_total:.4f} s | {min_ms_patch:.3f} ms |\n")
        f.write(f"| **Max** | {max_time_total:.4f} s | {max_ms_patch:.3f} ms |\n\n")
        f.write("## 2. Fine-Grained Inner-Loop Breakdown (14 Metrics)\n\n")
        f.write(f"1. **Total Solver Time:** {mean_time_total:.4f} s ({mean_ms_patch:.3f} ms/patch)\n")
        f.write(f"2. **Time Per Iteration:** {mean_time_per_iter:.4f} ms\n")
        f.write(f"3. **Objective Evaluation Time:** {ms_obj:.3f} ms/patch\n")
        f.write(f"4. **Gradient Evaluation Time:** {ms_grad:.3f} ms/patch\n")
        f.write(f"5. **Line Search Time:** {ms_cand:.3f} ms/patch\n")
        f.write(f"6. **Midpoint Evaluation Time:** {ms_mid:.3f} ms/patch\n")
        f.write(f"7. **Support Mask Time:** {ms_mask:.3f} ms/patch\n")
        f.write(f"8. **Residual Computation Time:** {ms_res:.3f} ms/patch\n")
        f.write(f"9. **Peak Heap Memory:** {peak_mem / 1024:.2f} KiB\n")
        f.write(f"10. **Iterations Per Patch:** {total_iters / num_patches:.2f}\n")
        f.write(f"11. **NumPy Mat-Vec Products:** {total_matvec} total\n")
        f.write(f"12. **Exponential Evaluations:** {total_exp} total\n")
        f.write(f"13. **Active Coordinate Ratio:** {summary_metrics['operation_counts_500_patches']['mean_active_ratio']:.4f}\n")
        f.write(f"14. **Accepted Steps:** {total_accepted} / {total_iters} ({100.0 * total_accepted / total_iters:.1f}%)\n\n")
        f.write("## 3. Reconstruction Quality Baseline\n\n")
        f.write(f"- **Mean Residual:** {np.mean(residuals):.6f}\n")
        f.write(f"- **Mean Objective:** {np.mean(final_objs):.6f}\n")
        f.write(f"- **Patch PSNR:** {mean_patch_psnr:.2f} dB (Overall: {overall_psnr:.2f} dB)\n")
        f.write(f"- **Patch SSIM:** {mean_patch_ssim:.4f}\n")
        f.write(f"- **Patch MSE:** {mean_patch_mse:.6e}\n")

    # Save deep_baseline.csv for Part 1
    deep_baseline_csv = os.path.join(res_opt_dir, "deep_baseline.csv")
    with open(deep_baseline_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "version",
            "solver_file",
            "code_hash",
            "runtime_seconds",
            "mean_ms_patch",
            "median_ms_patch",
            "std_ms_patch",
            "iterations",
            "objective",
            "residual",
            "active_ratio",
            "psnr",
            "ssim",
            "mse",
            "peak_memory_kib"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "version": solver_label,
            "solver_file": os.path.basename(solver_filepath),
            "code_hash": code_hash,
            "runtime_seconds": f"{mean_time_total:.4f}",
            "mean_ms_patch": f"{mean_ms_patch:.3f}",
            "median_ms_patch": f"{median_ms_patch:.3f}",
            "std_ms_patch": f"{std_ms_patch:.3f}",
            "iterations": f"{total_iters / num_patches:.2f}",
            "objective": f"{np.mean(final_objs):.6f}",
            "residual": f"{np.mean(residuals):.6f}",
            "active_ratio": f"{summary_metrics['operation_counts_500_patches']['mean_active_ratio']:.4f}",
            "psnr": f"{overall_psnr:.2f}",
            "ssim": f"{mean_patch_ssim:.4f}",
            "mse": f"{mean_patch_mse:.6e}",
            "peak_memory_kib": f"{peak_mem / 1024:.2f}",
        })

    print(f"\nSuccessfully generated:")
    print(f"  -> {deep_baseline_csv}")
    print(f"  -> {summary_json_path}")
    print(f"  -> {report_md_path}")

    return summary_metrics


if __name__ == "__main__":
    run_deep_profiling()
