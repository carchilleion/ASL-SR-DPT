"""
ASL-SR-DPT Benchmark Runner
Main command-line driver executing experiments E1 to E8 on the BSD68 dataset.
Features:
  - Strict matched comparison pipeline across ASL-SR-DPT, OMP, and LASSO-ADMM.
  - One sensing matrix per trial reused across patches and images.
  - Exact separation of setup_time vs iterative solve_time (time.perf_counter).
  - Deterministic random seed policy.
  - Resume support with robust composite key checkpointing.
  - Isolated pilot mode with duration extrapolation.
  - Raw CSV logging and automated summary generation.
"""

import os
import sys

# Step 3 of Testing Manual: Enforce single-threaded execution for strict comparability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import json
import uuid
import argparse
import csv
import numpy as np
from PIL import Image

# Import local modules
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
from reconstruction import extract_patches, reconstruct_image, dct_patch, idct_patch
from dataset import load_bsd68
from metrics import compute_all_metrics
from validation import run_e1_verification
from summary import generate_summary

CODE_VERSION = "v7.0.0-fixed"

RAW_CSV_FIELDS = [
    "trial",
    "image_id",
    "noise_sigma",
    "sensing_mode",
    "solver",
    "psnr",
    "ssim",
    "mse",
    "solve_time",
    "setup_time",
    "iterations",
    "mean_iterations_per_patch",
    "mean_active_support_count",
    "final_sigma",
    "residual",
    "active_support_ratio",
    "failed_line_searches",
    "accepted_steps",
    "seed",
    "code_version",
    "patch_count",
    "run_id",
    "experiment",
]


def load_config(config_path="configs/final_config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_image_grayscale(img_arr, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clipped = np.clip(img_arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(clipped, mode="L").save(output_path)


def get_completed_keys(csv_path):
    """
    Read completed observation keys from existing raw CSV.
    Uses full 8-element composite key to prevent incorrect resume matches across
    experiments, seeds, or code versions.
    """
    completed = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    k = (
                        str(row.get("experiment", "")),
                        int(row["trial"]),
                        str(row["image_id"]),
                        float(row["noise_sigma"]),
                        str(row["sensing_mode"]),
                        str(row["solver"]),
                        int(row.get("seed", 0)),
                        str(row.get("code_version", "")),
                    )
                    completed.add(k)
                except (KeyError, ValueError):
                    pass
    return completed


def solve_patches(
    solver_name,
    measurements,
    sensing_matrix,
    config,
    sensing_mode="standard",
    dc_coefficients=None,
    solver_obj=None,
    cholesky_factor=None,
    omp_col_norms=None,
):
    """
    Solve sparse recovery for all patch measurements in an image.
    Strictly measures iterative solve_time using time.perf_counter().
    """
    patch_hats = []
    solve_times = []
    iterations_list = []
    residuals_list = []
    final_sigmas = []
    active_ratios = []
    mean_active_counts = []
    failed_ls_list = []
    accepted_steps_list = []

    num_patches = len(measurements)

    try:
        from tqdm import tqdm
        iterator = tqdm(range(num_patches), desc=f"  Solving {solver_name}", leave=False)
    except ImportError:
        iterator = range(num_patches)

    for i in iterator:
        y_meas = measurements[i]
        dc_val = dc_coefficients[i] if dc_coefficients is not None else None

        # -------------------------------------------------------------
        # ASL-SR-DPT Solver Variants
        # -------------------------------------------------------------
        if solver_name.startswith("ASL-SR-DPT"):
            solver = solver_obj
            use_midpoint = config.get("use_midpoint", True)
            if solver_name == "ASL-SR-DPT-MidpointOFF":
                use_midpoint = False
            elif solver_name == "ASL-SR-DPT-MidpointON":
                use_midpoint = True

            decrease_factor = config.get("sigma_decay", 0.95)
            if "decay-" in solver_name:
                try:
                    decrease_factor = float(solver_name.split("decay-")[-1])
                except ValueError:
                    pass

            t0 = time.perf_counter()
            z_hat, diag = solver.denoise_patch(
                y_meas,
                sigma_min=config.get("sigma_min", 0.01),
                decrease_factor=decrease_factor,
                max_iter=config.get("max_iter", 150),
                initial_mu=config.get("initial_mu", 0.2),
                armijo_c=config.get("armijo_c", 1e-4),
                beta_decay=config.get("beta_decay", 0.5),
                max_backtracks=config.get("max_backtracks", 10),
                support_threshold_multiplier=config.get("support_threshold_multiplier", 1e-5),
                support_reopen_interval=config.get("support_reopen_interval", 3),
                use_midpoint=use_midpoint,
                return_diagnostics=True,
            )
            t1 = time.perf_counter()

            solve_times.append(t1 - t0)
            iterations_list.append(diag["iterations"])
            residuals_list.append(diag["final_residual"])
            final_sigmas.append(diag["final_sigma"])
            active_ratios.append(diag.get("active_support_ratio", 0.0))
            mean_active_counts.append(diag.get("mean_active_count", 0.0))
            failed_ls_list.append(diag["failed_line_searches"])
            accepted_steps_list.append(diag["accepted_steps"])

            if sensing_mode == "dc":
                full_coeff = restore_dc_component(dc_val, z_hat)
            else:
                full_coeff = z_hat

            p_spatial = idct_patch(full_coeff.reshape((8, 8)))
            patch_hats.append(p_spatial)

        # -------------------------------------------------------------
        # OMP Baseline
        # -------------------------------------------------------------
        elif solver_name == "OMP":
            max_coeff = config.get("omp_max_coefficients", sensing_matrix.shape[0])
            rel_tol = config.get("omp_relative_residual_tol", 1e-5)

            t0 = time.perf_counter()
            theta_hat, diag = run_omp(
                y_meas,
                sensing_matrix,
                relative_residual_tol=rel_tol,
                max_coefficients=max_coeff,
                col_norms=omp_col_norms,
                return_diagnostics=True,
            )
            t1 = time.perf_counter()

            solve_times.append(t1 - t0)
            iterations_list.append(diag["iterations"])
            residuals_list.append(diag["final_residual"])
            final_sigmas.append(0.0)
            active_ratios.append(float(diag["iterations"]) / float(sensing_matrix.shape[1]))
            mean_active_counts.append(float(diag["iterations"]))
            failed_ls_list.append(0)
            accepted_steps_list.append(diag["iterations"])

            if sensing_mode == "dc":
                full_coeff = restore_dc_component(dc_val, theta_hat)
            else:
                full_coeff = theta_hat

            p_spatial = idct_patch(full_coeff.reshape((8, 8)))
            patch_hats.append(p_spatial)

        # -------------------------------------------------------------
        # LASSO-ADMM Baseline
        # -------------------------------------------------------------
        elif solver_name == "LASSO-ADMM":
            lambda_lasso = config.get("lambda_lasso", config.get("lasso_lambda", 0.01))
            rho = config.get("rho", config.get("lasso_rho", 1.0))
            tol = config.get("lasso_tol", 1e-4)
            max_iter = config.get("lasso_max_iter", 100)

            t0 = time.perf_counter()
            z_hat, diag = run_lasso_admm(
                y_meas,
                sensing_matrix,
                lambda_lasso=lambda_lasso,
                rho=rho,
                tol=tol,
                max_iter=max_iter,
                L=cholesky_factor,
                return_diagnostics=True,
            )
            t1 = time.perf_counter()

            solve_times.append(t1 - t0)
            iterations_list.append(diag["iterations"])
            residuals_list.append(diag["final_primal_residual"])
            final_sigmas.append(0.0)
            active_nz = int(np.count_nonzero(np.abs(z_hat) > 1e-5))
            active_ratios.append(float(active_nz) / float(sensing_matrix.shape[1]))
            mean_active_counts.append(float(active_nz))
            failed_ls_list.append(0)
            accepted_steps_list.append(diag["iterations"])

            if sensing_mode == "dc":
                full_coeff = restore_dc_component(dc_val, z_hat)
            else:
                full_coeff = z_hat

            p_spatial = idct_patch(full_coeff.reshape((8, 8)))
            patch_hats.append(p_spatial)

        else:
            raise ValueError(f"Unknown solver: {solver_name}")

    total_solve_time = float(np.sum(solve_times))
    mean_iterations = float(np.mean(iterations_list))
    mean_residual = float(np.mean(residuals_list))
    mean_sigma = float(np.mean(final_sigmas)) if final_sigmas else 0.0
    mean_active_ratio = float(np.mean(active_ratios))
    mean_active_count = float(np.mean(mean_active_counts))
    sum_failed_ls = int(np.sum(failed_ls_list))
    sum_accepted_steps = int(np.sum(accepted_steps_list))

    summary_stats = {
        "solve_time": total_solve_time,
        "iterations": mean_iterations,
        "mean_iterations_per_patch": mean_iterations,
        "mean_active_support_count": mean_active_count,
        "residual": mean_residual,
        "final_sigma": mean_sigma,
        "active_support_ratio": mean_active_ratio,
        "failed_line_searches": sum_failed_ls,
        "accepted_steps": sum_accepted_steps,
    }

    return patch_hats, summary_stats


def run_benchmark(args):
    """
    Main benchmark execution loop.
    """
    # Root-cause profiling mode (Part 3)
    if getattr(args, "profile", False):
        from profiler import run_profiling_suite
        img_path = os.path.join(args.dataset_dir, "test001.png")
        noise_val = args.noise if args.noise is not None else 15.0
        run_profiling_suite(
            image_path=img_path,
            num_patches=getattr(args, "patches", 500),
            noise_sigma=noise_val,
            output_dir="results/profiling",
            seed=args.seed
        )
        return

    # E1 Verification Short-circuit
    if args.experiment == "E1":
        print("Executing Experiment E1: Solver Verification & Sanity Checks")
        results, ok = run_e1_verification(dataset_dir=args.dataset_dir, verbose=True)
        report_path = os.path.join(args.output_dir, "E1_verification_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"E1 verification report saved to: {report_path}")
        return ok

    config = load_config(args.config)

    # Determine noise levels and sensing mode
    noise_levels = [args.noise] if args.noise is not None else config.get("noise_levels", [15.0, 25.0, 50.0])
    if isinstance(noise_levels, (int, float)):
        noise_levels = [float(noise_levels)]
    else:
        noise_levels = [float(nl) for nl in noise_levels]

    sensing_mode = args.sensing if args.sensing is not None else "standard"
    if args.experiment == "E3":
        sensing_mode = "dc"

    # Select solvers based on experiment
    if args.experiment in ["E2", "E3"]:
        solvers = ["ASL-SR-DPT", "OMP", "LASSO-ADMM"]
    elif args.experiment == "E4":
        solvers = ["ASL-SR-DPT-MidpointON", "ASL-SR-DPT-MidpointOFF"]
    elif args.experiment == "E5":
        schedules = config.get("validation_parameters", {}).get("sigma_decay_grid", [0.90, 0.95, 0.98])
        solvers = [f"ASL-SR-DPT-decay-{s}" for s in schedules]
    elif args.experiment == "E6":
        lambdas = config.get("validation_parameters", {}).get("lambda_reg_grid", [0.01, 0.05, 0.1, 0.2, 0.5])
        solvers = [f"ASL-SR-DPT-lambda-{l}" for l in lambdas]
    elif args.experiment == "E7":
        solvers = ["ASL-SR-DPT", "OMP"]
    elif args.experiment == "E8":
        solvers = ["ASL-SR-DPT", "LASSO-ADMM"]
    else:
        solvers = ["ASL-SR-DPT", "OMP", "LASSO-ADMM"]

    # Load dataset
    print(f"Loading BSD68 dataset from: {args.dataset_dir}")
    dataset = load_bsd68(dataset_dir=args.dataset_dir, limit=args.images, strict_count=68)
    image_ids = list(dataset.keys())
    print(f"Loaded {len(image_ids)} images for benchmark (Experiment {args.experiment}).")

    # PART 17: Pilot mode isolation
    if args.pilot:
        exp_output_dir = os.path.join(args.output_dir, "PILOT")
        os.makedirs(exp_output_dir, exist_ok=True)
        raw_csv_path = os.path.join(exp_output_dir, f"PILOT_{args.experiment}_{sensing_mode}_raw.csv")
        print("=" * 60)
        print("[PILOT MODE] RUNNING PERFORMANCE PILOT BENCHMARK")
        print("=" * 60)
        print("NOTE: Pilot run is for runtime estimation only.")
        print(f"Outputs are stored isolated in: {exp_output_dir}")
        print("=" * 60)
    else:
        exp_output_dir = os.path.join(args.output_dir, args.experiment)
        os.makedirs(exp_output_dir, exist_ok=True)
        raw_csv_path = os.path.join(exp_output_dir, f"{args.experiment}_{sensing_mode}_raw.csv")

    completed_keys = set()
    if args.resume:
        completed_keys = get_completed_keys(raw_csv_path)
        print(f"Resume enabled: {len(completed_keys)} observations currently in {raw_csv_path}")

    csv_file_exists = os.path.exists(raw_csv_path)
    csv_file = open(raw_csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=RAW_CSV_FIELDS)
    if not csv_file_exists:
        csv_writer.writeheader()
        csv_file.flush()

    base_seed = args.seed if args.seed is not None else config.get("base_seed", 20260908)
    patch_size = config.get("patch_size", 8)
    stride = config.get("stride", 2)
    M_std = config.get("M", 38)
    N_std = config.get("N", 64)
    M_ac = config.get("M_ac", 37)

    representative_ids = {"test001", "test004", "test009"}
    total_runs_planned = args.trials * len(noise_levels) * len(image_ids) * len(solvers)
    print(f"Starting Benchmark: {total_runs_planned} total observations planned.")

    pilot_timings = {}
    skipped_observations = 0
    executed_observations = 0

    for trial in range(1, args.trials + 1):
        trial_seed = base_seed + trial * 10000
        sensing_seed = trial_seed + 5000

        # STEP 15: Generate ONE sensing matrix per trial and reuse across patches and images
        if sensing_mode == "dc":
            A = generate_dc_sensing(M_ac=M_ac, N_ac=63, seed=sensing_seed)
        else:
            A = generate_random_sensing(M=M_std, N=N_std, seed=sensing_seed)

        # STEP 16 & 17: Precompute factorizations ONCE per trial
        setup_times = {}

        # ASL-SR-DPT Solver Precomputation
        t_asl_setup_0 = time.perf_counter()
        asl_solver_instances = {}
        for sname in solvers:
            if sname.startswith("ASL-SR-DPT"):
                lam = config.get("lambda_reg", 0.1)
                if "lambda-" in sname:
                    try:
                        lam = float(sname.split("lambda-")[-1])
                    except ValueError:
                        pass
                asl_solver_instances[sname] = HybridSparseSolverV7(
                    A, lambda_reg=lam, tol=config.get("tol", 1e-5), seed=trial_seed
                )
        t_asl_setup_1 = time.perf_counter()
        asl_setup_dur = (t_asl_setup_1 - t_asl_setup_0) / max(len(asl_solver_instances), 1)

        # LASSO-ADMM Precomputation (Cholesky factor L)
        lasso_cholesky = None
        lasso_setup_dur = 0.0
        if "LASSO-ADMM" in solvers:
            t_lasso_setup_0 = time.perf_counter()
            lasso_rho = config.get("rho", config.get("lasso_rho", 1.0))
            lasso_cholesky = precompute_lasso_admm(A, rho=lasso_rho)
            t_lasso_setup_1 = time.perf_counter()
            lasso_setup_dur = t_lasso_setup_1 - t_lasso_setup_0

        # PART 7: OMP Column Norm Precomputation
        omp_col_norms = None
        omp_setup_dur = 0.0
        if "OMP" in solvers:
            t_omp_setup_0 = time.perf_counter()
            omp_col_norms = precompute_omp(A)
            t_omp_setup_1 = time.perf_counter()
            omp_setup_dur = t_omp_setup_1 - t_omp_setup_0

        setup_times["ASL-SR-DPT"] = asl_setup_dur
        setup_times["OMP"] = omp_setup_dur
        setup_times["LASSO-ADMM"] = lasso_setup_dur

        for noise_sigma in noise_levels:
            for img_idx, img_id in enumerate(image_ids):
                clean_img = dataset[img_id]["image"]
                noise_seed = trial_seed + 1000 + img_idx

                # Check if all solvers for this condition are already completed
                pending_solvers = []
                for s in solvers:
                    comp_key = (
                        str(args.experiment),
                        trial,
                        img_id,
                        float(noise_sigma),
                        sensing_mode,
                        s,
                        trial_seed,
                        CODE_VERSION,
                    )
                    if comp_key in completed_keys:
                        skipped_observations += 1
                    else:
                        pending_solvers.append(s)

                if not pending_solvers:
                    continue

                # STEP 14: Generate ONE noisy image for all matched solvers
                noisy_img = add_awgn(clean_img, sigma_noise=noise_sigma, seed=noise_seed)

                # Save clean and noisy representative images once
                if (args.save_reconstructions or (args.save_representative and img_id in representative_ids)) and trial == 1:
                    save_dir = os.path.join("results", "reconstructions", args.experiment)
                    save_image_grayscale(clean_img, os.path.join(save_dir, f"{img_id}_clean.png"))
                    save_image_grayscale(noisy_img, os.path.join(save_dir, f"{img_id}_noisy_sigma{noise_sigma}.png"))

                # Extract patches once
                patch_records = extract_patches(noisy_img, patch_size=patch_size, stride=stride)
                patch_count = len(patch_records)

                # Extract DCT coefficients and generate measurements once
                measurements = []
                dc_coeffs = []
                for p_rec in patch_records:
                    p_dct = dct_patch(p_rec["patch"])
                    if sensing_mode == "dc":
                        theta_dc, y_ac = generate_dc_preserving_measurement(p_dct, A)
                        measurements.append(y_ac)
                        dc_coeffs.append(theta_dc)
                    else:
                        y_std = generate_standard_measurement(p_dct, A)
                        measurements.append(y_std)

                # Execute matched solvers on the EXACT SAME measurements
                for solver_name in pending_solvers:
                    run_id = f"{args.experiment}_{trial}_{img_id}_{noise_sigma}_{solver_name}_{uuid.uuid4().hex[:6]}"

                    # Retrieve precomputed solver objects
                    s_obj = asl_solver_instances.get(solver_name) if solver_name.startswith("ASL-SR-DPT") else None
                    c_fac = lasso_cholesky if solver_name == "LASSO-ADMM" else None

                    patch_hats, stats = solve_patches(
                        solver_name=solver_name,
                        measurements=measurements,
                        sensing_matrix=A,
                        config=config,
                        sensing_mode=sensing_mode,
                        dc_coefficients=dc_coeffs if sensing_mode == "dc" else None,
                        solver_obj=s_obj,
                        cholesky_factor=c_fac,
                        omp_col_norms=omp_col_norms,
                    )

                    # Reconstruct full image using 2D Hamming aggregation
                    recon_records = [
                        {"patch": p_h, "x": prec["x"], "y": prec["y"]}
                        for p_h, prec in zip(patch_hats, patch_records)
                    ]
                    reconstructed = reconstruct_image(recon_records, clean_img.shape, patch_size=patch_size)

                    # Compute full-image spatial metrics
                    metrics_dict = compute_all_metrics(clean_img, reconstructed)

                    # Assign setup time
                    s_time = setup_times.get(solver_name, 0.0)
                    if solver_name.startswith("ASL-SR-DPT"):
                        s_time = asl_setup_dur

                    row_data = {
                        "trial": trial,
                        "image_id": img_id,
                        "noise_sigma": noise_sigma,
                        "sensing_mode": sensing_mode,
                        "solver": solver_name,
                        "psnr": round(metrics_dict["psnr"], 4),
                        "ssim": round(metrics_dict["ssim"], 4),
                        "mse": round(metrics_dict["mse"], 6),
                        "solve_time": round(stats["solve_time"], 4),
                        "setup_time": round(s_time, 6),
                        "iterations": round(stats["iterations"], 2),
                        "mean_iterations_per_patch": round(stats["mean_iterations_per_patch"], 2),
                        "mean_active_support_count": round(stats["mean_active_support_count"], 2),
                        "final_sigma": round(stats["final_sigma"], 4),
                        "residual": round(stats["residual"], 4),
                        "active_support_ratio": round(stats["active_support_ratio"], 4),
                        "failed_line_searches": stats["failed_line_searches"],
                        "accepted_steps": stats["accepted_steps"],
                        "seed": trial_seed,
                        "code_version": CODE_VERSION,
                        "patch_count": patch_count,
                        "run_id": run_id,
                        "experiment": args.experiment,
                    }

                    csv_writer.writerow(row_data)
                    csv_file.flush()

                    comp_key = (
                        str(args.experiment),
                        trial,
                        img_id,
                        float(noise_sigma),
                        sensing_mode,
                        solver_name,
                        trial_seed,
                        CODE_VERSION,
                    )
                    completed_keys.add(comp_key)
                    executed_observations += 1

                    if args.pilot:
                        pilot_timings[solver_name] = pilot_timings.get(solver_name, []) + [stats["solve_time"]]

                    # Save representative reconstruction
                    if (args.save_reconstructions or (args.save_representative and img_id in representative_ids)) and trial == 1:
                        save_dir = os.path.join("results", "reconstructions", args.experiment)
                        save_image_grayscale(
                            reconstructed,
                            os.path.join(save_dir, f"{img_id}_{solver_name}_sigma{noise_sigma}_{sensing_mode}.png")
                        )

                    print(
                        f"[{args.experiment}] Tr:{trial} Img:{img_id} Sigma:{noise_sigma} Solver:{solver_name} -> "
                        f"PSNR:{metrics_dict['psnr']:.2f}dB SSIM:{metrics_dict['ssim']:.4f} Time:{stats['solve_time']:.2f}s"
                    )

    csv_file.close()

    if args.resume:
        print(f"\nResume summary: {skipped_observations} observations skipped, {executed_observations} newly executed.")

    print(f"Raw results logged to: {raw_csv_path}")

    # STEP 33 & 34: Run invalid data audit and generate summary CSV
    print("Running data audit and generating summary...")
    if args.pilot:
        summary_path = os.path.join(exp_output_dir, f"PILOT_{args.experiment}_{sensing_mode}_summary.csv")
    else:
        summary_path = os.path.join("results", "summaries", f"{args.experiment}_{sensing_mode}_summary.csv")

    generate_summary(raw_csv_path, output_summary_path=summary_path)

    # STEP 35: Pilot Mode Report
    if args.pilot:
        print("\n" + "=" * 60)
        print("[PILOT MODE] ESTIMATED BENCHMARK DURATION SUMMARY")
        print("=" * 60)
        full_images = 68
        full_trials = 50
        full_noises = 3
        total_benchmark_runs = full_images * full_trials * full_noises

        for sname, times in pilot_timings.items():
            avg_img_time = np.mean(times)
            total_sec = avg_img_time * total_benchmark_runs
            total_hrs = total_sec / 3600.0
            print(f"Solver: {sname:<20} | Time/Image: {avg_img_time:.2f}s | Projected 50-Trial Runtime: {total_hrs:.2f} hours")
        print("=" * 60)
        print("NOTE: Pilot workload complete. No production benchmark files were overwritten.\n")

    return True


def parse_args():
    """
    Robust command line interface for ASL-SR-DPT Benchmark Runner.
    Enforces parameter validation and safe defaults (images=1, experiment=E1).
    """
    parser = argparse.ArgumentParser(
        description="ASL-SR-DPT Benchmark Runner"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="E1",
        choices=["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"],
        help="Experiment identifier."
    )

    parser.add_argument(
        "--images",
        type=int,
        default=1,
        help="Number of BSD68 images. Use 1 for smoke tests and 68 for the final benchmark."
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of randomized trials."
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=None,
        choices=[15.0, 25.0, 50.0],
        help="Noise sigma. Leave unset to use all configured noise levels."
    )

    parser.add_argument(
        "--sensing",
        type=str,
        default=None,
        choices=["standard", "dc"],
        help="Sensing mode."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260908,
        help="Base random seed."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume without duplicating completed observations."
    )

    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run a pilot and estimate runtime."
    )

    parser.add_argument(
        "--save-reconstructions",
        action="store_true",
        help="Save all reconstructed images."
    )

    parser.add_argument(
        "--save-representative",
        action="store_true",
        help="Save fixed representative reconstructions."
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/BSD68",
        help="BSD68 directory."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/raw",
        help="Raw output directory."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/final_config.json",
        help="Configuration file."
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run root-cause profiling mode (saves to results/profiling/)."
    )

    parser.add_argument(
        "--patches",
        type=int,
        default=500,
        help="Number of patches to profile in profile mode (default 500)."
    )

    args = parser.parse_args()

    if not 1 <= args.images <= 68:
        parser.error("--images must be between 1 and 68.")

    if args.trials < 1:
        parser.error("--trials must be at least 1.")

    return args


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
