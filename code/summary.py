"""
Summary generation and data audit module for ASL-SR-DPT.
Implements:
  1. Strict invalid data audit (NaN, Inf, negative time, duplicate keys).
  2. Logging failed runs to logs/failed_runs.csv.
  3. Calculation of sample mean, standard deviation (ddof=1).
  4. Speedup and latency reduction calculations relative to baselines.
  5. Export of formatted summary CSVs.
"""

import os
import csv
import datetime
import numpy as np


def audit_row(row, seen_keys):
    """
    Audit a single raw result row for validity.
    Returns (is_valid, critical_error, review_warning).
      - is_valid (bool): False if critical corruption occurs (row excluded from summary).
      - critical_error (str or None): Reason for exclusion if invalid.
      - review_warning (str or None): Anomaly triggering manual review investigation (row kept in summary).
    """
    key = (
        row.get("trial"),
        row.get("image_id"),
        row.get("noise_sigma"),
        row.get("sensing_mode"),
        row.get("solver"),
    )

    if any(k is None or str(k).strip() == "" for k in key):
        return False, "Missing observation key field", None

    if key in seen_keys:
        return False, f"Duplicate observation key: {key}", None
    seen_keys.add(key)

    review_warning = None

    # Validate metrics
    for metric_name in ["psnr", "ssim", "mse"]:
        val_str = row.get(metric_name)
        if val_str is None or str(val_str).strip() == "":
            return False, f"Missing metric: {metric_name}", None
        try:
            val = float(val_str)
            if not np.isfinite(val):
                return False, f"Non-finite metric {metric_name}={val}", None
        except ValueError:
            return False, f"Invalid numeric format for {metric_name}: {val_str}", None

    psnr = float(row["psnr"])
    ssim = float(row["ssim"])

    # Manual review triggers (data is mathematically finite and kept, but flagged for inspection)
    if psnr > 60.0:
        review_warning = f"Unusually high PSNR ({psnr:.2f} dB > 60 dB) - check for zero-residual or flat synthetic patch"
    elif psnr < 5.0:
        review_warning = f"Unusually low PSNR ({psnr:.2f} dB < 5 dB) - check for severe convergence failure"
    elif ssim < 0.05:
        review_warning = f"Unusually low SSIM ({ssim:.4f} < 0.05) - check for structural collapse"

    # Validate timing
    try:
        solve_time = float(row.get("solve_time", 0.0))
        setup_time = float(row.get("setup_time", 0.0))
        if not np.isfinite(solve_time) or solve_time <= 0.0:
            return False, f"Invalid solve_time: {solve_time}", None
        if not np.isfinite(setup_time) or setup_time < 0.0:
            return False, f"Invalid setup_time: {setup_time}", None
    except ValueError:
        return False, "Invalid timing values", None

    # Validate iterations
    try:
        iters = float(row.get("iterations", -1))
        if iters < 0:
            return False, f"Invalid iteration count: {iters}", None
    except ValueError:
        return False, "Invalid iteration value", None

    return True, None, review_warning


def log_failed_run(failed_log_path, row, error_msg):
    """
    Append an invalid run record to logs/failed_runs.csv.
    """
    os.makedirs(os.path.dirname(failed_log_path), exist_ok=True)
    file_exists = os.path.isfile(failed_log_path)

    fieldnames = [
        "timestamp",
        "run_id",
        "experiment",
        "trial",
        "image_id",
        "solver",
        "error",
    ]

    with open(failed_log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.datetime.now().isoformat(),
            "run_id": row.get("run_id", "unknown"),
            "experiment": row.get("experiment", "unknown"),
            "trial": row.get("trial", "unknown"),
            "image_id": row.get("image_id", "unknown"),
            "solver": row.get("solver", "unknown"),
            "error": error_msg,
        })


def log_manual_review(review_log_path, row, warning_msg):
    """
    Append an observation flagged for manual review to logs/manual_review_flags.csv.
    """
    os.makedirs(os.path.dirname(review_log_path), exist_ok=True)
    file_exists = os.path.isfile(review_log_path)

    fieldnames = [
        "timestamp",
        "run_id",
        "experiment",
        "trial",
        "image_id",
        "solver",
        "warning",
    ]

    with open(review_log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.datetime.now().isoformat(),
            "run_id": row.get("run_id", "unknown"),
            "experiment": row.get("experiment", "unknown"),
            "trial": row.get("trial", "unknown"),
            "image_id": row.get("image_id", "unknown"),
            "solver": row.get("solver", "unknown"),
            "warning": warning_msg,
        })


def generate_summary(
    raw_csv_path,
    output_summary_path=None,
    failed_runs_log="logs/failed_runs.csv",
    review_log="logs/manual_review_flags.csv",
):
    """
    Read raw CSV, audit data, compute mean/std, calculate speedups, and write summary CSV.
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv_path}")

    valid_rows = []
    seen_keys = set()

    with open(raw_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_valid, err, warn = audit_row(row, seen_keys)
            if is_valid:
                valid_rows.append(row)
                if warn:
                    log_manual_review(review_log, row, warn)
            else:
                log_failed_run(failed_runs_log, row, err)

    if not valid_rows:
        print(f"Warning: No valid rows found in {raw_csv_path}")
        return None

    # Group rows by (experiment, noise_sigma, sensing_mode, solver)
    groups = {}
    for r in valid_rows:
        gkey = (
            r.get("experiment", "unknown"),
            float(r["noise_sigma"]),
            r["sensing_mode"],
            r["solver"],
        )
        groups.setdefault(gkey, []).append(r)

    summary_records = []
    for gkey, rlist in sorted(groups.items()):
        exp, noise, sensing, solver = gkey
        n_obs = len(rlist)

        def get_stat(field):
            vals = [float(item[field]) for item in rlist if item.get(field) is not None and str(item[field]).strip() != ""]
            if not vals:
                return 0.0, 0.0
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            return mean, std

        psnr_m, psnr_s = get_stat("psnr")
        ssim_m, ssim_s = get_stat("ssim")
        mse_m, mse_s = get_stat("mse")
        solve_m, solve_s = get_stat("solve_time")
        setup_m, setup_s = get_stat("setup_time")
        iter_m, iter_s = get_stat("iterations")
        res_m, res_s = get_stat("residual")
        supp_m, supp_s = get_stat("active_support_ratio")
        failed_m, _ = get_stat("failed_line_searches")
        accepted_m, _ = get_stat("accepted_steps")

        total_time_m = solve_m + setup_m

        summary_records.append({
            "experiment": exp,
            "noise_sigma": noise,
            "sensing_mode": sensing,
            "solver": solver,
            "count": n_obs,
            "psnr_mean": round(psnr_m, 4),
            "psnr_std": round(psnr_s, 4),
            "ssim_mean": round(ssim_m, 4),
            "ssim_std": round(ssim_s, 4),
            "mse_mean": round(mse_m, 6),
            "mse_std": round(mse_s, 6),
            "solve_time_mean": round(solve_m, 4),
            "solve_time_std": round(solve_s, 4),
            "setup_time_mean": round(setup_m, 6),
            "setup_time_std": round(setup_s, 6),
            "total_time_mean": round(total_time_m, 4),
            "iterations_mean": round(iter_m, 2),
            "iterations_std": round(iter_s, 2),
            "residual_mean": round(res_m, 4),
            "active_support_ratio_mean": round(supp_m, 4),
            "failed_line_searches_mean": round(failed_m, 2),
            "accepted_steps_mean": round(accepted_m, 2),
        })

    # Speedup and latency reduction calculation
    # Relative to baselines within each (experiment, noise_sigma, sensing_mode)
    by_condition = {}
    for rec in summary_records:
        cond_key = (rec["experiment"], rec["noise_sigma"], rec["sensing_mode"])
        by_condition.setdefault(cond_key, {})[rec["solver"]] = rec

    for cond_key, solver_dict in by_condition.items():
        asl_rec = solver_dict.get("ASL-SR-DPT")
        if asl_rec:
            asl_time = asl_rec["solve_time_mean"]
            for sname, srec in solver_dict.items():
                if sname != "ASL-SR-DPT" and srec["solve_time_mean"] > 0:
                    base_time = srec["solve_time_mean"]
                    # Speedup = base_time / asl_time
                    speedup = base_time / asl_time if asl_time > 0 else 0.0
                    # Latency reduction % = 100 * (base_time - asl_time) / base_time
                    latency_red = 100.0 * (base_time - asl_time) / base_time
                    srec["speedup_vs_asl"] = round(1.0 / speedup if speedup > 0 else 0.0, 3)
                    asl_rec[f"speedup_vs_{sname}"] = round(speedup, 3)
                    asl_rec[f"latency_reduction_pct_vs_{sname}"] = round(latency_red, 2)

    if output_summary_path is None:
        base_name = os.path.splitext(os.path.basename(raw_csv_path))[0]
        output_summary_path = os.path.join("results", "summaries", f"{base_name}_summary.csv")

    os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)
    if summary_records:
        # Collect all keys present across summary records
        all_fieldnames = []
        for s in summary_records:
            for k in s.keys():
                if k not in all_fieldnames:
                    all_fieldnames.append(k)

        with open(output_summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames)
            writer.writeheader()
            writer.writerows(summary_records)

        print(f"Summary successfully written to: {output_summary_path}")

    return summary_records
