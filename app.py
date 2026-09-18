"""
ASL-SR-DPT Pilot Benchmark Standalone Application
Distributed benchmarking environment for Team A (Khevin) and Team B (Marc).
Enforces strictly non-overlapping 225-evaluation workloads, single-threaded execution,
rigorous 5-step safety gates, true deterministic reproducibility, real solver diagnostics,
and comprehensive central merge auditing.
"""

import os
import sys

# 1. Enforce strict single-threaded execution BEFORE NumPy/SciPy/BLAS imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import json
import csv
import io
import zipfile
import platform
import hashlib
import math
from datetime import datetime

import numpy as np
import scipy
from PIL import Image

# Ensure repository root and code directory are in sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(REPO_ROOT, "code")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# Core verified solver and pipeline imports
from hybrid_sparse_solver_v7_optimized import HybridSparseSolverV7Optimized
from hybrid_sparse_solver_v7_fixed import (
    run_omp,
    precompute_omp,
    run_lasso_admm,
    precompute_lasso_admm,
)
from sensing import (
    generate_random_sensing,
    generate_dc_sensing,
    add_awgn,
)
from reconstruction import (
    extract_patches,
    dct_patch,
    idct_patch,
    reconstruct_image,
)
from metrics import compute_all_metrics

import streamlit as st
import pandas as pd


# ==============================================================================
# 2. Configuration, Codebase Hashing & Integrity
# ==============================================================================
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "pilot_config.json")

RELEVANT_CODE_FILES = [
    os.path.join(REPO_ROOT, "configs", "pilot_config.json"),
    os.path.join(CODE_DIR, "hybrid_sparse_solver_v7_optimized.py"),
    os.path.join(CODE_DIR, "hybrid_sparse_solver_v7_fixed.py"),
    os.path.join(CODE_DIR, "sensing.py"),
    os.path.join(CODE_DIR, "reconstruction.py"),
    os.path.join(CODE_DIR, "metrics.py"),
    os.path.join(REPO_ROOT, "app.py"),
]
_req_path = os.path.join(REPO_ROOT, "requirements.txt")
if os.path.exists(_req_path):
    RELEVANT_CODE_FILES.append(_req_path)


def compute_file_sha256(filepath):
    """Compute SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return "missing"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_codebase_integrity_hash():
    """
    Computes an aggregate SHA-256 hash across relevant solver, sensing,
    reconstruction, metrics, config, app orchestration, and requirements files.
    Changing any of these files immediately alters CODE_HASH.
    """
    file_hashes = {}
    combined = hashlib.sha256()
    for fpath in RELEVANT_CODE_FILES:
        rel = os.path.relpath(fpath, REPO_ROOT)
        h = compute_file_sha256(fpath)
        file_hashes[rel] = h
        combined.update(f"{rel}:{h}".encode("utf-8"))
    return combined.hexdigest(), file_hashes


def load_pilot_config(path=CONFIG_PATH):
    """Load frozen pilot configuration and compute its SHA-256 hash."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    cfg_hash = compute_file_sha256(path)
    return config, cfg_hash


try:
    PILOT_CONFIG, CONFIG_HASH = load_pilot_config()
except Exception:
    PILOT_CONFIG, CONFIG_HASH = {}, "CONFIG_ERROR"

CODE_HASH, FILE_HASHES = compute_codebase_integrity_hash()


def get_environment_info():
    """Collect certified environment metadata with explicit scipy version (Requirement 5)."""
    return {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "cpu": platform.processor() or platform.machine(),
        "thread_settings": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "not_set"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "not_set"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "not_set"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "not_set"),
        },
        "app_version": "ASL-SR-DPT-PILOT-1.0",
        "config_hash": CONFIG_HASH,
        "code_hash": CODE_HASH,
    }


def compute_deterministic_seeds(image_id, noise_sigma, trial, base_seed=20260908):
    """
    Deterministic seed formulas:
      clean_id = int(str(image_id).replace("test", "").lstrip("0") or "0")
      sensing_seed = base_seed + clean_id * 1000000 + int(noise_sigma) * 1000 + int(trial)
      noise_seed = base_seed + int(trial) * 100000 + int(noise_sigma) * 1000 + clean_id
    """
    clean_id = int(str(image_id).replace("test", "").lstrip("0") or "0")
    sensing_seed = base_seed + clean_id * 1000000 + int(noise_sigma) * 1000 + int(trial)
    noise_seed = base_seed + int(trial) * 100000 + int(noise_sigma) * 1000 + clean_id
    return sensing_seed, noise_seed


def get_team_directories(team_name):
    """Create and return separated paths for each team."""
    tn = str(team_name).strip().lower()
    if "b" in tn.split() or tn.endswith("b") or "team_b" in tn or "team b" in tn:
        team_slug = "team_b"
    else:
        team_slug = "team_a"
    base_dir = os.path.join(REPO_ROOT, "results", team_slug)
    paths = {
        "base": base_dir,
        "raw": os.path.join(base_dir, "raw"),
        "logs": os.path.join(base_dir, "logs"),
        "failures": os.path.join(base_dir, "failures"),
        "reconstructions": os.path.join(base_dir, "reconstructions"),
        "summaries": os.path.join(base_dir, "summaries"),
        "raw_csv": os.path.join(base_dir, "raw", f"pilot_raw_results_{team_slug}.csv"),
        "failures_csv": os.path.join(base_dir, "failures", "pilot_failures.csv"),
        "team_report_md": os.path.join(base_dir, "summaries", "team_report.md"),
        "team_summary_csv": os.path.join(base_dir, "summaries", "team_summary.csv"),
        "env_json": os.path.join(base_dir, "logs", "environment.json"),
    }
    for p in [paths["raw"], paths["logs"], paths["failures"], paths["reconstructions"], paths["summaries"]]:
        os.makedirs(p, exist_ok=True)
    return paths


# ==============================================================================
# 3. CSV Schema & Validation Invariants
# ==============================================================================
RAW_COLUMNS = [
    "run_id",
    "image_id",
    "noise_level",
    "noise_sigma",
    "trial",
    "solver",
    "team",
    "sensing_seed",
    "noise_seed",
    "psnr",
    "ssim",
    "mse",
    "setup_time",
    "solve_time",
    "milliseconds_per_patch",
    "ms_per_patch",
    "iteration_count",
    "measurement_residual",
    "relative_residual",
    "normalized_residual",
    "active_support_ratio",
    "mean_active_support_count",
    "final_sigma",
    "accepted_steps",
    "failed_line_searches",
    "support_size",
    "nonzero_coefficient_count",
    "primal_residual",
    "dual_residual",
    "sensing_m",
    "sensing_n",
    "sensing_architecture",
    "diag_dc_error",
    "diag_ac_error",
    "config_hash",
    "code_version",
    "code_hash",
    "timestamp",
]


def load_existing_composite_keys(raw_csv_path, expected_config_hash=None, expected_code_hash=None):
    """
    Load completed experiment keys (image_id, noise_level, trial, solver) to prevent duplicates.
    Also validates that existing dataset matches the current configuration and codebase hash (Requirement 19).
    """
    keys = set()
    if os.path.exists(raw_csv_path):
        try:
            df = pd.read_csv(raw_csv_path)
            if not df.empty:
                if expected_config_hash and "config_hash" in df.columns:
                    mismatches = df[df["config_hash"].astype(str) != str(expected_config_hash)]
                    if not mismatches.empty:
                        raise ValueError(
                            f"Configuration hash mismatch in {raw_csv_path}! "
                            f"Dataset contains records generated under config hash '{mismatches.iloc[0]['config_hash']}', "
                            f"but current configuration hash is '{expected_config_hash}'. "
                            f"Cannot mix results across differing configurations."
                        )
                if expected_code_hash and "code_hash" in df.columns:
                    mismatches = df[df["code_hash"].astype(str) != str(expected_code_hash)]
                    if not mismatches.empty:
                        raise ValueError(
                            f"Codebase integrity hash mismatch in {raw_csv_path}! "
                            f"Dataset contains records from code hash '{mismatches.iloc[0]['code_hash']}', "
                            f"but current code hash is '{expected_code_hash}'. "
                            f"Cannot mix results across differing code versions."
                        )
                for _, r in df.iterrows():
                    try:
                        k = (str(r["image_id"]), float(r["noise_sigma"]), int(r["trial"]), str(r["solver"]))
                        keys.add(k)
                    except (KeyError, ValueError):
                        pass
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
    return keys


def validate_dpt_batch_diagnostics(diag, expected_batch_size):
    """
    Strictly validates ASL-SR-DPT diagnostic schema per batch and per patch (Requirement 3):
    - Array lengths match batch size
    - Iterations > 0
    - 0 <= |S_k| <= 63 for every iteration k
    - Accepted steps and failed line searches are non-negative
    - Final sigma is positive and finite
    Returns (passed: bool, error_message: str).
    """
    required_keys = [
        "patch_iterations",
        "patch_active_counts",
        "patch_accepted_steps",
        "patch_failed_line_searches",
        "patch_final_sigma",
    ]
    if not isinstance(diag, dict):
        return False, f"DPT diagnostics must be a dict, got {type(diag)}"
    for k in required_keys:
        if k not in diag:
            return False, f"Missing required DPT diagnostic key: '{k}'"

    for k in required_keys:
        val = diag[k]
        if not hasattr(val, "__len__") or len(val) != expected_batch_size:
            return False, (
                f"DPT diagnostic '{k}' length mismatch: expected {expected_batch_size}, "
                f"got {len(val) if hasattr(val, '__len__') else type(val)}"
            )

    for b in range(expected_batch_size):
        iters = diag["patch_iterations"][b]
        if iters <= 0 or not math.isfinite(iters):
            return False, f"Patch {b} iteration count must be > 0, got {iters}"

        act_counts = diag["patch_active_counts"][b]
        if len(act_counts) != int(iters):
            return False, f"Patch {b} active count history length ({len(act_counts)}) != iterations ({iters})"

        for it_idx, s_k in enumerate(act_counts):
            if s_k < 0 or s_k > 63:
                return False, f"Patch {b} iteration {it_idx} active support size |S_k|={s_k} out of bounds [0, 63]"

        acc = diag["patch_accepted_steps"][b]
        if acc < 0 or acc > iters:
            return False, f"Patch {b} accepted steps ({acc}) invalid for {iters} iterations"

        failed = diag["patch_failed_line_searches"][b]
        if failed < 0:
            return False, f"Patch {b} failed line searches ({failed}) must be >= 0"

        sig = diag["patch_final_sigma"][b]
        if sig <= 0 or not math.isfinite(sig):
            return False, f"Patch {b} final sigma ({sig}) must be positive and finite"

    return True, ""


def validate_evaluation_record(record, assigned_images, valid_solvers, valid_noise_levels, valid_trials, config_hash):
    """
    Validates a result record before writing to disk (Requirement 15).
    Verifies metric finiteness, validity of parameters, workload membership, and config consistency.
    """
    numeric_fields = ["psnr", "ssim", "mse", "setup_time", "solve_time", "ms_per_patch", "measurement_residual"]
    for f in numeric_fields:
        val = record.get(f)
        if val is None or not math.isfinite(float(val)):
            return False, f"Metric '{f}' must be finite, got {val}"

    if float(record["solve_time"]) <= 0:
        return False, f"solve_time must be strictly positive, got {record['solve_time']}"

    solver = str(record.get("solver"))
    if solver not in valid_solvers:
        return False, f"Invalid solver '{solver}', expected one of {valid_solvers}"

    if float(record.get("noise_sigma")) not in [float(x) for x in valid_noise_levels]:
        return False, f"Invalid noise level '{record.get('noise_sigma')}'"

    if int(record.get("trial")) not in [int(x) for x in valid_trials]:
        return False, f"Invalid trial '{record.get('trial')}'"

    if str(record.get("image_id")) not in assigned_images:
        return False, f"Image '{record.get('image_id')}' is not in assigned workload: {assigned_images}"

    if str(record.get("config_hash")) != str(config_hash):
        return False, f"config_hash mismatch: expected {config_hash}, got {record.get('config_hash')}"

    # Strict Sensing dimensions & architecture validation
    s_m = record.get("sensing_m")
    s_n = record.get("sensing_n")
    s_arch = record.get("sensing_architecture")
    if solver == "ASL-SR-DPT":
        if s_m != 37 or s_n != 63 or s_arch != "dc_preserving":
            return False, f"ASL-SR-DPT sensing dimensions invalid: M={s_m}, N={s_n}, arch={s_arch}"
        if record.get("active_support_ratio") == "" or not math.isfinite(float(record.get("active_support_ratio"))):
            return False, "ASL-SR-DPT active_support_ratio must be present and finite"
        r_act = float(record["active_support_ratio"])
        if r_act < 0.0 or r_act > 1.0:
            return False, f"ASL-SR-DPT active_support_ratio out of range [0, 1]: {r_act}"
        if record.get("mean_active_support_count") == "" or not math.isfinite(float(record.get("mean_active_support_count"))):
            return False, "ASL-SR-DPT mean_active_support_count must be present and finite"
        m_act = float(record["mean_active_support_count"])
        if m_act < 0.0 or m_act > 63.0:
            return False, f"ASL-SR-DPT mean_active_support_count out of range [0, 63]: {m_act}"
        if record.get("final_sigma") == "" or float(record["final_sigma"]) <= 0:
            return False, f"ASL-SR-DPT final_sigma must be positive, got {record.get('final_sigma')}"
        if record.get("accepted_steps") == "" or float(record["accepted_steps"]) < 0:
            return False, f"ASL-SR-DPT accepted_steps must be >= 0, got {record.get('accepted_steps')}"
        if record.get("failed_line_searches") == "" or float(record["failed_line_searches"]) < 0:
            return False, f"ASL-SR-DPT failed_line_searches must be >= 0, got {record.get('failed_line_searches')}"
    elif solver in ["OMP", "LASSO-ADMM"]:
        if s_m != 38 or s_n != 64 or s_arch != "standard":
            return False, f"{solver} sensing dimensions invalid: M={s_m}, N={s_n}, arch={s_arch}"
        if record.get("active_support_ratio") != "":
            return False, f"{solver} active_support_ratio must be blank/empty, got {record.get('active_support_ratio')}"
        if record.get("mean_active_support_count") != "":
            return False, f"{solver} mean_active_support_count must be blank/empty, got {record.get('mean_active_support_count')}"
        if record.get("final_sigma") != "":
            return False, f"{solver} final_sigma must be blank/empty, got {record.get('final_sigma')}"
        if record.get("accepted_steps") != "":
            return False, f"{solver} accepted_steps must be blank/empty, got {record.get('accepted_steps')}"
        if record.get("failed_line_searches") != "":
            return False, f"{solver} failed_line_searches must be blank/empty, got {record.get('failed_line_searches')}"

    return True, ""


def append_evaluation_record(raw_csv_path, record):
    """Append completed evaluation row immediately to CSV with atomic flush."""
    file_exists = os.path.exists(raw_csv_path)
    with open(raw_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
        f.flush()


def log_failure_record(failures_csv_path, failure_dict):
    """Log individual evaluation failure without interrupting the entire benchmark."""
    cols = ["image_id", "noise", "trial", "solver", "error_type", "error_message", "timestamp"]
    file_exists = os.path.exists(failures_csv_path)
    with open(failures_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if not file_exists:
            writer.writeheader()
        writer.writerow(failure_dict)
        f.flush()


# ==============================================================================
# 4. Deep Parameter Validation & Safety Gates
# ==============================================================================
def validate_frozen_research_parameters(config, config_hash):
    """
    Deep validation of all frozen research parameters against thesis specification (Requirement 8).
    Returns (passed: bool, errors: list[str]).
    """
    errors = []

    if config.get("patch_size") != 8:
        errors.append(f"patch_size must be 8, got {config.get('patch_size')}")
    if config.get("stride") != 2:
        errors.append(f"stride must be 2, got {config.get('stride')}")
    if config.get("N") != 64:
        errors.append(f"N must be 64, got {config.get('N')}")
    if config.get("M") != 38:
        errors.append(f"M must be 38, got {config.get('M')}")
    if config.get("M_ac") != 37:
        errors.append(f"M_ac must be 37, got {config.get('M_ac')}")
    if config.get("N_ac") != 63:
        errors.append(f"N_ac must be 63, got {config.get('N_ac')}")

    dpt = config.get("asl_sr_dpt", {})
    if dpt.get("sensing_mode") != "dc_preserving":
        errors.append(f"ASL-SR-DPT sensing_mode must be 'dc_preserving', got {dpt.get('sensing_mode')}")
    if abs(float(dpt.get("lambda_reg", 0)) - 0.1) > 1e-9:
        errors.append(f"ASL-SR-DPT lambda_reg must be 0.1, got {dpt.get('lambda_reg')}")
    if abs(float(dpt.get("sigma_min", 0)) - 0.01) > 1e-9:
        errors.append(f"ASL-SR-DPT sigma_min must be 0.01, got {dpt.get('sigma_min')}")
    if abs(float(dpt.get("sigma_decay", 0)) - 0.95) > 1e-9:
        errors.append(f"ASL-SR-DPT sigma_decay must be 0.95, got {dpt.get('sigma_decay')}")
    if int(dpt.get("max_iter", 0)) != 150:
        errors.append(f"ASL-SR-DPT max_iter must be 150, got {dpt.get('max_iter')}")
    if abs(float(dpt.get("tol", 0)) - 1e-5) > 1e-9:
        errors.append(f"ASL-SR-DPT tol must be 1e-5, got {dpt.get('tol')}")
    if abs(float(dpt.get("initial_mu", 0)) - 0.2) > 1e-9:
        errors.append(f"ASL-SR-DPT initial_mu must be 0.2, got {dpt.get('initial_mu')}")
    if abs(float(dpt.get("armijo_c", 0)) - 1e-4) > 1e-9:
        errors.append(f"ASL-SR-DPT armijo_c must be 1e-4, got {dpt.get('armijo_c')}")
    if abs(float(dpt.get("beta_decay", 0)) - 0.5) > 1e-9:
        errors.append(f"ASL-SR-DPT beta_decay must be 0.5, got {dpt.get('beta_decay')}")
    if int(dpt.get("max_backtracks", 0)) != 10:
        errors.append(f"ASL-SR-DPT max_backtracks must be 10, got {dpt.get('max_backtracks')}")
    if abs(float(dpt.get("support_threshold_multiplier", 0)) - 1e-5) > 1e-9:
        errors.append(f"ASL-SR-DPT support_threshold_multiplier must be 1e-5, got {dpt.get('support_threshold_multiplier')}")
    if int(dpt.get("support_reopen_interval", 0)) != 3:
        errors.append(f"ASL-SR-DPT support_reopen_interval must be 3, got {dpt.get('support_reopen_interval')}")
    if dpt.get("use_midpoint") is not True:
        errors.append(f"ASL-SR-DPT use_midpoint must be True, got {dpt.get('use_midpoint')}")

    omp = config.get("omp", {})
    if omp.get("sensing_mode") != "standard":
        errors.append(f"OMP sensing_mode must be 'standard', got {omp.get('sensing_mode')}")
    if int(omp.get("max_coefficients", 0)) != 38:
        errors.append(f"OMP max_coefficients must be 38, got {omp.get('max_coefficients')}")
    if abs(float(omp.get("relative_residual_tol", 0)) - 1e-5) > 1e-9:
        errors.append(f"OMP relative_residual_tol must be 1e-5, got {omp.get('relative_residual_tol')}")

    admm = config.get("lasso_admm", {})
    if admm.get("sensing_mode") != "standard":
        errors.append(f"LASSO-ADMM sensing_mode must be 'standard', got {admm.get('sensing_mode')}")
    if abs(float(admm.get("lambda_lasso", 0)) - 0.01) > 1e-9:
        errors.append(f"LASSO-ADMM lambda_lasso must be 0.01, got {admm.get('lambda_lasso')}")
    if abs(float(admm.get("rho", 0)) - 1.0) > 1e-9:
        errors.append(f"LASSO-ADMM rho must be 1.0, got {admm.get('rho')}")
    if int(admm.get("max_iter", 0)) != 100:
        errors.append(f"LASSO-ADMM max_iter must be 100, got {admm.get('max_iter')}")
    if abs(float(admm.get("tol", 0)) - 1e-4) > 1e-9:
        errors.append(f"LASSO-ADMM tol must be 1e-4, got {admm.get('tol')}")
    if abs(float(admm.get("internal_tol", 0)) - 1e-5) > 1e-9:
        errors.append(f"LASSO-ADMM internal_tol must be 1e-5, got {admm.get('internal_tol')}")

    return (len(errors) == 0), errors


def validate_team_assignment_integrity(config):
    """Verifies team assignment integrity and disjointness (Requirement 9)."""
    team_a = config.get("team_assignments", {}).get("Team A", [])
    team_b = config.get("team_assignments", {}).get("Team B", [])
    errors = []
    if len(team_a) != 5:
        errors.append(f"Team A must have exactly 5 images, got {len(team_a)}: {team_a}")
    if len(team_b) != 5:
        errors.append(f"Team B must have exactly 5 images, got {len(team_b)}: {team_b}")

    overlap = set(team_a).intersection(set(team_b))
    if len(overlap) > 0:
        errors.append(f"Overlapping image assignment detected between teams: {overlap}")

    union_set = set(team_a).union(set(team_b))
    expected_10 = {f"test{i:03d}" for i in range(1, 11)}
    if union_set != expected_10:
        errors.append(f"Union of team assignments must be test001..test010, got {sorted(list(union_set))}")

    return (len(errors) == 0), errors


def validate_gate_5_experiment_integrity(team_name, config, config_hash, code_hash):
    """
    STEP 5: Final Experiment Integrity Verification (Requirement 7).
    Validates workload keys, sensing dimensions, dataset availability, and storage.
    """
    errors = []
    assigned = config.get("team_assignments", {}).get(team_name, [])
    if len(assigned) != 5:
        errors.append(f"Team must have exactly 5 images, got {len(assigned)}")

    noises = config.get("noise_levels", [])
    if len(noises) != 3 or set(noises) != {15.0, 25.0, 50.0}:
        errors.append(f"Expected 3 noise levels [15.0, 25.0, 50.0], got {noises}")

    trials = config.get("trials", [])
    if len(trials) != 5 or set(trials) != {1, 2, 3, 4, 5}:
        errors.append(f"Expected 5 trials [1, 2, 3, 4, 5], got {trials}")

    solvers = config.get("solvers", [])
    if len(solvers) != 3 or set(solvers) != {"ASL-SR-DPT", "OMP", "LASSO-ADMM"}:
        errors.append(f"Expected 3 solvers ['ASL-SR-DPT', 'OMP', 'LASSO-ADMM'], got {solvers}")

    expected_count = len(assigned) * len(noises) * len(trials) * len(solvers)
    if expected_count != 225:
        errors.append(f"Expected 225 workload keys, got {expected_count}")

    team_ok, team_errs = validate_team_assignment_integrity(config)
    if not team_ok:
        errors.extend(team_errs)

    param_ok, param_errs = validate_frozen_research_parameters(config, config_hash)
    if not param_ok:
        errors.extend(param_errs)

    dataset_dir = os.path.join(REPO_ROOT, config.get("dataset_dir", "data/BSD68"))
    for img_id in assigned:
        p = os.path.join(dataset_dir, f"{img_id}.png")
        if not os.path.exists(p):
            errors.append(f"Dataset image missing: {p}")

    paths = get_team_directories(team_name)
    if not os.access(paths["raw"], os.W_OK):
        errors.append(f"Output directory not writable: {paths['raw']}")

    return (len(errors) == 0), errors


# ==============================================================================
# 5. Single-Image Evaluation Engine (Strict Research Standards)
# ==============================================================================
def execute_single_pilot_evaluation(
    image_id,
    noise_sigma,
    trial,
    solver_name,
    team_name,
    config,
    config_hash,
    code_hash,
    dataset_dir="data/BSD68",
    patch_callback=None,
):
    """
    Executes a single full-image solver evaluation conforming strictly to research methodology:
    - Runs COMPLETE image processing pipeline on all 37,604 patches (no shortcuts).
    - Obtains real, measured solver diagnostics (no hardcoded/fabricated values).
    - Calculates active-support ratio strictly per thesis definition: sum(|S_k|) / (63 * K).
    - Isolates pure iterative solve latency from setup, DCT, and aggregation.
    - Uses bitwise identical noise input across all 3 solvers for matched conditions.
    """
    img_filename = f"{image_id}.png" if not image_id.endswith(".png") else image_id
    img_path = os.path.join(REPO_ROOT, dataset_dir, img_filename)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # Load clean grayscale image in [0, 1]
    with Image.open(img_path) as pil_img:
        clean_img = np.asarray(pil_img.convert("L"), dtype=float) / 255.0

    base_seed = config.get("seed_protocol", {}).get("base_seed", 20260908)
    sensing_seed, noise_seed = compute_deterministic_seeds(image_id, noise_sigma, trial, base_seed)

    # Inject AWGN (Requirement 14: Matched noisy image generated from deterministic noise seed)
    noisy_img = add_awgn(clean_img, float(noise_sigma), seed=noise_seed)

    # Patch extraction (8x8, stride 2) and DCT projection: exactly 37,604 patches
    patch_records = extract_patches(clean_img, patch_size=8, stride=2)
    noisy_patch_records = extract_patches(noisy_img, patch_size=8, stride=2)
    total_patches = len(patch_records)

    noisy_thetas = np.array([dct_patch(r["patch"]).reshape(-1) for r in noisy_patch_records])
    clean_thetas = np.array([dct_patch(r["patch"]).reshape(-1) for r in patch_records])

    # Setup phase: generate sensing and precomputations (isolated from solve time)
    t_setup_start = time.perf_counter()
    if solver_name == "ASL-SR-DPT":
        # Validate sensing dimensions: M_ac=37, N_ac=63 (Requirement 13)
        A_ac = generate_dc_sensing(M_ac=37, N_ac=63, seed=sensing_seed)
        if A_ac.shape != (37, 63):
            raise ValueError(f"ASL-SR-DPT sensing matrix A_ac has invalid shape: {A_ac.shape}, expected (37, 63)")
        solver_dpt = HybridSparseSolverV7Optimized(
            A=A_ac,
            lambda_reg=config["asl_sr_dpt"]["lambda_reg"],
            tol=config["asl_sr_dpt"]["tol"],
        )
        Y_dc = noisy_thetas[:, 0]
        Y_ac = (A_ac @ noisy_thetas[:, 1:].T).T  # (P, 37)
        sensing_m = 37
        sensing_n = 63
        sensing_arch = "dc_preserving"
    else:
        # Standard sensing dimensions: M=38, N=64 (Requirement 13)
        A_std = generate_random_sensing(M=38, N=64, seed=sensing_seed)
        if A_std.shape != (38, 64):
            raise ValueError(f"Standard sensing matrix A_std has invalid shape: {A_std.shape}, expected (38, 64)")
        Y_std = (A_std @ noisy_thetas.T).T  # (P, 38)
        if solver_name == "OMP":
            col_norms = precompute_omp(A_std)
        elif solver_name == "LASSO-ADMM":
            L_cholesky = precompute_lasso_admm(A_std, rho=config["lasso_admm"]["rho"])
        sensing_m = 38
        sensing_n = 64
        sensing_arch = "standard"
    setup_time = time.perf_counter() - t_setup_start

    # Solve phase: iterative solve latency strictly measured (Requirement 18)
    t_solve_start = time.perf_counter()

    if solver_name == "ASL-SR-DPT":
        batch_size = config["asl_sr_dpt"].get("batch_size", 50)
        recovered_ac = []

        total_active_support_sum = 0
        total_dpt_iterations_sum = 0
        total_accepted_steps_sum = 0
        total_failed_ls_sum = 0
        final_sigmas = []

        for start in range(0, total_patches, batch_size):
            end = min(start + batch_size, total_patches)
            batch_len = end - start
            Y_sub = Y_ac[start:end].T
            Z_sub, diag_sub = solver_dpt.denoise_batch(
                Y_sub,
                sigma_min=config["asl_sr_dpt"]["sigma_min"],
                decrease_factor=config["asl_sr_dpt"]["sigma_decay"],
                max_iter=config["asl_sr_dpt"]["max_iter"],
                initial_mu=config["asl_sr_dpt"]["initial_mu"],
                armijo_c=config["asl_sr_dpt"]["armijo_c"],
                beta_decay=config["asl_sr_dpt"]["beta_decay"],
                max_backtracks=config["asl_sr_dpt"]["max_backtracks"],
                support_threshold_multiplier=config["asl_sr_dpt"]["support_threshold_multiplier"],
                support_reopen_interval=config["asl_sr_dpt"]["support_reopen_interval"],
                use_midpoint=config["asl_sr_dpt"]["use_midpoint"],
                return_diagnostics=True,
            )
            # Strict per-batch schema validation (Requirement 3)
            is_valid_diag, diag_err = validate_dpt_batch_diagnostics(diag_sub, batch_len)
            if not is_valid_diag:
                raise ValueError(f"ASL-SR-DPT batch diagnostics validation failed for patches [{start}:{end}]: {diag_err}")

            recovered_ac.append(Z_sub.T)

            # Accumulate real diagnostics (Requirement 1 & 2)
            for p_counts in diag_sub["patch_active_counts"]:
                total_active_support_sum += sum(p_counts)
                total_dpt_iterations_sum += len(p_counts)
            total_accepted_steps_sum += sum(diag_sub["patch_accepted_steps"])
            total_failed_ls_sum += sum(diag_sub["patch_failed_line_searches"])
            final_sigmas.extend(diag_sub["patch_final_sigma"])

            if patch_callback and (start % 2500 == 0 or end == total_patches):
                patch_callback(end, total_patches, f"ASL-SR-DPT: {end:,}/{total_patches:,} patches solved")

        solve_time = time.perf_counter() - t_solve_start

        Z_ac_final = np.vstack(recovered_ac)
        Z_final = np.empty((total_patches, 64), dtype=float)
        Z_final[:, 0] = Y_dc
        Z_final[:, 1:] = Z_ac_final

        residuals = np.linalg.norm(Z_ac_final @ A_ac.T - Y_ac, axis=1)
        y_norms = np.maximum(np.linalg.norm(Y_ac, axis=1), 1e-8)
        meas_res = float(np.mean(residuals))
        rel_res = float(np.mean(residuals / y_norms))
        norm_res = float(np.mean(residuals / np.sqrt(37.0)))

        # Actual executed iteration count and thesis-defined active-support ratio
        iter_count = float(total_dpt_iterations_sum) / float(total_patches)
        active_ratio = (
            float(total_active_support_sum) / (63.0 * float(total_dpt_iterations_sum))
            if total_dpt_iterations_sum > 0 else 1.0
        )
        mean_act_count = (
            float(total_active_support_sum) / float(total_dpt_iterations_sum)
            if total_dpt_iterations_sum > 0 else 63.0
        )
        final_sig = float(np.mean(final_sigmas))
        acc_steps = float(total_accepted_steps_sum) / float(total_patches)
        failed_ls = float(total_failed_ls_sum) / float(total_patches)

        supp_size = ""
        nonzero_count = float(np.mean(np.sum(np.abs(Z_final[:, 1:]) > 1e-4, axis=1)))
        admm_p = ""
        admm_d = ""

    elif solver_name == "OMP":
        recovered_thetas = np.empty((total_patches, 64), dtype=float)
        res_list = np.empty(total_patches, dtype=float)
        iters_list = np.empty(total_patches, dtype=float)

        for i in range(total_patches):
            th, diag = run_omp(
                y=Y_std[i],
                A=A_std,
                relative_residual_tol=config["omp"]["relative_residual_tol"],
                max_coefficients=config["omp"]["max_coefficients"],
                col_norms=col_norms,
                return_diagnostics=True,
            )
            recovered_thetas[i] = th
            res_list[i] = diag["final_residual"]
            iters_list[i] = diag["iterations"]

            if patch_callback and ((i + 1) % 2000 == 0 or i == total_patches - 1):
                patch_callback(i + 1, total_patches, f"OMP: {i + 1:,}/{total_patches:,} patches solved")

        solve_time = time.perf_counter() - t_solve_start
        Z_final = recovered_thetas

        y_norms = np.maximum(np.linalg.norm(Y_std, axis=1), 1e-8)
        meas_res = float(np.mean(res_list))
        rel_res = float(np.mean(res_list / y_norms))
        norm_res = float(np.mean(res_list / np.sqrt(38.0)))

        iter_count = float(np.mean(iters_list))
        supp_size = iter_count
        nonzero_count = supp_size

        # OMP does not have active_support_ratio (Requirement 11)
        active_ratio = ""
        mean_act_count = ""
        final_sig = ""
        acc_steps = ""
        failed_ls = ""
        admm_p = ""
        admm_d = ""

    elif solver_name == "LASSO-ADMM":
        recovered_thetas = np.empty((total_patches, 64), dtype=float)
        res_list = np.empty(total_patches, dtype=float)
        iters_list = np.empty(total_patches, dtype=float)
        pri_list = np.empty(total_patches, dtype=float)
        dual_list = np.empty(total_patches, dtype=float)

        for i in range(total_patches):
            th, diag = run_lasso_admm(
                y=Y_std[i],
                A=A_std,
                lambda_lasso=config["lasso_admm"]["lambda_lasso"],
                rho=config["lasso_admm"]["rho"],
                tol=config["lasso_admm"]["tol"],
                max_iter=config["lasso_admm"]["max_iter"],
                L=L_cholesky,
                return_diagnostics=True,
            )
            recovered_thetas[i] = th
            res_list[i] = diag["final_measurement_residual"]
            iters_list[i] = diag["iterations"]
            pri_list[i] = diag["final_primal_residual"]
            dual_list[i] = diag["final_dual_residual"]

            if patch_callback and ((i + 1) % 2000 == 0 or i == total_patches - 1):
                patch_callback(i + 1, total_patches, f"LASSO-ADMM: {i + 1:,}/{total_patches:,} patches solved")

        solve_time = time.perf_counter() - t_solve_start
        Z_final = recovered_thetas

        y_norms = np.maximum(np.linalg.norm(Y_std, axis=1), 1e-8)
        meas_res = float(np.mean(res_list))
        rel_res = float(np.mean(res_list / y_norms))
        norm_res = float(np.mean(res_list / np.sqrt(38.0)))

        iter_count = float(np.mean(iters_list))
        supp_size = ""
        nonzero_count = float(np.mean(np.sum(np.abs(Z_final) > 1e-4, axis=1)))

        # ADMM does not have active_support_ratio (Requirement 11)
        active_ratio = ""
        mean_act_count = ""
        final_sig = ""
        acc_steps = ""
        failed_ls = ""
        admm_p = float(np.mean(pri_list))
        admm_d = float(np.mean(dual_list))
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    # Spatial reconstruction via 2D IDCT and normalized 2D Hamming aggregation (all 37,604 patches)
    rec_patches = []
    for i in range(total_patches):
        p_spatial = idct_patch(Z_final[i].reshape(8, 8))
        rec_patches.append({
            "patch": p_spatial,
            "x": patch_records[i]["x"],
            "y": patch_records[i]["y"],
        })
    reconstructed_img = reconstruct_image(rec_patches, image_shape=clean_img.shape, patch_size=8)
    metrics = compute_all_metrics(clean_img, reconstructed_img)
    ms_patch = (solve_time / float(total_patches)) * 1000.0

    # Diagnostic errors (kept separate from primary performance metrics, Requirement 12)
    dc_err = float(np.mean(np.abs(Z_final[:, 0] - clean_thetas[:, 0])))
    ac_err = float(np.mean(np.linalg.norm(Z_final[:, 1:] - clean_thetas[:, 1:], axis=1)))

    run_id = f"{image_id}_s{int(noise_sigma)}_t{trial}_{solver_name.replace('-', '_')}"

    record = {
        "run_id": run_id,
        "image_id": str(image_id),
        "noise_level": float(noise_sigma),
        "noise_sigma": float(noise_sigma),
        "trial": int(trial),
        "solver": str(solver_name),
        "team": str(team_name),
        "sensing_seed": int(sensing_seed),
        "noise_seed": int(noise_seed),
        "psnr": round(float(metrics["psnr"]), 4),
        "ssim": round(float(metrics["ssim"]), 4),
        "mse": round(float(metrics["mse"]), 6),
        "setup_time": round(float(setup_time), 4),
        "solve_time": round(float(solve_time), 4),
        "milliseconds_per_patch": round(float(ms_patch), 4),
        "ms_per_patch": round(float(ms_patch), 4),
        "iteration_count": round(float(iter_count), 2),
        "measurement_residual": round(float(meas_res), 4),
        "relative_residual": round(float(rel_res), 4),
        "normalized_residual": round(float(norm_res), 4),
        "active_support_ratio": round(float(active_ratio), 6) if active_ratio != "" else "",
        "mean_active_support_count": round(float(mean_act_count), 2) if mean_act_count != "" else "",
        "final_sigma": round(float(final_sig), 6) if final_sig != "" else "",
        "accepted_steps": round(float(acc_steps), 2) if acc_steps != "" else "",
        "failed_line_searches": round(float(failed_ls), 2) if failed_ls != "" else "",
        "support_size": round(float(supp_size), 2) if supp_size != "" else "",
        "nonzero_coefficient_count": round(float(nonzero_count), 2) if nonzero_count != "" else "",
        "primal_residual": round(float(admm_p), 6) if admm_p != "" else "",
        "dual_residual": round(float(admm_d), 6) if admm_d != "" else "",
        "sensing_m": int(sensing_m),
        "sensing_n": int(sensing_n),
        "sensing_architecture": str(sensing_arch),
        "diag_dc_error": round(float(dc_err), 6),
        "diag_ac_error": round(float(ac_err), 6),
        "config_hash": config_hash,
        "code_version": config.get("version", "ASL-SR-DPT-PILOT-1.0"),
        "code_hash": code_hash,
        "timestamp": datetime.now().isoformat(),
    }

    # Validate record before returning (Requirement 15)
    assigned_images = config.get("team_assignments", {}).get(team_name, [image_id])
    valid_solvers = config.get("solvers", ["ASL-SR-DPT", "OMP", "LASSO-ADMM"])
    valid_noise_levels = config.get("noise_levels", [15.0, 25.0, 50.0])
    valid_trials = config.get("trials", [1, 2, 3, 4, 5])

    is_valid, err_msg = validate_evaluation_record(
        record, assigned_images, valid_solvers, valid_noise_levels, valid_trials, config_hash
    )
    if not is_valid:
        raise ValueError(f"Evaluation record failed integrity validation: {err_msg}")

    return record, reconstructed_img, noisy_img, Z_final


# ==============================================================================
# 6. True Reproducibility Test Suite (Gate 3)
# ==============================================================================
def run_reproducibility_test(config, config_hash, code_hash, progress_callback=None):
    """
    Runs fixed condition (test001, sigma=15.0, trial=1) twice for all 3 solvers
    using the COMPLETE full-image processing pipeline (Requirements 3 & 4).
    Compares reconstructed image arrays and coefficient arrays with byte-level and tolerance checks.
    Classifies each solver as EXACT_REPRODUCTION or NUMERICALLY_REPRODUCIBLE.
    Supports optional progress_callback(run_num, total_runs, solver, sub_run, p_cur, p_tot, msg).
    """
    test_img = "test001"
    test_sigma = 15.0
    test_trial = 1
    solvers = ["ASL-SR-DPT", "OMP", "LASSO-ADMM"]
    results = []
    total_runs = 6
    run_num = 0

    for s in solvers:
        # First execution (Run 1)
        run_num += 1
        if progress_callback:
            progress_callback(run_num, total_runs, s, 1, 0, 37604, f"Starting {s} (Run 1/2)...")

        def make_cb1(r_n, s_name):
            def cb(p_cur, p_tot, msg):
                if progress_callback:
                    progress_callback(r_n, total_runs, s_name, 1, p_cur, p_tot, msg)
            return cb

        rec1, img1, _, z1 = execute_single_pilot_evaluation(
            test_img, test_sigma, test_trial, s, "Verification", config, config_hash, code_hash,
            patch_callback=make_cb1(run_num, s)
        )
        if progress_callback:
            progress_callback(run_num, total_runs, s, 1, 37604, 37604, f"Completed {s} Run 1/2 in {rec1['solve_time']:.1f}s | PSNR: {rec1['psnr']:.2f} dB")

        # Second execution (Run 2)
        run_num += 1
        if progress_callback:
            progress_callback(run_num, total_runs, s, 2, 0, 37604, f"Starting {s} (Run 2/2 duplicate solve)...")

        def make_cb2(r_n, s_name):
            def cb(p_cur, p_tot, msg):
                if progress_callback:
                    progress_callback(r_n, total_runs, s_name, 2, p_cur, p_tot, msg)
            return cb

        rec2, img2, _, z2 = execute_single_pilot_evaluation(
            test_img, test_sigma, test_trial, s, "Verification", config, config_hash, code_hash,
            patch_callback=make_cb2(run_num, s)
        )
        if progress_callback:
            progress_callback(run_num, total_runs, s, 2, 37604, 37604, f"Completed {s} Run 2/2 in {rec2['solve_time']:.1f}s | PSNR: {rec2['psnr']:.2f} dB")

        # Array byte comparisons and SHA-256 hashes
        img1_bytes = np.ascontiguousarray(img1, dtype=np.float64).tobytes()
        img2_bytes = np.ascontiguousarray(img2, dtype=np.float64).tobytes()
        img_hash1 = hashlib.sha256(img1_bytes).hexdigest()
        img_hash2 = hashlib.sha256(img2_bytes).hexdigest()

        z1_bytes = np.ascontiguousarray(z1, dtype=np.float64).tobytes()
        z2_bytes = np.ascontiguousarray(z2, dtype=np.float64).tobytes()
        z_hash1 = hashlib.sha256(z1_bytes).hexdigest()
        z_hash2 = hashlib.sha256(z2_bytes).hexdigest()

        bytes_match = (img1_bytes == img2_bytes) and (z1_bytes == z2_bytes)
        max_img_diff = float(np.max(np.abs(img1 - img2)))
        max_z_diff = float(np.max(np.abs(z1 - z2)))

        psnr_diff = abs(rec1["psnr"] - rec2["psnr"])
        ssim_diff = abs(rec1["ssim"] - rec2["ssim"])
        mse_diff = abs(rec1["mse"] - rec2["mse"])
        res_diff = abs(rec1["measurement_residual"] - rec2["measurement_residual"])

        # Two formal classifications (Requirement 4)
        if bytes_match:
            classification = "EXACT_REPRODUCTION"
            passed = True
        elif (max_img_diff < 1e-5 and max_z_diff < 1e-5 and psnr_diff < 1e-4 and ssim_diff < 1e-4 and mse_diff < 1e-6):
            classification = "NUMERICALLY_REPRODUCIBLE"
            passed = True
        else:
            classification = "FAILED"
            passed = False

        if progress_callback:
            progress_callback(run_num, total_runs, s, 2, 37604, 37604, f">>> {s} Result: {classification} (Bytes match: {bytes_match}, Max Δ: {max_img_diff:.2e})")

        results.append({
            "solver": s,
            "classification": classification,
            "passed": passed,
            "bytes_match": bytes_match,
            "image_sha256": img_hash1 if bytes_match else f"{img_hash1[:8]}.. / {img_hash2[:8]}..",
            "coeff_sha256": z_hash1 if (z1_bytes == z2_bytes) else f"{z_hash1[:8]}.. / {z_hash2[:8]}..",
            "max_image_diff": max_img_diff,
            "max_coeff_diff": max_z_diff,
            "psnr_run1": rec1["psnr"],
            "psnr_run2": rec2["psnr"],
            "psnr_diff": psnr_diff,
            "ssim_diff": ssim_diff,
            "mse_diff": mse_diff,
            "res_diff": res_diff,
        })
    return results


# ==============================================================================
# 7. Team Report Generator & Integrity Audit (Requirements 16 & 17)
# ==============================================================================
def generate_team_report(team_name, raw_csv_path, report_path, config, config_hash, code_hash, gates_status, repro_status):
    """
    Produces team_report.md upon completion of all 225 evaluations.
    Enforces strict completion criteria and appends formal Pilot Integrity Summary (Requirements 16 & 17).
    Does NOT report premature scientific superiority.
    """
    if not os.path.exists(raw_csv_path):
        return
    df = pd.read_csv(raw_csv_path)
    env = get_environment_info()
    assigned_images = config.get("team_assignments", {}).get(team_name, [])

    total_expected = 225
    completed = len(df)
    unique_keys = set(zip(df["image_id"], df["noise_sigma"], df["trial"], df["solver"]))
    duplicates = completed - len(unique_keys)

    # Completion validation criteria (Requirement 16)
    is_completed = (
        completed == 225
        and len(unique_keys) == 225
        and duplicates == 0
        and len(df["solver"].unique()) == 3
        and len(df["noise_sigma"].unique()) == 3
        and len(df["trial"].unique()) == 5
        and len(df["image_id"].unique()) == 5
    )
    status_label = "COMPLETED" if is_completed else "INCOMPLETE OR INVALID"

    paths = get_team_directories(team_name)
    failures_count = 0
    if os.path.exists(paths["failures_csv"]):
        try:
            failures_count = len(pd.read_csv(paths["failures_csv"]))
        except Exception:
            pass

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# ASL-SR-DPT Pilot Benchmark: Team Report ({team_name})\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Configuration Hash:** `{config_hash}`  \n")
        f.write(f"**Codebase Hash:** `{code_hash}`  \n")
        f.write(f"**Status:** `{status_label}`  \n\n")
        f.write("---\n\n")

        f.write("## 1. Workload Summary\n\n")
        f.write(f"- **Team Identifier:** {team_name}\n")
        f.write(f"- **Assigned Images:** {', '.join(assigned_images)}\n")
        f.write(f"- **Expected Evaluations:** {total_expected}\n")
        f.write(f"- **Completed Evaluations:** {completed}\n")
        f.write(f"- **Unique Evaluations:** {len(unique_keys)}\n")
        f.write(f"- **Duplicate Count:** {duplicates}\n")
        f.write(f"- **Total Recorded Solve Time:** {df['solve_time'].sum():.2f} seconds\n\n")

        f.write("## 2. Performance Summary by Solver and Noise Level\n\n")
        summary_table = []
        for (s, n), grp in df.groupby(["solver", "noise_sigma"]):
            act_rat_val = pd.to_numeric(grp["active_support_ratio"], errors="coerce").mean()
            act_rat_str = f"{act_rat_val:.4f}" if pd.notna(act_rat_val) else "N/A"

            summary_table.append({
                "Solver": s,
                "Noise (σ)": n,
                "Count": len(grp),
                "Mean PSNR (dB)": f"{grp['psnr'].mean():.4f}",
                "Mean SSIM": f"{grp['ssim'].mean():.4f}",
                "Mean MSE": f"{grp['mse'].mean():.6f}",
                "Mean Solve Time (s)": f"{grp['solve_time'].mean():.2f}",
                "ms / patch": f"{grp['ms_per_patch'].mean():.2f}",
                "Mean Residual": f"{grp['measurement_residual'].mean():.4f}",
                "Mean Iterations": f"{grp['iteration_count'].mean():.1f}",
                "Active Support Ratio": act_rat_str,
            })
        summary_df = pd.DataFrame(summary_table)
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 3. ASL-SR-DPT Algorithmic Metrics\n\n")
        dpt_df = df[df["solver"] == "ASL-SR-DPT"]
        if not dpt_df.empty:
            mean_act = pd.to_numeric(dpt_df["active_support_ratio"], errors="coerce").mean()
            mean_act_str = f"{mean_act:.6f}" if pd.notna(mean_act) else "N/A"
            mean_supp_cnt = pd.to_numeric(dpt_df["mean_active_support_count"], errors="coerce").mean()
            mean_supp_str = f"{mean_supp_cnt:.2f}" if pd.notna(mean_supp_cnt) else "N/A"

            f.write(f"- **Mean Active-Support Ratio (Thesis Def: sum|S_k| / 63K):** {mean_act_str}\n")
            f.write(f"- **Mean Active-Support Count per Iteration:** {mean_supp_str} / 63\n")
            f.write(f"- **Mean Executed Iterations:** {dpt_df['iteration_count'].mean():.2f}\n")
            f.write(f"- **Mean Accepted Steps:** {pd.to_numeric(dpt_df['accepted_steps'], errors='coerce').mean():.2f}\n")
            f.write(f"- **Mean Failed Line Searches:** {pd.to_numeric(dpt_df['failed_line_searches'], errors='coerce').mean():.2f}\n")
            f.write(f"- **Diagnostic DC Error:** {dpt_df['diag_dc_error'].mean():.6f}\n")
            f.write(f"- **Diagnostic AC Error:** {dpt_df['diag_ac_error'].mean():.6f}\n\n")

        f.write("## 4. Host Environment Information\n\n")
        f.write(f"- **Operating System:** {env['os']}\n")
        f.write(f"- **Python Version:** {env['python_version']}\n")
        f.write(f"- **NumPy Version:** {env['numpy_version']}\n")
        f.write(f"- **SciPy Version:** {env['scipy_version']}\n")
        f.write(f"- **Pandas Version:** {env['pandas_version']}\n")
        f.write(f"- **Processor:** {env['cpu']}\n")
        f.write(f"- **Thread Locking Settings:** {env['thread_settings']}\n\n")

        # Pilot Integrity Summary (Requirement 17)
        f.write("## 5. Pilot Integrity Summary\n\n")
        f.write(f"- **Expected evaluations:** {total_expected}\n")
        f.write(f"- **Completed evaluations:** {completed}\n")
        f.write(f"- **Unique evaluations:** {len(unique_keys)}\n")
        f.write(f"- **Duplicate evaluations:** {duplicates}\n")
        f.write(f"- **Failed evaluations:** {failures_count}\n")
        f.write(f"- **Missing evaluations:** {total_expected - len(unique_keys)}\n\n")

        f.write(f"- **Configuration hash:** `{config_hash}`\n")
        f.write(f"- **Code version:** `{config.get('version', 'ASL-SR-DPT-PILOT-1.0')}`\n")
        f.write(f"- **Codebase hash:** `{code_hash}`\n")
        f.write(f"- **Assigned images:** {', '.join(assigned_images)}\n\n")

        f.write("### Safety Gates Status:\n")
        for g_name, g_val in gates_status.items():
            f.write(f"- **{g_name}:** {'PASS' if g_val else 'FAIL'}\n")

        f.write("\n### Reproducibility Status:\n")
        for s_name, s_val in repro_status.items():
            f.write(f"- **{s_name}:** {s_val}\n")

        f.write("\n---\n\n")
        f.write("> **Academic Integrity Note:** This report contains observational measurements only. ")
        f.write("Hypothesis testing, statistical significance tests, and definitive scientific comparisons ")
        f.write("are strictly deferred until both Team A and Team B datasets are centrally merged and audited via `merge_pilot_results.py`.\n")


# ==============================================================================
# 8. Streamlit GUI Presentation Layer
# ==============================================================================
def main():
    st.set_page_config(
        page_title="ASL-SR-DPT Pilot Benchmark",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Glass Minimalism styling
    st.markdown("""
        <style>
            /* Base typography: Use native system font stack for maximum crispness and zero layout shift */
            .stApp, .main-title, .sub-title, .workload-bar, .glass-panel, .glass-status-card {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            }

            /* Strictly preserve Material Symbols and Streamlit icon fonts */
            [data-testid*="Icon"],
            [data-testid*="icon"],
            [data-testid="stExpanderToggleIcon"],
            .material-symbols-rounded,
            .material-symbols-outlined,
            .material-icons,
            [class*="material-symbols"] {
                font-family: 'Material Symbols Rounded', 'Material Icons' !important;
                font-style: normal !important;
                font-weight: 400 !important;
                letter-spacing: normal !important;
                text-transform: none !important;
                display: inline-block !important;
                white-space: nowrap !important;
                word-wrap: normal !important;
                direction: ltr !important;
                -webkit-font-smoothing: antialiased !important;
            }

            /* Header Typography */
            .main-title {
                font-size: 2.05rem;
                font-weight: 700;
                letter-spacing: -0.035em;
                color: #f8fafc;
                margin-bottom: 0.25rem;
                line-height: 1.25;
            }
            .sub-title {
                font-size: 0.95rem;
                font-weight: 400;
                letter-spacing: -0.01em;
                color: #94a3b8;
                margin-bottom: 1.5rem;
                line-height: 1.45;
            }

            /* Frosted Glass Workload Bar */
            .workload-bar {
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 20px;
                background: rgba(255, 255, 255, 0.04);
                backdrop-filter: blur(16px) saturate(180%);
                -webkit-backdrop-filter: blur(16px) saturate(180%);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 14px 22px;
                margin-bottom: 22px;
                box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.2);
            }
            .workload-col {
                display: flex;
                flex-direction: column;
                gap: 3px;
            }
            .workload-label {
                font-size: 0.72rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.07em;
                color: #94a3b8;
            }
            .workload-value {
                font-size: 0.92rem;
                font-weight: 600;
                color: #f8fafc;
            }
            .workload-divider {
                height: 30px;
                width: 1px;
                background: rgba(255, 255, 255, 0.1);
            }

            /* Glass Pill Badges */
            .glass-pill {
                display: inline-flex;
                align-items: center;
                padding: 3px 12px;
                border-radius: 6px;
                font-size: 0.8rem;
                font-weight: 600;
                letter-spacing: 0.03em;
                background: rgba(59, 130, 246, 0.2);
                color: #60a5fa;
                border: 1px solid rgba(59, 130, 246, 0.4);
            }

            /* Sidebar Glass Panel */
            .glass-panel {
                background: rgba(255, 255, 255, 0.04);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 10px;
                padding: 14px 16px;
                margin-bottom: 14px;
            }

            /* Metric Containers Override */
            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.04) !important;
                backdrop-filter: blur(14px) saturate(180%) !important;
                -webkit-backdrop-filter: blur(14px) saturate(180%) !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                border-radius: 12px !important;
                padding: 16px 20px !important;
                box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.2) !important;
            }
            div[data-testid="stMetricLabel"] p {
                font-size: 0.75rem !important;
                font-weight: 600 !important;
                text-transform: uppercase !important;
                letter-spacing: 0.06em !important;
                color: #94a3b8 !important;
            }
            div[data-testid="stMetricValue"] {
                font-size: 1.85rem !important;
                font-weight: 700 !important;
                color: #f8fafc !important;
                letter-spacing: -0.02em !important;
            }

            /* Glass Tabs */
            div[data-testid="stTabs"] [data-baseweb="tab-list"] {
                background: rgba(255, 255, 255, 0.03) !important;
                backdrop-filter: blur(10px) !important;
                border-radius: 10px !important;
                padding: 4px !important;
                gap: 6px !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                margin-bottom: 20px !important;
            }
            div[data-testid="stTabs"] [data-baseweb="tab"] {
                border-radius: 8px !important;
                padding: 8px 18px !important;
                font-size: 0.88rem !important;
                font-weight: 500 !important;
                color: #94a3b8 !important;
                border: none !important;
                background: transparent !important;
                transition: all 0.15s ease !important;
            }
            div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
                color: #f8fafc !important;
                background: rgba(255, 255, 255, 0.05) !important;
            }
            div[data-testid="stTabs"] [aria-selected="true"] {
                background: rgba(255, 255, 255, 0.12) !important;
                color: #f8fafc !important;
                font-weight: 600 !important;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15) !important;
            }

            /* Glass Status Cards (Execution & Progress) */
            .glass-status-card {
                background: rgba(255, 255, 255, 0.04);
                backdrop-filter: blur(16px) saturate(180%);
                -webkit-backdrop-filter: blur(16px) saturate(180%);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-left: 3px solid #3b82f6 !important;
                border-radius: 10px;
                padding: 16px 20px;
                margin-bottom: 16px;
                box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.2);
            }
            .glass-status-title {
                font-size: 1.0rem;
                font-weight: 600;
                color: #f8fafc;
                letter-spacing: -0.01em;
            }
            .glass-status-meta {
                margin-top: 6px;
                font-size: 0.88rem;
                color: #cbd5e1;
            }
            .glass-status-detail {
                margin-top: 4px;
                font-size: 0.82rem;
                color: #94a3b8;
            }

            /* Clean Expanders - Preserve summary & icons untouched */
            div[data-testid="stExpander"] {
                background: rgba(255, 255, 255, 0.03) !important;
                backdrop-filter: blur(10px) !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                border-radius: 10px !important;
                margin-bottom: 14px !important;
            }

            /* Minimal Buttons */
            div.stButton > button {
                border-radius: 8px !important;
                font-weight: 600 !important;
                font-size: 0.9rem !important;
                letter-spacing: 0.01em !important;
            }

            /* Light theme support */
            @media (prefers-color-scheme: light) {
                .main-title { color: #0f172a; }
                .sub-title { color: #475569; }
                .workload-bar {
                    background: rgba(0, 0, 0, 0.02);
                    border: 1px solid rgba(0, 0, 0, 0.08);
                }
                .workload-label { color: #64748b; }
                .workload-value { color: #0f172a; }
                .workload-divider { background: rgba(0, 0, 0, 0.08); }
                div[data-testid="stMetric"] {
                    background: rgba(0, 0, 0, 0.02) !important;
                    border: 1px solid rgba(0, 0, 0, 0.08) !important;
                }
                div[data-testid="stMetricLabel"] p { color: #64748b !important; }
                div[data-testid="stMetricValue"] { color: #0f172a !important; }
                div[data-testid="stTabs"] [data-baseweb="tab-list"] {
                    background: rgba(0, 0, 0, 0.03) !important;
                    border: 1px solid rgba(0, 0, 0, 0.06) !important;
                }
                div[data-testid="stTabs"] [data-baseweb="tab"] { color: #64748b !important; }
                div[data-testid="stTabs"] [aria-selected="true"] {
                    background: #ffffff !important;
                    color: #0f172a !important;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
                }
                .glass-panel {
                    background: rgba(0, 0, 0, 0.02);
                    border: 1px solid rgba(0, 0, 0, 0.08);
                }
                .glass-status-card {
                    background: rgba(0, 0, 0, 0.02);
                    border: 1px solid rgba(0, 0, 0, 0.08);
                    border-left: 3px solid #2563eb !important;
                }
                .glass-status-title { color: #0f172a; }
                .glass-status-meta { color: #334155; }
                .glass-status-detail { color: #64748b; }
            }
        </style>
    """, unsafe_allow_html=True)

    # Initial team selection via CLI or Session State
    cli_team = "Team A"
    if "--team" in sys.argv:
        try:
            t_idx = sys.argv.index("--team") + 1
            if t_idx < len(sys.argv):
                val = sys.argv[t_idx].upper()
                if "B" in val.split() or val.endswith("B"):
                    cli_team = "Team B"
                else:
                    cli_team = "Team A"
        except Exception:
            pass

    if "selected_team" not in st.session_state:
        st.session_state.selected_team = cli_team
    if "reproducibility_passed" not in st.session_state:
        st.session_state.reproducibility_passed = False
        st.session_state.reproducibility_results = []
        rep_file = os.path.join(REPO_ROOT, "results", "reproducibility_validation_report.json")
        if os.path.exists(rep_file):
            try:
                with open(rep_file, "r", encoding="utf-8") as rf:
                    rep_data = json.load(rf)
                if (
                    rep_data.get("config_hash") == CONFIG_HASH
                    and rep_data.get("code_hash") == CODE_HASH
                    and rep_data.get("overall_passed") is True
                ):
                    st.session_state.reproducibility_passed = True
                    st.session_state.reproducibility_results = rep_data.get("results", [])
            except Exception:
                pass
    if "reproducibility_results" not in st.session_state:
        st.session_state.reproducibility_results = []
    if "benchmark_running" not in st.session_state:
        st.session_state.benchmark_running = False

    # Sidebar setup
    with st.sidebar:
        st.markdown("### Pilot Control Center")
        st.markdown(f"""
        <div class="glass-panel">
            <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; color: #94a3b8; font-weight: 600;">Configuration</div>
            <div style="font-size: 0.92rem; font-weight: 600; color: #f8fafc; margin-top: 3px;">ASL_SR_DPT_FINAL</div>
            <div style="font-size: 0.75rem; color: #cbd5e1; font-family: ui-monospace, monospace; margin-top: 6px;">Config: <span style="color: #60a5fa;">{CONFIG_HASH[:12]}...</span></div>
            <div style="font-size: 0.75rem; color: #cbd5e1; font-family: ui-monospace, monospace; margin-top: 3px;">Code: <span style="color: #a78bfa;">{CODE_HASH[:12]}...</span></div>
        </div>
        """, unsafe_allow_html=True)

        team_options = ["Team A", "Team B"]
        default_index = 0 if st.session_state.selected_team == "Team A" else 1
        selected_team = st.radio(
            "Select Assigned Team Workload:",
            options=team_options,
            index=default_index,
            key="team_radio_selector",
            help="Strictly allocates pre-assigned non-overlapping 225-evaluation workload."
        )
        st.session_state.selected_team = selected_team
        paths = get_team_directories(selected_team)

        st.markdown("---")
        st.markdown("### Certified Host Environment")
        env_info = get_environment_info()
        st.markdown(f"""
        <div class="glass-panel" style="font-size: 0.82rem; line-height: 1.85;">
            <div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">OS:</span> <b style="color: #f8fafc;">{platform.system()} {platform.release()}</b></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">Python:</span> <code style="color: #34d399;">{env_info['python_version']}</code></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">NumPy:</span> <code style="color: #34d399;">{env_info['numpy_version']}</code></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">SciPy:</span> <code style="color: #34d399;">{env_info['scipy_version']}</code></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">Pandas:</span> <code style="color: #34d399;">{env_info['pandas_version']}</code></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">OMP Threads:</span> <code style="color: #38bdf8;">{env_info['thread_settings']['OMP_NUM_THREADS']}</code></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #94a3b8;">MKL Threads:</span> <code style="color: #38bdf8;">{env_info['thread_settings']['MKL_NUM_THREADS']}</code></div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Single-threaded execution verified.")

    # Header section
    st.markdown('<div class="main-title">ASL-SR-DPT Distributed Pilot Benchmark</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Accelerated Single-Loop Sparse Recovery with Dynamic Parameter Tuning for Mobile Image Denoising</div>',
        unsafe_allow_html=True,
    )

    assigned_images = PILOT_CONFIG.get("team_assignments", {}).get(selected_team, [])

    # Load existing keys with code/config consistency validation (Requirement 19)
    try:
        existing_keys = load_existing_composite_keys(paths["raw_csv"], CONFIG_HASH, CODE_HASH)
    except ValueError as val_err:
        st.error(f"Integrity Error: {val_err}")
        existing_keys = set()

    completed_count = len(existing_keys)

    # Top info banner
    st.markdown(f"""
    <div class="workload-bar">
        <div class="workload-col">
            <span class="workload-label">Active Track</span>
            <span class="glass-pill">{selected_team}</span>
        </div>
        <div class="workload-divider"></div>
        <div class="workload-col">
            <span class="workload-label">Assigned Images</span>
            <span class="workload-value" style="font-family: monospace;">{', '.join(assigned_images)}</span>
        </div>
        <div class="workload-divider"></div>
        <div class="workload-col">
            <span class="workload-label">Target Workload</span>
            <span class="workload-value">225 evaluations (5 × 3 × 5 × 3)</span>
        </div>
        <div class="workload-divider"></div>
        <div class="workload-col">
            <span class="workload-label">Completed</span>
            <span class="workload-value" style="font-family: monospace;">{completed_count} / 225</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tab navigation
    tab_dash, tab_gates, tab_report, tab_export = st.tabs([
        "Live Benchmark Dashboard",
        "Pilot Safety Gates (Steps 1–5)",
        "Team Summary & Report",
        "Export & Merge Center",
    ])

    # ==========================================================================
    # TAB 2: Safety Gates (Steps 1-5)
    # ==========================================================================
    with tab_gates:
        st.subheader("Pilot Benchmark Pre-Run Safety Gates")
        st.write("All 5 verification gates must pass before starting the 225-evaluation workload.")

        # Gate 1: Environment Verification
        g1_pass = (
            int(platform.python_version_tuple()[0]) >= 3
            and int(platform.python_version_tuple()[1]) >= 10
            and os.environ.get("OMP_NUM_THREADS") == "1"
            and os.environ.get("OPENBLAS_NUM_THREADS") == "1"
            and os.environ.get("MKL_NUM_THREADS") == "1"
        )
        with st.expander("STEP 1: Environment Verification", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Python Runtime:** {platform.python_version()} (>= 3.10 required)")
                st.write(f"**NumPy Version:** {np.__version__}")
                st.write(f"**SciPy Version:** {scipy.__version__}")
                st.write(f"**Pandas Version:** {pd.__version__}")
            with col2:
                st.write(f"**Thread Locking:** OMP={os.environ.get('OMP_NUM_THREADS')}, MKL={os.environ.get('MKL_NUM_THREADS')}, OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}")
                st.write(f"**Host OS:** {platform.platform()}")
                st.write(f"**CPU Architecture:** {env_info['cpu']}")
            if g1_pass:
                st.success("STEP 1 PASSED: Environment is verified and single-threaded execution is locked.")
            else:
                st.error("STEP 1 FAILED: Single-threaded execution flags or Python runtime constraint violated.")

        # Gate 2: Strengthened Configuration Verification (Requirement 8)
        g2_pass, g2_errors = validate_frozen_research_parameters(PILOT_CONFIG, CONFIG_HASH)
        with st.expander("STEP 2: Configuration Integrity Verification", expanded=True):
            st.write(f"**Configuration File:** `configs/pilot_config.json`")
            st.write(f"**Configuration SHA-256 Digest:** `{CONFIG_HASH}`")
            st.write(f"**Codebase Integrity SHA-256:** `{CODE_HASH}`")
            st.write(f"**Sensing Dimensions:** $M=38, N=64$; AC Sensing: $M_{{ac}}=37, N_{{ac}}=63$")
            st.write(f"**Regularization Parameters:** $\\lambda_{{DPT}}=0.1, \\lambda_{{LASSO}}=0.01$")
            if g2_pass:
                st.success("STEP 2 PASSED: All frozen research parameters match the exact thesis specification.")
            else:
                for err in g2_errors:
                    st.error(f"Config Error: {err}")

        # Gate 3: Reproducibility Test (Requirements 3 & 4)
        with st.expander("STEP 3: Full-Image Deterministic Reproducibility Verification", expanded=True):
            st.write("Executes `test001` ($\\sigma=15.0$, Trial 1) twice for ASL-SR-DPT, OMP, and LASSO-ADMM across all 37,604 patches.")
            st.write("Verifies byte-level array matching and generates SHA-256 digests.")

            btn_label = "Run Full-Image Reproducibility Verification" if not st.session_state.reproducibility_passed else "Re-run Reproducibility Verification"
            run_clicked = st.button(btn_label, key="btn_repro", type="primary" if not st.session_state.reproducibility_passed else "secondary")

            if run_clicked:
                status_placeholder = st.empty()
                overall_bar = st.progress(0.0)
                patch_bar = st.progress(0.0)

                st.markdown("##### Live Execution Terminal Log")
                terminal_box = st.empty()
                terminal_logs = []
                t_repro_start = time.time()

                def log_term(msg):
                    ts = datetime.now().strftime("%H:%M:%S")
                    line = f"[{ts}] {msg}"
                    terminal_logs.append(line)
                    terminal_box.code("\n".join(terminal_logs[-30:]), language="bash")

                log_term("=== Initiating Gate 3 Deterministic Reproducibility Verification ===")
                log_term("Reference Image: test001 (512x512) -> 37,604 patches (8x8, stride 2) | Noise: sigma=15.0 | Trial: 1")
                log_term("Scope: 3 Solvers x 2 Executions = 6 Full-Image Solves (225,624 patches total)")

                def repro_ui_callback(run_num, total_runs, solver, sub_run, p_cur, p_tot, msg):
                    overall_frac = min(1.0, max(0.0, (run_num - 1 + (p_cur / float(p_tot))) / float(total_runs)))
                    patch_frac = min(1.0, max(0.0, p_cur / float(p_tot)))
                    overall_pct = overall_frac * 100.0
                    patch_pct = patch_frac * 100.0

                    elapsed = time.time() - t_repro_start
                    if overall_frac > 0.02:
                        est_total = elapsed / overall_frac
                        rem = max(0.0, est_total - elapsed)
                        rem_str = f"{int(rem // 60):02d}m {int(rem % 60):02d}s"
                    else:
                        rem_str = "estimating..."

                    status_placeholder.markdown(f"""
                    <div class="glass-status-card">
                        <div class="glass-status-title">
                            Step {run_num} of {total_runs}: {solver} (Execution {sub_run}/2)
                        </div>
                        <div class="glass-status-meta">
                            <b>Overall Progress:</b> <span style="color: #60a5fa; font-weight: 600;">{overall_pct:.1f}%</span> &nbsp;|&nbsp; 
                            <b>Current Patch:</b> <span style="font-weight: 600;">{p_cur:,} / {p_tot:,}</span> ({patch_pct:.1f}%)
                        </div>
                        <div class="glass-status-detail">
                            <b>Elapsed:</b> {int(elapsed // 60):02d}m {int(elapsed % 60):02d}s &nbsp;|&nbsp; 
                            <b>Estimated Remaining:</b> ~{rem_str} &nbsp;|&nbsp; 
                            <span>{msg}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    overall_bar.progress(overall_frac, text=f"Overall Verification: {overall_pct:.1f}% (Step {run_num} of {total_runs})")
                    patch_bar.progress(patch_frac, text=f"Current Solver ({solver}): {p_cur:,} / {p_tot:,} patches ({patch_pct:.1f}%)")

                    if msg and (msg.startswith("Starting") or msg.startswith("Completed") or msg.startswith(">>>") or msg.startswith("===")):
                        log_term(msg)

                try:
                    repro_results = run_reproducibility_test(
                        PILOT_CONFIG, CONFIG_HASH, CODE_HASH, progress_callback=repro_ui_callback
                    )
                    all_passed = all(r["passed"] for r in repro_results)
                    st.session_state.reproducibility_passed = all_passed
                    st.session_state.reproducibility_results = repro_results

                    rep_path = os.path.join(REPO_ROOT, "results", "reproducibility_validation_report.json")
                    with open(rep_path, "w", encoding="utf-8") as rf:
                        json.dump({
                            "timestamp": datetime.now().isoformat(),
                            "config_hash": CONFIG_HASH,
                            "code_hash": CODE_HASH,
                            "elapsed_seconds": time.time() - t_repro_start,
                            "overall_passed": all_passed,
                            "results": repro_results,
                        }, rf, indent=2)

                    log_term(f"=== Verification Complete! All Passed: {all_passed} ===")
                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    st.error(f"Error during reproducibility verification: {e}")
                    log_term(f"[ERROR] {e}")

            if st.session_state.reproducibility_passed:
                st.success("STEP 3 PASSED: All solvers verified reproducible under repeated seeds.")
                if st.session_state.reproducibility_results:
                    st.dataframe(pd.DataFrame(st.session_state.reproducibility_results), use_container_width=True)
            else:
                st.info("STEP 3 PENDING: Reproducibility check not yet executed or failed. Click above to run verification.")

        # Gate 4: Workload & Dataset Verification (Requirement 9)
        g4_team_ok, g4_team_errs = validate_team_assignment_integrity(PILOT_CONFIG)
        g4_missing_images = []
        for img_id in assigned_images:
            p = os.path.join(REPO_ROOT, "data", "BSD68", f"{img_id}.png")
            if not os.path.exists(p):
                g4_missing_images.append(img_id)
        g4_pass = g4_team_ok and (len(g4_missing_images) == 0) and os.access(paths["raw"], os.W_OK)

        with st.expander("STEP 4: Workload & Storage Verification", expanded=True):
            st.write(f"**Assigned Images:** {', '.join(assigned_images)}")
            st.write(f"**Output Directory Writable:** `{paths['base']}`")
            if not g4_team_ok:
                for err in g4_team_errs:
                    st.error(f"Team Integrity Error: {err}")
            elif len(g4_missing_images) > 0:
                st.error(f"Missing images in data/BSD68: {g4_missing_images}")
            else:
                st.success("STEP 4 PASSED: Non-overlapping workload assignment and dataset integrity certified.")

        # Gate 5: Final Experiment Integrity Verification (Requirement 7)
        g5_pass, g5_errors = validate_gate_5_experiment_integrity(selected_team, PILOT_CONFIG, CONFIG_HASH, CODE_HASH)
        with st.expander("STEP 5: Final Experiment Integrity Verification", expanded=True):
            st.write("Verifies that all 11 experiment integrity preconditions are satisfied:")
            st.write("- Exactly 5 assigned images & non-overlapping with peer team")
            st.write("- Exactly 3 noise levels $\\sigma \\in \\{15, 25, 50\\}$ & 5 randomized trials")
            st.write("- Exactly 3 frozen solvers (ASL-SR-DPT with DC preservation, OMP, LASSO-ADMM)")
            st.write("- Exactly 225 expected workload keys for this team")
            st.write("- Codebase and configuration SHA-256 consistency")

            if g5_pass:
                st.success("STEP 5 PASSED: Final experiment integrity preconditions satisfied.")
            else:
                for err in g5_errors:
                    st.error(f"Experiment Integrity Error: {err}")

        # Overall readiness definition (Requirement 7)
        all_gates_pass = (
            g1_pass
            and g2_pass
            and st.session_state.reproducibility_passed
            and g4_pass
            and g5_pass
        )

        st.markdown("---")
        if all_gates_pass:
            st.success("ALL 5 GATES PASSED: Benchmark execution is unlocked.")
        else:
            st.warning("All 5 verification gates must pass to unlock benchmark execution.")

    # ==========================================================================
    # TAB 1: Live Benchmark Dashboard
    # ==========================================================================
    with tab_dash:
        solvers = PILOT_CONFIG.get("solvers", ["ASL-SR-DPT", "OMP", "LASSO-ADMM"])
        noises = PILOT_CONFIG.get("noise_levels", [15.0, 25.0, 50.0])
        trials = PILOT_CONFIG.get("trials", [1, 2, 3, 4, 5])

        task_list = []
        for img in assigned_images:
            for n in noises:
                for t in trials:
                    for s in solvers:
                        task_list.append((str(img), float(n), int(t), str(s)))

        total_tasks = len(task_list)  # exactly 225
        remaining_tasks = [k for k in task_list if k not in existing_keys]
        completed_count = total_tasks - len(remaining_tasks)

        # Metrics cards
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Target Evaluations", f"{total_tasks}")
        col_m2.metric("Completed", f"{completed_count}")
        col_m3.metric("Remaining", f"{len(remaining_tasks)}")
        col_m4.metric("Progress", f"{(completed_count / total_tasks)*100:.1f}%")
        col_m5.metric("Failures Logged", f"{len(pd.read_csv(paths['failures_csv'])) if os.path.exists(paths['failures_csv']) else 0}")

        st.progress(completed_count / float(total_tasks))

        # Control area
        can_start = (len(remaining_tasks) > 0) and all_gates_pass
        btn_label = "Start Benchmark Workload" if completed_count == 0 else "Resume Benchmark Workload"

        start_clicked = st.button(
            btn_label,
            disabled=(not can_start or st.session_state.benchmark_running),
            type="primary",
            use_container_width=True,
        )

        st.caption(
            "**Interruption & Resumption Safety:** Every completed evaluation is flushed atomically to disk. "
            "To pause, close the terminal or browser. Clicking Resume will automatically pick up from the next pending task."
        )

        # Live Execution Loop
        if start_clicked:
            st.session_state.benchmark_running = True

            progress_bar = st.progress(completed_count / float(total_tasks))
            status_placeholder = st.empty()
            recent_table_placeholder = st.empty()
            img_preview_col1, img_preview_col2 = st.columns(2)

            evals_done_session = 0

            for idx, (img_id, noise_sigma, trial, solver_name) in enumerate(task_list):
                key = (img_id, noise_sigma, trial, solver_name)
                if key in existing_keys:
                    continue

                def dash_patch_cb(p_cur, p_tot, p_msg):
                    p_pct = (p_cur / float(p_tot)) * 100.0
                    sub_eval_pct = ((completed_count + evals_done_session + (p_cur / float(p_tot))) / float(total_tasks)) * 100.0
                    status_placeholder.markdown(f"""
                        <div class="glass-status-card">
                            <div class="glass-status-title">
                                Execution {completed_count + evals_done_session + 1} / {total_tasks}: {solver_name}
                            </div>
                            <div class="glass-status-meta">
                                Image: <code>{img_id}</code> &nbsp;|&nbsp; Noise: <code>σ={int(noise_sigma)}</code> &nbsp;|&nbsp; Trial: <code>{trial}</code>
                            </div>
                            <div class="glass-status-detail">
                                Patch Progress: <b>{p_cur:,} / {p_tot:,}</b> ({p_pct:.1f}%) &nbsp;|&nbsp; 
                                Overall Workload: <b>{sub_eval_pct:.1f}%</b>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                try:
                    record, rec_img, noisy_img, _ = execute_single_pilot_evaluation(
                        img_id, noise_sigma, trial, solver_name, selected_team, PILOT_CONFIG, CONFIG_HASH, CODE_HASH,
                        patch_callback=dash_patch_cb
                    )
                    append_evaluation_record(paths["raw_csv"], record)
                    existing_keys.add(key)
                    evals_done_session += 1

                    # Live visual preview
                    with img_preview_col1:
                        st.image(noisy_img, caption=f"Noisy Input ({img_id}, σ={int(noise_sigma)})", clamp=True)
                    with img_preview_col2:
                        st.image(rec_img, caption=f"Reconstruction ({solver_name}, PSNR: {record['psnr']} dB)", clamp=True)

                except Exception as exc:
                    log_failure_record(paths["failures_csv"], {
                        "image_id": img_id,
                        "noise": noise_sigma,
                        "trial": trial,
                        "solver": solver_name,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "timestamp": datetime.now().isoformat(),
                    })

                # Update progress
                current_total_done = completed_count + evals_done_session
                progress_bar.progress(current_total_done / float(total_tasks))

                # Display latest table
                if os.path.exists(paths["raw_csv"]):
                    latest_df = pd.read_csv(paths["raw_csv"])
                    recent_table_placeholder.dataframe(
                        latest_df.tail(5)[["image_id", "noise_sigma", "trial", "solver", "psnr", "ssim", "solve_time", "ms_per_patch"]]
                    )

            st.session_state.benchmark_running = False
            if len(existing_keys) == total_tasks:
                gates_dict = {
                    "Gate 1 (Environment)": g1_pass,
                    "Gate 2 (Config Integrity)": g2_pass,
                    "Gate 3 (Reproducibility)": st.session_state.reproducibility_passed,
                    "Gate 4 (Workload & Storage)": g4_pass,
                    "Gate 5 (Experiment Integrity)": g5_pass,
                }
                repro_dict = {r["solver"]: r["classification"] for r in st.session_state.reproducibility_results}
                generate_team_report(
                    selected_team, paths["raw_csv"], paths["team_report_md"], PILOT_CONFIG, CONFIG_HASH, CODE_HASH, gates_dict, repro_dict
                )
                st.success(f"Team workload completed: all {total_tasks} evaluations finished.")
            st.rerun()

        # Recent Results Preview
        st.markdown("### Recent Evaluation Records")
        if os.path.exists(paths["raw_csv"]):
            df_recent = pd.read_csv(paths["raw_csv"])
            st.dataframe(df_recent.tail(10), use_container_width=True)
        else:
            st.info("No completed evaluations recorded yet.")

    # ==========================================================================
    # TAB 3: Team Report
    # ==========================================================================
    with tab_report:
        st.subheader(f"Summary Report: {selected_team}")
        if os.path.exists(paths["raw_csv"]):
            df_full = pd.read_csv(paths["raw_csv"])
            if len(df_full) == 225:
                st.success("225 / 225 Evaluations Completed. Master Team Report is generated.")
            else:
                st.warning(f"In Progress: {len(df_full)} / 225 evaluations completed.")

            # Summary Table
            st.markdown("#### Performance Breakdown by Solver and Noise")
            summary_grp = df_full.groupby(["solver", "noise_sigma"]).agg({
                "psnr": ["count", "mean", "std"],
                "ssim": ["mean", "std"],
                "mse": "mean",
                "solve_time": "mean",
                "ms_per_patch": "mean",
                "measurement_residual": "mean",
            }).round(4)
            st.dataframe(summary_grp, use_container_width=True)

            # Markdown Team Report View
            if os.path.exists(paths["team_report_md"]):
                with open(paths["team_report_md"], "r", encoding="utf-8") as rf:
                    st.markdown(rf.read())
        else:
            st.info("Run evaluations to generate team report.")

    # ==========================================================================
    # TAB 4: Export & Merge Center
    # ==========================================================================
    with tab_export:
        st.subheader("Results Export & Central Merge Preparation")
        st.write("Teammates can download their certified result packages here for final submission.")

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.markdown("#### Individual File Downloads")
            if os.path.exists(paths["raw_csv"]):
                with open(paths["raw_csv"], "rb") as f:
                    st.download_button(
                        label="Export Raw CSV",
                        data=f.read(),
                        file_name=f"pilot_raw_results_{selected_team.lower().replace(' ', '_')}.csv",
                        mime="text/csv",
                    )
            if os.path.exists(paths["failures_csv"]):
                with open(paths["failures_csv"], "rb") as f:
                    st.download_button(
                        label="Export Failure Log",
                        data=f.read(),
                        file_name=f"pilot_failures_{selected_team.lower().replace(' ', '_')}.csv",
                        mime="text/csv",
                    )
            if os.path.exists(paths["team_report_md"]):
                with open(paths["team_report_md"], "rb") as f:
                    st.download_button(
                        label="Export Team Report (Markdown)",
                        data=f.read(),
                        file_name=f"team_report_{selected_team.lower().replace(' ', '_')}.md",
                        mime="text/markdown",
                    )
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "rb") as f:
                    st.download_button(
                        label="Export Configuration JSON",
                        data=f.read(),
                        file_name="pilot_config.json",
                        mime="application/json",
                    )

        with col_ex2:
            st.markdown("#### Comprehensive ZIP Bundle")
            st.write("Bundles all raw CSVs, failure logs, summaries, and configuration into a single archive.")
            if st.button("Generate Teammate ZIP Package"):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for root, _, files in os.walk(paths["base"]):
                        for file in files:
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, paths["base"])
                            zf.write(full_path, arcname=rel_path)
                    if os.path.exists(CONFIG_PATH):
                        zf.write(CONFIG_PATH, arcname="pilot_config.json")

                st.download_button(
                    label="Download Complete Team Results (.ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"asl_sr_dpt_pilot_{selected_team.lower().replace(' ', '_')}.zip",
                    mime="application/zip",
                )

        st.markdown("---")
        st.markdown("#### Central Merge Instructions")
        st.code("""
# When both Team A (Khevin) and Team B (Marc) have completed their 225 evaluations:
python merge_pilot_results.py \\
    --team-a results/team_a/raw/pilot_raw_results_team_a.csv \\
    --team-b results/team_b/raw/pilot_raw_results_team_b.csv \\
    --team-a-failures results/team_a/failures/pilot_failures.csv \\
    --team-b-failures results/team_b/failures/pilot_failures.csv \\
    --config configs/pilot_config.json \\
    --output results/pilot_combined
        """, language="bash")
        st.caption("Verifies that 225 (Team A) + 225 (Team B) = exactly 450 unique evaluations without duplicates.")


if __name__ == "__main__":
    main()
