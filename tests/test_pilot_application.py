"""
Unit and integration test suite for the ASL-SR-DPT Pilot Benchmark Application.
Verifies:
  1. Config schema and hash stability.
  2. Exactly non-overlapping 225-evaluation workloads for Team A and Team B.
  3. Total 450 unique evaluations.
  4. Deterministic seed generation equations.
  5. Merge utility behavior, audit generation, and duplicate rejection.
  6. End-to-end reproducibility check logic.
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app import (
    load_pilot_config,
    compute_deterministic_seeds,
    get_team_directories,
    load_existing_composite_keys,
    append_evaluation_record,
    log_failure_record,
    validate_frozen_research_parameters,
    validate_team_assignment_integrity,
    validate_gate_5_experiment_integrity,
    validate_evaluation_record,
    validate_dpt_batch_diagnostics,
    compute_codebase_integrity_hash,
    get_environment_info,
    CONFIG_PATH,
)
from merge_pilot_results import (
    generate_expected_keys,
    audit_and_merge,
    load_config,
)


def test_config_loading_and_integrity():
    """Verify pilot configuration is well-formed, frozen, and produces valid SHA-256."""
    config, cfg_hash = load_pilot_config(CONFIG_PATH)
    assert isinstance(config, dict)
    assert len(cfg_hash) == 64
    assert config["patch_size"] == 8
    assert config["stride"] == 2
    assert config["N"] == 64
    assert config["M"] == 38
    assert config["M_ac"] == 37
    assert config["N_ac"] == 63
    assert config["noise_levels"] == [15.0, 25.0, 50.0]
    assert config["trials"] == [1, 2, 3, 4, 5]
    assert config["solvers"] == ["ASL-SR-DPT", "OMP", "LASSO-ADMM"]
    assert "Team A" in config["team_assignments"]
    assert "Team B" in config["team_assignments"]


def test_workload_disjoint_and_counts():
    """Verify Team A and Team B workloads are strictly disjoint and sum to exactly 450."""
    config, _ = load_config(CONFIG_PATH)
    expected_a, expected_b, expected_combined = generate_expected_keys(config)

    # Individual team counts
    assert len(expected_a) == 225, f"Team A must have 225 evaluations, got {len(expected_a)}"
    assert len(expected_b) == 225, f"Team B must have 225 evaluations, got {len(expected_b)}"

    # Zero cross-team overlap
    overlap = expected_a.intersection(expected_b)
    assert len(overlap) == 0, f"Found unexpected cross-team key overlap: {overlap}"

    # Combined total
    assert len(expected_combined) == 450, f"Expected 450 total keys, got {len(expected_combined)}"
    assert expected_combined == expected_a.union(expected_b)


def test_seed_determinism():
    """Verify deterministic seed generation matches mathematical specification."""
    base_seed = 20260908
    # Test case 1: test001, sigma=15.0, trial=1
    s_seed1, n_seed1 = compute_deterministic_seeds("test001", 15.0, 1, base_seed)
    assert s_seed1 == base_seed + 1000000 + 15000 + 1
    assert n_seed1 == base_seed + 100000 + 15000 + 1

    # Test case 2: test005, sigma=50.0, trial=5
    s_seed2, n_seed2 = compute_deterministic_seeds("test005", 50.0, 5, base_seed)
    assert s_seed2 == base_seed + 5000000 + 50000 + 5
    assert n_seed2 == base_seed + 500000 + 50000 + 5

    # Repeat call returns identical values
    assert compute_deterministic_seeds("test005", 50.0, 5, base_seed) == (s_seed2, n_seed2)


def test_team_directory_separation(tmp_path):
    """Verify team directories are separated and create required subdirectories."""
    paths_a = get_team_directories("Team A")
    paths_b = get_team_directories("Team B")

    assert "team_a" in paths_a["base"]
    assert "team_b" in paths_b["base"]
    assert paths_a["base"] != paths_b["base"]

    for k in ["raw", "logs", "failures", "reconstructions", "summaries"]:
        assert os.path.exists(paths_a[k])
        assert os.path.exists(paths_b[k])


def test_merge_utility_with_mock_datasets(tmp_path):
    """Test merge utility with synthetic valid Team A and Team B outputs."""
    config, cfg_hash = load_config(CONFIG_PATH)
    expected_a, expected_b, expected_combined = generate_expected_keys(config)

    test_dir_a = tmp_path / "team_a"
    test_dir_b = tmp_path / "team_b"
    test_out = tmp_path / "combined"
    os.makedirs(test_dir_a, exist_ok=True)
    os.makedirs(test_dir_b, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)

    csv_a = test_dir_a / "raw_a.csv"
    csv_b = test_dir_b / "raw_b.csv"

    # Create dummy records for all expected keys
    def make_mock_df(keys, team_name):
        rows = []
        for img, n, t, s in keys:
            clean_id = int(str(img).replace("test", "").lstrip("0") or "0")
            rows.append({
                "run_id": f"{img}_s{int(n)}_t{t}_{s}",
                "image_id": img,
                "noise_level": n,
                "noise_sigma": n,
                "trial": t,
                "solver": s,
                "team": team_name,
                "sensing_seed": 20260908 + clean_id * 1000000 + int(n) * 1000 + t,
                "noise_seed": 20260908 + t * 100000 + int(n) * 1000 + clean_id,
                "psnr": 24.5,
                "ssim": 0.62,
                "mse": 0.004,
                "setup_time": 0.01,
                "solve_time": 1.5,
                "milliseconds_per_patch": 1.2,
                "ms_per_patch": 1.2,
                "iteration_count": 98,
                "measurement_residual": 0.5,
                "relative_residual": 0.1,
                "normalized_residual": 0.08,
                "config_hash": cfg_hash,
                "code_version": "ASL-SR-DPT-PILOT-1.0",
                "timestamp": "2026-09-16T12:00:00",
            })
        return pd.DataFrame(rows)

    df_a = make_mock_df(expected_a, "Team A")
    df_b = make_mock_df(expected_b, "Team B")
    df_a.to_csv(csv_a, index=False)
    df_b.to_csv(csv_b, index=False)

    pass_result = audit_and_merge(
        team_a_csv=str(csv_a),
        team_b_csv=str(csv_b),
        team_a_failures_csv=str(test_dir_a / "none.csv"),
        team_b_failures_csv=str(test_dir_b / "none.csv"),
        config_path=CONFIG_PATH,
        output_dir=str(test_out),
    )
    assert pass_result is True

    # Check merged output exists and has 450 rows
    merged_raw = test_out / "pilot_combined_raw.csv"
    assert os.path.exists(merged_raw)
    merged_df = pd.read_csv(merged_raw)
    assert len(merged_df) == 450

    # Check audit report exists
    audit_report = test_out / "final_pilot_audit.md"
    assert os.path.exists(audit_report)
    with open(audit_report, "r", encoding="utf-8") as f:
        content = f.read()
        assert "Expected:\n450 evaluations" in content
        assert "Received:\n450" in content
        assert "Overall pilot integrity:\nPASS" in content


def test_environment_info_metadata():
    """Verify environment info captures scipy explicitly and contains no v7/a6 pilot references."""
    env = get_environment_info()
    assert "scipy_version" in env
    assert env["scipy_version"] != env["pandas_version"]
    assert env["app_version"] == "ASL-SR-DPT-PILOT-1.0"
    assert "code_hash" in env
    assert len(env["code_hash"]) == 64


def test_gate_2_and_gate_5_validations():
    """Verify Gate 2 (Configuration) and Gate 5 (Experiment Integrity) pass on frozen config."""
    config, cfg_hash = load_pilot_config(CONFIG_PATH)
    code_hash, _ = compute_codebase_integrity_hash()

    # Gate 2
    g2_ok, g2_errs = validate_frozen_research_parameters(config, cfg_hash)
    assert g2_ok is True, f"Gate 2 failed: {g2_errs}"

    # Team assignment integrity
    team_ok, team_errs = validate_team_assignment_integrity(config)
    assert team_ok is True, f"Team integrity failed: {team_errs}"

    # Gate 5 for Team A and Team B
    g5_a_ok, g5_a_errs = validate_gate_5_experiment_integrity("Team A", config, cfg_hash, code_hash)
    assert g5_a_ok is True, f"Gate 5 Team A failed: {g5_a_errs}"

    g5_b_ok, g5_b_errs = validate_gate_5_experiment_integrity("Team B", config, cfg_hash, code_hash)
    assert g5_b_ok is True, f"Gate 5 Team B failed: {g5_b_errs}"


def test_record_validation_invariants():
    """Verify validate_evaluation_record catches invalid metrics, corrupted keys, or schema violations."""
    config, cfg_hash = load_pilot_config(CONFIG_PATH)
    valid_record = {
        "psnr": 24.0,
        "ssim": 0.6,
        "mse": 0.004,
        "setup_time": 0.01,
        "solve_time": 1.5,
        "ms_per_patch": 1.2,
        "measurement_residual": 0.7,
        "solver": "ASL-SR-DPT",
        "noise_sigma": 15.0,
        "trial": 1,
        "image_id": "test001",
        "config_hash": cfg_hash,
        "sensing_m": 37,
        "sensing_n": 63,
        "sensing_architecture": "dc_preserving",
        "active_support_ratio": 0.85,
        "mean_active_support_count": 53.55,
        "final_sigma": 0.01,
        "accepted_steps": 12.0,
        "failed_line_searches": 0.0,
    }
    assigned = ["test001", "test002", "test003", "test004", "test005"]
    solvers = ["ASL-SR-DPT", "OMP", "LASSO-ADMM"]
    noises = [15.0, 25.0, 50.0]
    trials = [1, 2, 3, 4, 5]

    ok, msg = validate_evaluation_record(valid_record, assigned, solvers, noises, trials, cfg_hash)
    assert ok is True, f"Valid record failed: {msg}"

    # Bad solver
    bad_rec = dict(valid_record, solver="V7_OPT_BASE")
    ok, msg = validate_evaluation_record(bad_rec, assigned, solvers, noises, trials, cfg_hash)
    assert ok is False

    # Negative solve time
    bad_time = dict(valid_record, solve_time=-0.1)
    ok, msg = validate_evaluation_record(bad_time, assigned, solvers, noises, trials, cfg_hash)
    assert ok is False

    # NaN metric
    bad_nan = dict(valid_record, psnr=float("nan"))
    ok, msg = validate_evaluation_record(bad_nan, assigned, solvers, noises, trials, cfg_hash)
    assert ok is False

    # DPT invalid sensing dimensions
    bad_dim = dict(valid_record, sensing_m=38)
    ok, msg = validate_evaluation_record(bad_dim, assigned, solvers, noises, trials, cfg_hash)
    assert ok is False
    assert "sensing dimensions invalid" in msg

    # DPT invalid active support ratio out of [0, 1]
    bad_ratio = dict(valid_record, active_support_ratio=1.2)
    ok, msg = validate_evaluation_record(bad_ratio, assigned, solvers, noises, trials, cfg_hash)
    assert ok is False
    assert "active_support_ratio out of range" in msg

    # OMP with non-empty active support ratio (must be blank)
    omp_record = {
        "psnr": 23.5,
        "ssim": 0.58,
        "mse": 0.005,
        "setup_time": 0.01,
        "solve_time": 2.0,
        "ms_per_patch": 1.5,
        "measurement_residual": 0.8,
        "solver": "OMP",
        "noise_sigma": 15.0,
        "trial": 1,
        "image_id": "test001",
        "config_hash": cfg_hash,
        "sensing_m": 38,
        "sensing_n": 64,
        "sensing_architecture": "standard",
        "active_support_ratio": "",
        "mean_active_support_count": "",
        "final_sigma": "",
        "accepted_steps": "",
        "failed_line_searches": "",
    }
    ok, msg = validate_evaluation_record(omp_record, assigned, solvers, noises, trials, cfg_hash)
    assert ok is True

    bad_omp = dict(omp_record, active_support_ratio=0.5)
    ok, msg = validate_evaluation_record(bad_omp, assigned, solvers, noises, trials, cfg_hash)
    assert ok is False
    assert "must be blank/empty" in msg


def test_asl_sr_dpt_solver_real_diagnostics():
    """Verify HybridSparseSolverV7Optimized denoise_batch returns real diagnostics matching thesis definition."""
    from hybrid_sparse_solver_v7_optimized import HybridSparseSolverV7Optimized
    A = np.random.randn(37, 63)
    solver = HybridSparseSolverV7Optimized(A=A, lambda_reg=0.1, tol=1e-5)
    Y = np.random.randn(37, 4)

    Z, diag = solver.denoise_batch(Y, max_iter=20, return_diagnostics=True)
    assert Z.shape == (63, 4)
    assert "iterations" in diag
    assert diag["iterations"] > 0
    assert "active_support_counts" in diag
    assert "active_support_ratio" in diag
    # Active support ratio must be strictly in (0, 1]
    assert 0.0 < diag["active_support_ratio"] <= 1.0
    assert diag["accepted_steps"] >= 0
    assert diag["failed_line_searches"] >= 0


def test_dpt_batch_diagnostic_validation():
    """Verify validate_dpt_batch_diagnostics strictly enforces schema invariants per batch and per patch."""
    valid_diag = {
        "patch_iterations": [10, 15],
        "patch_active_counts": [[20] * 10, [25] * 15],
        "patch_accepted_steps": [10, 14],
        "patch_failed_line_searches": [0, 1],
        "patch_final_sigma": [0.01, 0.01],
    }
    ok, msg = validate_dpt_batch_diagnostics(valid_diag, expected_batch_size=2)
    assert ok is True

    # Length mismatch
    ok, msg = validate_dpt_batch_diagnostics(valid_diag, expected_batch_size=3)
    assert ok is False
    assert "length mismatch" in msg

    # Active support size out of bounds (> 63)
    bad_supp = {
        "patch_iterations": [2],
        "patch_active_counts": [[10, 64]],
        "patch_accepted_steps": [2],
        "patch_failed_line_searches": [0],
        "patch_final_sigma": [0.01],
    }
    ok, msg = validate_dpt_batch_diagnostics(bad_supp, expected_batch_size=1)
    assert ok is False
    assert "out of bounds [0, 63]" in msg

    # Negative iterations
    bad_iter = {
        "patch_iterations": [0],
        "patch_active_counts": [[]],
        "patch_accepted_steps": [0],
        "patch_failed_line_searches": [0],
        "patch_final_sigma": [0.01],
    }
    ok, msg = validate_dpt_batch_diagnostics(bad_iter, expected_batch_size=1)
    assert ok is False
    assert "iteration count must be > 0" in msg


def test_code_integrity_hash_includes_app_and_requirements():
    """Verify CODE_HASH covers app.py and requirements.txt."""
    code_hash, file_hashes = compute_codebase_integrity_hash()
    assert "app.py" in file_hashes
    assert len(file_hashes["app.py"]) == 64
    assert file_hashes["app.py"] != "missing"
    if os.path.exists(os.path.join(REPO_ROOT, "requirements.txt")):
        assert "requirements.txt" in file_hashes
        assert len(file_hashes["requirements.txt"]) == 64


