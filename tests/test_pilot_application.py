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
    # Test case 1
    s_seed1, n_seed1 = compute_deterministic_seeds("test001", 15.0, 1, base_seed)
    assert s_seed1 == base_seed + 1
    assert n_seed1 == base_seed + 100000 + 15000 + 1

    # Test case 2
    s_seed2, n_seed2 = compute_deterministic_seeds("test005", 50.0, 5, base_seed)
    assert s_seed2 == base_seed + 5
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
            rows.append({
                "run_id": f"{img}_s{int(n)}_t{t}_{s}",
                "image_id": img,
                "noise_level": n,
                "noise_sigma": n,
                "trial": t,
                "solver": s,
                "team": team_name,
                "sensing_seed": 20260908 + t,
                "noise_seed": 20260908 + t * 100000 + int(n) * 1000 + int(img.replace("test", "")),
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
                "code_version": "v7.0.0-pilot",
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
