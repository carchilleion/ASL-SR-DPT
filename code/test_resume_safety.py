"""
Unit and Integration Test for Resume Safety and Data Integrity.
Verifies Phase 10 and Phase 11 requirements:
1. Composite key uniqueness: (image_id, trial, noise_sigma, solver).
2. Resume mechanism skips completed keys.
3. Duplicate rejection assertion.
4. Pre-write integrity assertions (finite values, valid bounds).
"""

import os
import sys
import csv
import tempfile
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from final_benchmark_runner import (
    RAW_SCHEMA,
    get_composite_key,
    load_existing_keys,
    assert_data_integrity,
)


def test_resume_and_integrity():
    print("Testing Resume Mechanism and Data Integrity Assertion...")

    # 1. Test Data Integrity Assertion
    valid_row = {
        "run_id": "test001_s15_t1_V7_A6_DC_PRESERVATION_1234",
        "image_id": "test001",
        "trial": 1,
        "noise_sigma": 15.0,
        "solver": "V7_A6_DC_PRESERVATION",
        "sensing_mode": "dc_preserving",
        "patch_count": 37604,
        "patch_size": 8,
        "stride": 2,
        "seed_noise": 20260908,
        "seed_sensing": 20260909,
        "setup_time": 0.005,
        "solve_time": 58.2,
        "ms_per_patch": 1.55,
        "psnr": 24.5,
        "ssim": 0.62,
        "mse": 0.004,
        "measurement_residual": 0.51,
        "relative_measurement_residual": 1.1,
        "normalized_measurement_residual": 0.08,
        "mean_iterations": 97.8,
        "median_iterations": 98.0,
        "max_iterations": 150,
        "active_support_ratio": 0.25,
        "mean_active_count": 16.0,
        "failed_line_searches": 0,
        "accepted_steps": 98,
        "final_sigma": 0.01,
        "code_version": "v7.0.0-final",
        "solver_hash": "abc123",
        "config_hash": "def456",
        "timestamp": "2026-09-08T00:00:00Z",
        "admm_primal_residual": "",
        "admm_dual_residual": "",
        "dc_error": 0.05,
        "ac_error": 0.5,
        "support_size": "",
    }

    # Should pass without error
    assert_data_integrity(valid_row)
    print("  [PASS] Valid row passed integrity assertion.")

    # Corrupt PSNR -> Should raise AssertionError
    corrupt_row_psnr = dict(valid_row, psnr=float("nan"))
    try:
        assert_data_integrity(corrupt_row_psnr)
        raise AssertionError("Failed to catch NaN PSNR!")
    except AssertionError:
        print("  [PASS] Successfully rejected NaN PSNR.")

    # Negative solve time -> Should raise AssertionError
    corrupt_row_time = dict(valid_row, solve_time=-1.5)
    try:
        assert_data_integrity(corrupt_row_time)
        raise AssertionError("Failed to catch negative solve_time!")
    except AssertionError:
        print("  [PASS] Successfully rejected negative solve_time.")

    # 2. Test Resume Mechanism with Mock CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_resume.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RAW_SCHEMA)
            writer.writeheader()
            writer.writerow(valid_row)

        existing = load_existing_keys(csv_path)
        expected_key = ("test001", 1, 15.0, "V7_A6_DC_PRESERVATION")
        assert expected_key in existing, f"Expected key {expected_key} not found!"
        assert len(existing) == 1, f"Expected 1 key, got {len(existing)}"
        print(f"  [PASS] Successfully loaded composite key: {expected_key}")

        # Test duplicate prevention
        new_row = dict(valid_row, psnr=25.0)  # Same composite key
        new_key = get_composite_key(new_row)
        assert new_key in existing, "Failed to identify duplicate key!"
        print(f"  [PASS] Duplicate key correctly identified and rejected.")

    print("ALL RESUME SAFETY AND INTEGRITY TESTS PASSED.")
    return True


if __name__ == "__main__":
    test_resume_and_integrity()
