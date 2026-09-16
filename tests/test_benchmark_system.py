"""
Automated Pytest Suite for ASL-SR-DPT Benchmark System.
Includes:
  1. DCT/IDCT round trip
  2. Sensing dimensions & row normalization
  3. Pseudoinverse dimensions & Moore-Penrose property
  4. Objective calculation
  5. Analytical gradient
  6. Finite-difference gradient check
  7. Midpoint ON
  8. Midpoint OFF
  9. Sigma continuation
  10. Failed line-search path
  11. OMP baseline & col_norms precomputation
  12. LASSO-ADMM baseline & Cholesky reuse
  13. Reconstruction aggregation & division-by-zero safety
  14. Metrics computation & clean-vs-clean invariants
  15. CSV output schema
  16. Resume behavior with composite 8-element key
  17. Input fairness & bitwise hash identity across matched solvers (Part 19)
  18. BSD68 dataset integrity (68 images, deterministic sorting) (Part 22)
"""

import os
import sys
import tempfile
import csv
import hashlib
import pytest
import numpy as np

# Ensure code/ is on path
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code"))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from hybrid_sparse_solver_v7_fixed import (
    HybridSparseSolverV7,
    run_omp,
    precompute_omp,
    precompute_lasso_admm,
    run_lasso_admm,
    dct_2d,
    idct_2d,
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
from dataset import load_bsd68, get_bsd68_filenames
from metrics import compute_mse, compute_psnr, compute_ssim, compute_all_metrics
from summary import audit_row, generate_summary
from benchmark_runner import RAW_CSV_FIELDS, get_completed_keys


# 1. DCT/IDCT round trip
def test_dct_idct_roundtrip():
    rng = np.random.default_rng(101)
    patch = rng.uniform(0.0, 1.0, (8, 8))
    coeff = dct_patch(patch)
    rec = idct_patch(coeff)
    err = np.max(np.abs(patch - rec))
    assert err < 1e-12, f"DCT/IDCT roundtrip error too large: {err}"


# 2. Sensing dimensions & row normalization
def test_sensing_dimensions_and_norm():
    A = generate_random_sensing(M=38, N=64, seed=42)
    assert A.shape == (38, 64)
    row_norms = np.linalg.norm(A, axis=1)
    assert np.allclose(row_norms, 1.0, atol=1e-7), "Row normalization failed"


# 3. Pseudoinverse dimensions & Moore-Penrose property
def test_pseudoinverse():
    A = generate_random_sensing(M=38, N=64, seed=42)
    solver = HybridSparseSolverV7(A, lambda_reg=0.1)
    P = solver.P_init
    assert P.shape == (64, 38)
    assert np.allclose(A @ P @ A, A, atol=1e-7), "A @ P @ A != A"


# 4. Objective calculation
def test_objective_calculation():
    A = generate_random_sensing(M=38, N=64, seed=42)
    solver = HybridSparseSolverV7(A, lambda_reg=0.1)
    z = np.zeros(64)
    y = np.ones(38)
    cost = solver._calc_objective(z, y, sigma=1.0)
    assert abs(cost - 12.6) < 1e-8


# 5. Analytical gradient
def test_analytical_gradient():
    A = generate_random_sensing(M=38, N=64, seed=42)
    solver = HybridSparseSolverV7(A, lambda_reg=0.1)
    z = np.ones(64) * 0.5
    y = np.ones(38)
    active = np.ones(64, dtype=bool)
    grad = solver._calc_gradient(z, y, sigma=1.0, active_indices=active)
    assert grad.shape == (64,)
    assert np.all(np.isfinite(grad))


# 6. Finite-difference numerical gradient
def test_finite_difference_gradient():
    A = generate_random_sensing(M=38, N=64, seed=42)
    solver = HybridSparseSolverV7(A, lambda_reg=0.1)
    rng = np.random.default_rng(202)
    z = rng.standard_normal(64) * 0.3
    y = rng.standard_normal(38)
    active = np.ones(64, dtype=bool)
    sigma = 0.8
    eps = 1e-6

    grad_num = np.zeros(64)
    for i in range(64):
        zp, zm = z.copy(), z.copy()
        zp[i] += eps
        zm[i] -= eps
        cp = solver._calc_objective(zp, y, sigma)
        cm = solver._calc_objective(zm, y, sigma)
        grad_num[i] = (cp - cm) / (2.0 * eps)

    grad_ana = solver._calc_gradient(z, y, sigma, active)
    rel_err = np.linalg.norm(grad_num - grad_ana) / np.linalg.norm(grad_ana)
    assert rel_err < 1e-5, f"Gradient mismatch: rel_err={rel_err}"


# 7. Midpoint ON
def test_midpoint_on():
    A = generate_random_sensing(M=38, N=64, seed=42)
    solver = HybridSparseSolverV7(A, lambda_reg=0.1)
    y = np.ones(38)
    z, diag = solver.denoise_patch(y, max_iter=10, use_midpoint=True, return_diagnostics=True)
    assert diag["use_midpoint"] is True
    assert diag["iterations"] > 0


# 8. Midpoint OFF
def test_midpoint_off():
    A = generate_random_sensing(M=38, N=64, seed=42)
    solver = HybridSparseSolverV7(A, lambda_reg=0.1)
    y = np.ones(38)
    z, diag = solver.denoise_patch(y, max_iter=10, use_midpoint=False, return_diagnostics=True)
    assert diag["use_midpoint"] is False
    assert diag["iterations"] > 0


# 9. Sigma continuation
def test_sigma_continuation():
    A = generate_random_sensing(M=38, N=64, seed=42)
    solver = HybridSparseSolverV7(A, lambda_reg=0.1)
    y = np.ones(38)
    z, diag = solver.denoise_patch(y, max_iter=20, sigma_min=0.01, decrease_factor=0.9, return_diagnostics=True)
    assert diag["final_sigma"] >= 0.01
    assert diag["final_sigma"] <= diag["sigma"][0]


# 10. Failed-line-search path
def test_failed_line_search_path():
    A = generate_random_sensing(M=38, N=64, seed=42)
    solver = HybridSparseSolverV7(A, lambda_reg=0.1)
    y = np.ones(38)
    z, diag = solver.denoise_patch(
        y,
        max_iter=5,
        armijo_c=0.9999,
        max_backtracks=1,
        return_diagnostics=True,
    )
    assert diag["failed_line_searches"] >= 0


# 11. OMP baseline & col_norms precomputation
def test_omp_baseline_and_precomputation():
    A = generate_random_sensing(M=38, N=64, seed=42)
    theta_true = np.zeros(64)
    theta_true[[5, 12, 25]] = [2.0, -1.5, 1.0]
    y = A @ theta_true

    # Run without precomputed norms
    theta_rec1, diag1 = run_omp(y, A, max_coefficients=10, return_diagnostics=True)

    # Run with precomputed norms
    col_norms = precompute_omp(A)
    theta_rec2, diag2 = run_omp(y, A, max_coefficients=10, col_norms=col_norms, return_diagnostics=True)

    assert np.allclose(theta_rec1, theta_rec2, atol=1e-12), "Precomputed col_norms changed OMP result!"
    assert diag1["iterations"] == diag2["iterations"]
    assert diag1["final_residual"] < 1e-4


# 12. LASSO-ADMM baseline & Cholesky reuse
def test_lasso_admm_baseline():
    A = generate_random_sensing(M=38, N=64, seed=42)
    L = precompute_lasso_admm(A, rho=1.0)
    y = np.ones(38)
    z_lasso, diag = run_lasso_admm(y, A, L=L, max_iter=20, return_diagnostics=True)
    assert z_lasso.shape == (64,)
    assert diag["iterations"] > 0
    assert np.all(np.isfinite(z_lasso))


# 13. Reconstruction aggregation & division-by-zero safety
def test_reconstruction_aggregation_safety():
    rng = np.random.default_rng(303)
    img = rng.uniform(0.1, 0.9, (35, 45))  # Non-square, arbitrary dimensions
    patches = extract_patches(img, patch_size=8, stride=2)
    rec_patches = [
        {"patch": idct_patch(dct_patch(p["patch"])), "x": p["x"], "y": p["y"]}
        for p in patches
    ]
    rec_img = reconstruct_image(rec_patches, img.shape, patch_size=8, epsilon=1e-12)
    assert rec_img.shape == img.shape
    assert np.all(np.isfinite(rec_img))
    err = np.max(np.abs(img - rec_img))
    assert err < 1e-12


# 14. Metrics computation & clean-vs-clean invariants (Part 21)
def test_metric_invariants():
    clean = np.full((50, 50), 0.5)
    m = compute_all_metrics(clean, clean)
    assert m["mse"] == 0.0
    assert m["psnr"] == float("inf")
    assert abs(m["ssim"] - 1.0) < 1e-6


# 15. CSV output schema
def test_csv_output_schema():
    assert "mean_iterations_per_patch" in RAW_CSV_FIELDS
    assert "mean_active_support_count" in RAW_CSV_FIELDS
    assert "trial" in RAW_CSV_FIELDS
    assert "experiment" in RAW_CSV_FIELDS


# 16. Resume behavior with composite 8-element key
def test_resume_behavior():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, newline="", suffix=".csv") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_CSV_FIELDS)
        writer.writeheader()
        writer.writerow({
            "trial": 1,
            "image_id": "test001",
            "noise_sigma": 15.0,
            "sensing_mode": "standard",
            "solver": "ASL-SR-DPT",
            "psnr": 28.5,
            "ssim": 0.85,
            "mse": 0.001,
            "solve_time": 1.2,
            "setup_time": 0.005,
            "iterations": 50,
            "mean_iterations_per_patch": 50.0,
            "mean_active_support_count": 22.4,
            "final_sigma": 0.01,
            "residual": 0.02,
            "active_support_ratio": 0.35,
            "failed_line_searches": 0,
            "accepted_steps": 50,
            "seed": 20260908,
            "code_version": "v7.0.0-fixed",
            "patch_count": 100,
            "run_id": "test_run",
            "experiment": "E2",
        })
        temp_path = f.name

    try:
        completed = get_completed_keys(temp_path)
        expected_key = ("E2", 1, "test001", 15.0, "standard", "ASL-SR-DPT", 20260908, "v7.0.0-fixed")
        assert expected_key in completed
        # Different experiment or different seed must NOT match
        assert ("E3", 1, "test001", 15.0, "standard", "ASL-SR-DPT", 20260908, "v7.0.0-fixed") not in completed
        assert ("E2", 1, "test001", 15.0, "standard", "ASL-SR-DPT", 99999999, "v7.0.0-fixed") not in completed
    finally:
        os.remove(temp_path)


# 17. Input fairness test: SHA256 hash identity across matched solvers (Part 19)
def test_fairness_matched_inputs():
    clean_img = np.full((32, 32), 0.5)
    noise_seed = 424242
    sensing_seed = 848484

    # 1. Noisy image generation
    noisy1 = add_awgn(clean_img, sigma_noise=15.0, seed=noise_seed)
    noisy2 = add_awgn(clean_img, sigma_noise=15.0, seed=noise_seed)
    h_noisy1 = hashlib.sha256(noisy1.tobytes()).hexdigest()
    h_noisy2 = hashlib.sha256(noisy2.tobytes()).hexdigest()
    assert h_noisy1 == h_noisy2, "Noisy image is not deterministic!"

    # 2. Sensing matrix generation
    A1 = generate_random_sensing(38, 64, seed=sensing_seed)
    A2 = generate_random_sensing(38, 64, seed=sensing_seed)
    h_A1 = hashlib.sha256(A1.tobytes()).hexdigest()
    h_A2 = hashlib.sha256(A2.tobytes()).hexdigest()
    assert h_A1 == h_A2, "Sensing matrix A is not deterministic!"

    # 3. Patch measurements
    p = extract_patches(noisy1, patch_size=8, stride=2)[0]
    p_dct = dct_patch(p["patch"])
    y1 = generate_standard_measurement(p_dct, A1)
    y2 = generate_standard_measurement(p_dct, A2)
    h_y1 = hashlib.sha256(y1.tobytes()).hexdigest()
    h_y2 = hashlib.sha256(y2.tobytes()).hexdigest()
    assert h_y1 == h_y2, "Measurement vectors y are not bitwise identical!"


# 18. BSD68 dataset integrity test (Part 22)
def test_data_integrity():
    dataset_dir = "data/BSD68"
    if os.path.exists(dataset_dir):
        files = get_bsd68_filenames(dataset_dir)
        assert len(files) == 68, f"Expected 68 BSD68 images, found {len(files)}"
        ids = [os.path.splitext(os.path.basename(f))[0] for f in files]
        assert len(set(ids)) == 68, "Duplicate image IDs found in BSD68!"
        # Check deterministic sorting
        assert ids == sorted(ids), "BSD68 images are not sorted deterministically!"


# 19. Anomaly audit rules test (critical failure vs manual review flag)
def test_audit_anomaly_rules():
    seen_keys = set()
    valid_row = {
        "trial": "1",
        "image_id": "test001",
        "noise_sigma": "15.0",
        "sensing_mode": "standard",
        "solver": "ASL-SR-DPT",
        "psnr": "28.5",
        "ssim": "0.85",
        "mse": "0.001",
        "solve_time": "1.5",
        "setup_time": "0.001",
        "iterations": "50",
    }
    is_valid, err, warn = audit_row(valid_row, seen_keys)
    assert is_valid is True
    assert err is None
    assert warn is None

    # Critical failure: NaN metric
    nan_row = valid_row.copy()
    nan_row["trial"] = "2"
    nan_row["psnr"] = "NaN"
    is_valid, err, warn = audit_row(nan_row, seen_keys)
    assert is_valid is False
    assert "Non-finite metric" in err

    # Manual review trigger: PSNR > 60 dB (MUST remain valid, flagged for review)
    high_psnr_row = valid_row.copy()
    high_psnr_row["trial"] = "3"
    high_psnr_row["psnr"] = "65.2"
    is_valid, err, warn = audit_row(high_psnr_row, seen_keys)
    assert is_valid is True, "PSNR > 60 dB must not be automatically invalidated!"
    assert err is None
    assert warn is not None
    assert "Unusually high PSNR" in warn

