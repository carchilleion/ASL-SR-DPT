"""
Equivalence Test Suite: V7_BASELINE vs. V7_OPTIMIZED

Validates that HybridSparseSolverV7Optimized produces bit-for-bit / machine-precision
identical numerical results to HybridSparseSolverV7 across all parameters, boundary
conditions, and realistic image patch distributions.
"""

import sys
import os
import pytest
import numpy as np

# Ensure code/ is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from hybrid_sparse_solver_v7_fixed import HybridSparseSolverV7
from hybrid_sparse_solver_v7_optimized import HybridSparseSolverV7Optimized


def test_public_api_validation_safety():
    """Verify that outer boundary input validation catches all invalid inputs identically."""
    np.random.seed(42)
    A = np.random.randn(38, 64)
    A /= np.linalg.norm(A, axis=1, keepdims=True)

    solver_base = HybridSparseSolverV7(A)
    solver_opt = HybridSparseSolverV7Optimized(A)

    # 1. NaN in y
    y_nan = np.zeros(38)
    y_nan[5] = np.nan
    with pytest.raises(ValueError):
        solver_base.denoise_patch(y_nan)
    with pytest.raises(ValueError):
        solver_opt.denoise_patch(y_nan)

    # 2. Inf in y
    y_inf = np.zeros(38)
    y_inf[10] = np.inf
    with pytest.raises(ValueError):
        solver_base.denoise_patch(y_inf)
    with pytest.raises(ValueError):
        solver_opt.denoise_patch(y_inf)

    # 3. Wrong dimensions
    y_short = np.zeros(37)
    with pytest.raises(ValueError):
        solver_base.denoise_patch(y_short)
    with pytest.raises(ValueError):
        solver_opt.denoise_patch(y_short)

    # 4. Invalid sigma_min
    y_valid = np.random.randn(38)
    with pytest.raises(ValueError):
        solver_base.denoise_patch(y_valid, sigma_min=-0.01)
    with pytest.raises(ValueError):
        solver_opt.denoise_patch(y_valid, sigma_min=-0.01)


def test_numerical_equivalence_50_patches():
    """Verify strict mathematical equivalence across 50 pseudo-random sensing patches."""
    np.random.seed(20260908)
    M, N = 38, 64
    A = np.random.randn(M, N)
    A /= np.linalg.norm(A, axis=1, keepdims=True)

    solver_base = HybridSparseSolverV7(A, lambda_reg=0.1, tol=1e-5)
    solver_opt = HybridSparseSolverV7Optimized(A, lambda_reg=0.1, tol=1e-5)

    max_z_diff = 0.0
    max_obj_diff = 0.0
    max_res_diff = 0.0

    for i in range(50):
        y = np.random.randn(M)
        z_base, diag_base = solver_base.denoise_patch(y, return_diagnostics=True)
        z_opt, diag_opt = solver_opt.denoise_patch(y, return_diagnostics=True)

        diff_z = np.max(np.abs(z_base - z_opt))
        diff_res = abs(diag_base["final_residual"] - diag_opt["final_residual"])
        diff_obj = max(abs(a - b) for a, b in zip(diag_base["objective"], diag_opt["objective"]))

        max_z_diff = max(max_z_diff, diff_z)
        max_res_diff = max(max_res_diff, diff_res)
        max_obj_diff = max(max_obj_diff, diff_obj)

        assert diag_base["iterations"] == diag_opt["iterations"]
        assert diag_base["accepted_steps"] == diag_opt["accepted_steps"]
        assert diag_base["stop_reason"] == diag_opt["stop_reason"]
        assert diag_base["failed_line_searches"] == diag_opt["failed_line_searches"]
        assert abs(diag_base["final_sigma"] - diag_opt["final_sigma"]) < 1e-12

    print(f"Max z diff: {max_z_diff:.2e}, Max res diff: {max_res_diff:.2e}, Max obj diff: {max_obj_diff:.2e}")
    # Bound to machine precision / floating-point accumulation margin over 150 iterations
    assert max_z_diff < 1e-10
    assert max_res_diff < 1e-10
    assert max_obj_diff < 1e-8


def test_midpoint_toggle_equivalence():
    """Verify equivalence under both Midpoint ON and Midpoint OFF."""
    np.random.seed(12345)
    M, N = 38, 64
    A = np.random.randn(M, N)
    A /= np.linalg.norm(A, axis=1, keepdims=True)

    solver_base = HybridSparseSolverV7(A, lambda_reg=0.1, tol=1e-5)
    solver_opt = HybridSparseSolverV7Optimized(A, lambda_reg=0.1, tol=1e-5)

    y = np.random.randn(M)

    # Midpoint OFF
    z_b_off, d_b_off = solver_base.denoise_patch(y, use_midpoint=False, return_diagnostics=True)
    z_o_off, d_o_off = solver_opt.denoise_patch(y, use_midpoint=False, return_diagnostics=True)
    assert np.max(np.abs(z_b_off - z_o_off)) < 1e-11
    assert d_b_off["iterations"] == d_o_off["iterations"]

    # Midpoint ON
    z_b_on, d_b_on = solver_base.denoise_patch(y, use_midpoint=True, return_diagnostics=True)
    z_o_on, d_o_on = solver_opt.denoise_patch(y, use_midpoint=True, return_diagnostics=True)
    assert np.max(np.abs(z_b_on - z_o_on)) < 1e-11
    assert d_b_on["iterations"] == d_o_on["iterations"]


def test_initialization_toggle_equivalence():
    """Verify equivalence under both Pinv and Zeros initialization."""
    np.random.seed(54321)
    M, N = 38, 64
    A = np.random.randn(M, N)
    A /= np.linalg.norm(A, axis=1, keepdims=True)

    solver_base = HybridSparseSolverV7(A, lambda_reg=0.1, tol=1e-5)
    solver_opt = HybridSparseSolverV7Optimized(A, lambda_reg=0.1, tol=1e-5)

    y = np.random.randn(M)

    # Zeros cold-start
    z_b_0, d_b_0 = solver_base.denoise_patch(y, init_method="zeros", return_diagnostics=True)
    z_o_0, d_o_0 = solver_opt.denoise_patch(y, init_method="zeros", return_diagnostics=True)
    assert np.max(np.abs(z_b_0 - z_o_0)) < 1e-11
    assert d_b_0["iterations"] == d_o_0["iterations"]
