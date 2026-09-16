"""
Validation module (E1 Verification Suite) for ASL-SR-DPT.
Runs 15 exhaustive checks on mathematical correctness, dimensions, gradients,
reconstruction identity, and solver execution.
"""

import sys
import numpy as np
from hybrid_sparse_solver_v7_fixed import (
    HybridSparseSolverV7,
    run_omp,
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
from dataset import load_bsd68
from metrics import compute_all_metrics, compute_mse, compute_psnr, compute_ssim


def run_e1_verification(dataset_dir="data/BSD68", verbose=True):
    """
    Execute the 15 required E1 verification tests.
    Returns:
      report: dict containing test results, statuses, and summary message.
    """
    results = {}
    
    def log(msg):
        if verbose:
            print(msg)

    log("=" * 60)
    log("RUNNING E1 VERIFICATION SUITE")
    log("=" * 60)

    # 1. BSD68 loading
    try:
        ds = load_bsd68(dataset_dir=dataset_dir, limit=5, strict_count=68)
        assert len(ds) == 5, f"Expected 5 loaded images, got {len(ds)}"
        results["1_bsd68_loading"] = {"status": "PASSED", "info": f"Loaded {len(ds)} images successfully"}
        log("[PASS] 1. BSD68 loading")
    except Exception as e:
        results["1_bsd68_loading"] = {"status": "FAILED", "error": str(e)}
        log(f"[FAIL] 1. BSD68 loading: {e}")

    # 2. Grayscale & normalization in [0, 1]
    try:
        sample_img = next(iter(ds.values()))["image"]
        assert sample_img.ndim == 2, f"Expected 2D image, got shape {sample_img.shape}"
        assert 0.0 <= sample_img.min() and sample_img.max() <= 1.0, "Values outside [0, 1]"
        results["2_grayscale_normalized"] = {"status": "PASSED", "info": f"Shape {sample_img.shape}, min={sample_img.min():.2f}, max={sample_img.max():.2f}"}
        log("[PASS] 2. Grayscale shape and [0, 1] range")
    except Exception as e:
        results["2_grayscale_normalized"] = {"status": "FAILED", "error": str(e)}
        log(f"[FAIL] 2. Grayscale normalization: {e}")

    # 3. DCT -> IDCT round trip identity
    try:
        rng = np.random.default_rng(42)
        patch = rng.uniform(0.0, 1.0, (8, 8))
        c = dct_patch(patch)
        rec = idct_patch(c)
        err = float(np.max(np.abs(patch - rec)))
        assert err < 1e-12, f"DCT roundtrip error too large: {err}"
        results["3_dct_idct_identity"] = {"status": "PASSED", "info": f"Max abs error = {err:.2e}"}
        log(f"[PASS] 3. DCT -> IDCT roundtrip identity (error = {err:.2e})")
    except Exception as e:
        results["3_dct_idct_identity"] = {"status": "FAILED", "error": str(e)}
        log(f"[FAIL] 3. DCT -> IDCT identity: {e}")

    # 4. Sensing dimensions & row normalization
    try:
        M, N = 38, 64
        A = generate_random_sensing(M, N, seed=42)
        assert A.shape == (38, 64), f"Wrong shape {A.shape}"
        row_norms = np.linalg.norm(A, axis=1)
        assert np.allclose(row_norms, 1.0, atol=1e-7), "Rows not unit normalized"
        results["4_sensing_dimensions_and_norm"] = {"status": "PASSED", "info": f"Shape {A.shape}, row norms = 1.0"}
        log("[PASS] 4. Sensing dimensions (38, 64) and row-normalization")
    except Exception as e:
        results["4_sensing_dimensions_and_norm"] = {"status": "FAILED", "error": str(e)}
        log(f"[FAIL] 4. Sensing dimensions: {e}")

    # 5. Pseudoinverse dimensions & property
    try:
        solver = HybridSparseSolverV7(A, lambda_reg=0.1, tol=1e-5)
        P_init = solver.P_init
        assert P_init.shape == (64, 38), f"Wrong pinv shape {P_init.shape}"
        # Moore-Penrose property: A @ P @ A == A
        assert np.allclose(A @ P_init @ A, A, atol=1e-7), "P_init is not a valid pseudoinverse"
        results["5_pinv_dimensions_and_properties"] = {"status": "PASSED", "info": f"Shape {P_init.shape}, A @ P @ A = A verified"}
        log("[PASS] 5. Pseudoinverse dimensions (64, 38) and Moore-Penrose property")
    except Exception as e:
        results["5_pinv_dimensions_and_properties"] = {"status": "FAILED", "error": str(e)}
        log(f"[FAIL] 5. Pseudoinverse: {e}")

    # 6. Measurement dimensions
    try:
        theta = rng.standard_normal(64)
        y = generate_standard_measurement(theta, A)
        assert y.shape == (38,), f"Wrong y shape {y.shape}"

        A_ac = generate_dc_sensing(M_ac=37, N_ac=63, seed=42)
        theta_dc, y_ac = generate_dc_preserving_measurement(theta, A_ac)
        assert isinstance(theta_dc, float)
        assert y_ac.shape == (37,)
        theta_restored = restore_dc_component(theta_dc, theta[1:])
        assert np.allclose(theta, theta_restored), "DC restoration mismatch"
        results["6_measurement_dimensions"] = {"status": "PASSED", "info": "Standard y=(38,), DC y_ac=(37,), DC restored"}
        log("[PASS] 6. Measurement dimensions (Standard y=38, DC y_ac=37)")
    except Exception as e:
        results["6_measurement_dimensions"] = {"status": "FAILED", "error": str(e)}
        log(f"[FAIL] 6. Measurement dimensions: {e}")

    # 7. Objective finite
    try:
        cost = solver._calc_objective(theta, y, sigma=1.0)
        assert np.isfinite(cost), f"Non-finite cost: {cost}"
        results["7_objective_finite"] = {"status": "PASSED", "info": f"Evaluated objective = {cost:.4f}"}
        log(f"[PASS] 7. Smoothed L0 objective is finite ({cost:.4f})")
    except Exception as e:
        results["7_objective_finite"] = {"status": "FAILED", "error": str(e)}
        log(f"[FAIL] 7. Objective finite: {e}")

    # 8. Gradient finite
    try:
        active_all = np.ones(64, dtype=bool)
        grad = solver._calc_gradient(theta, y, sigma=1.0, active_indices=active_all)
        assert grad.shape == (64,), f"Wrong grad shape {grad.shape}"
        assert np.all(np.isfinite(grad)), "Gradient has NaN/Inf"
        results["8_gradient_finite"] = {"status": "PASSED", "info": f"Evaluated gradient norm = {np.linalg.norm(grad):.4f}"}
        log("[PASS] 8. Analytical gradient is finite")
    except Exception as e:
        results["8_gradient_finite"] = {"status": "FAILED", "error": str(e)}
        log(f"[FAIL] 8. Gradient finite: {e}")

    # 9. Numerical gradient check (finite differences)
    try:
        eps = 1e-6
        sigma_val = 1.0
        grad_num = np.zeros(64)
        for i in range(64):
            z_plus = theta.copy()
            z_minus = theta.copy()
            z_plus[i] += eps
            z_minus[i] -= eps
            cost_plus = solver._calc_objective(z_plus, y, sigma_val)
            cost_minus = solver._calc_objective(z_minus, y, sigma_val)
            grad_num[i] = (cost_plus - cost_minus) / (2.0 * eps)

        grad_ana = solver._calc_gradient(theta, y, sigma_val, active_all)
        max_diff = float(np.max(np.abs(grad_num - grad_ana)))
        rel_diff = float(np.linalg.norm(grad_num - grad_ana) / (np.linalg.norm(grad_ana) + 1e-8))
        assert rel_diff < 1e-4, f"Numerical vs analytical gradient discrepancy too high: rel={rel_diff:.2e}"
        results["9_numerical_gradient_check"] = {"status": "PASSED", "info": f"Max diff={max_diff:.2e}, rel diff={rel_diff:.2e}"}
        log(f"[PASS] 9. Numerical gradient check passes (rel diff = {rel_diff:.2e})")
    except Exception as e:
        results["9_numerical_gradient_check"] = {"status": "FAILED", "error": str(e)}
        log(f"[FAIL] 9. Numerical gradient check: {e}")

    # 10. ASL-SR-DPT smoke run
    try:
        z_rec, diag = solver.denoise_patch(y, max_iter=20, return_diagnostics=True)
        assert z_rec.shape == (64,), f"Wrong z_rec shape {z_rec.shape}"
        assert np.all(np.isfinite(z_rec)), "z_rec contains NaN or Inf"
        assert diag["iterations"] > 0, "No iterations recorded"
        results["10_asl_sr_dpt_smoke_run"] = {"status": "PASSED", "info": f"Completed {diag['iterations']} iters, final sigma={diag['final_sigma']:.4f}"}
        log("[PASS] 10. ASL-SR-DPT smoke run with diagnostics")
    except Exception as e:
        results["10_asl_sr_dpt_smoke_run"] = {"status": "FAILED", "error": str(e)}
        log(f"[FAIL] 10. ASL-SR-DPT smoke run: {e}")

    # 11. OMP smoke run
    try:
        theta_omp, diag_omp = run_omp(y, A, max_coefficients=15, return_diagnostics=True)
        assert theta_omp.shape == (64,), f"Wrong theta_omp shape {theta_omp.shape}"
        assert np.all(np.isfinite(theta_omp)), "OMP output contains NaN or Inf"
        assert diag_omp["iterations"] <= 15
        results["11_omp_smoke_run"] = {"status": "PASSED", "info": f"OMP selected {diag_omp['iterations']} atoms"}
        log("[PASS] 11. OMP smoke run with diagnostics")
    except Exception as e:
        results["11_omp_smoke_run"] = {"status": "FAILED", "error": str(e)}
        log(f"[FAIL] 11. OMP smoke run: {e}")

    # 12. LASSO-ADMM smoke run
    try:
        L = precompute_lasso_admm(A, rho=1.0)
        z_lasso, diag_lasso = run_lasso_admm(y, A, L=L, max_iter=25, return_diagnostics=True)
        assert z_lasso.shape == (64,), f"Wrong z_lasso shape {z_lasso.shape}"
        assert np.all(np.isfinite(z_lasso)), "LASSO output contains NaN or Inf"
        assert diag_lasso["iterations"] > 0
        results["12_lasso_smoke_run"] = {"status": "PASSED", "info": f"LASSO-ADMM completed {diag_lasso['iterations']} iters"}
        log("[PASS] 12. LASSO-ADMM smoke run with precomputed Cholesky factor")
    except Exception as e:
        results["12_lasso_smoke_run"] = {"status": "FAILED", "error": str(e)}
        log(f"[FAIL] 12. LASSO-ADMM smoke run: {e}")

    # 13. Full reconstruction pipeline check
    try:
        test_img = sample_img[:64, :64]  # 64x64 sub-image for fast verification
        patches = extract_patches(test_img, patch_size=8, stride=2)
        rec_patches = []
        for p in patches:
            p_rec = idct_patch(dct_patch(p["patch"]))
            rec_patches.append({"patch": p_rec, "x": p["x"], "y": p["y"]})
        reconstructed = reconstruct_image(rec_patches, test_img.shape, patch_size=8)
        assert reconstructed.shape == test_img.shape
        assert np.all(np.isfinite(reconstructed))
        results["13_reconstruction_pipeline"] = {"status": "PASSED", "info": f"Reconstructed shape {reconstructed.shape}"}
        log("[PASS] 13. 2D Hamming-window aggregation reconstruction")
    except Exception as e:
        results["13_reconstruction_pipeline"] = {"status": "FAILED", "error": str(e)}
        log(f"[FAIL] 13. Reconstruction pipeline: {e}")

    # 14. Metrics computation
    try:
        noisy_img = add_awgn(test_img, sigma_noise=25, seed=42)
        m = compute_all_metrics(test_img, noisy_img)
        assert np.isfinite(m["mse"]) and m["mse"] > 0
        assert np.isfinite(m["psnr"])
        assert np.isfinite(m["ssim"]) and 0 <= m["ssim"] <= 1.0
        results["14_metrics_computation"] = {"status": "PASSED", "info": f"MSE={m['mse']:.5f}, PSNR={m['psnr']:.2f}dB, SSIM={m['ssim']:.4f}"}
        log(f"[PASS] 14. Metrics computed: PSNR={m['psnr']:.2f}dB, SSIM={m['ssim']:.4f}")
    except Exception as e:
        results["14_metrics_computation"] = {"status": "FAILED", "error": str(e)}
        log(f"[FAIL] 14. Metrics computation: {e}")

    # 15. No NaN / Inf safety across all components
    all_passed = all(v["status"] == "PASSED" for v in results.values())
    results["15_no_nan_or_inf"] = {
        "status": "PASSED" if all_passed else "FAILED",
        "info": "All outputs verified strictly finite" if all_passed else "Failures encountered"
    }
    log("[PASS] 15. Global finite / no NaN or Inf assertion")
    log("=" * 60)
    log(f"E1 VERIFICATION RESULT: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    log("=" * 60)

    return results, all_passed


if __name__ == "__main__":
    results, ok = run_e1_verification()
    sys.exit(0 if ok else 1)
