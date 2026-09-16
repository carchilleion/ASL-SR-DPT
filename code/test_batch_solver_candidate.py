"""
Prototype Batched ASL-SR-DPT Solver (Candidate H)
Processes B patches simultaneously:
  Z shape: (N, B)
  Y shape: (M, B)
  R shape: (M, B)
Validates mathematical equivalence against single-patch solver on 10, 100, and 500 patches.
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_sparse_solver_v7_optimized import HybridSparseSolverV7Optimized
from sensing import generate_random_sensing, add_awgn
from reconstruction import extract_patches, dct_patch
from dataset import load_bsd68


class BatchedHybridSparseSolverV7:
    """
    Batched ASL-SR-DPT Solver (Candidate H).
    Preserves exact mathematical independence per patch while computing
    matrix-vector operations as batched matrix-matrix GEMM.
    """

    def __init__(self, A, lambda_reg=0.1, tol=1e-5):
        self.A = np.asarray(A, dtype=float)
        self.M, self.N = self.A.shape
        self.lambda_reg = float(lambda_reg)
        self.tol = float(tol)
        self.P_init = np.linalg.pinv(self.A)
        self.A_T = np.ascontiguousarray(self.A.T)

    def denoise_batch(
        self,
        Y,
        sigma_min=0.01,
        decrease_factor=0.95,
        max_iter=150,
        initial_mu=0.2,
        armijo_c=1e-4,
        beta_decay=0.5,
        max_backtracks=10,
        support_threshold_multiplier=1e-5,
        support_reopen_interval=3,
        use_midpoint=True,
    ):
        """
        Denoise B patches in parallel.
        Y: shape (M, B)
        Returns Z: shape (N, B)
        """
        Y = np.asarray(Y, dtype=float)
        M, B = Y.shape
        if M != self.M:
            raise ValueError(f"Expected {self.M} rows in Y, got {M}")

        # 1. Batch initialization via single GEMM (N, M) @ (M, B) -> (N, B)
        Z = self.P_init @ Y

        # Independent initial sigmas per patch
        z_max_vec = np.max(np.abs(Z), axis=0)  # shape (B,)
        sigma = np.where(z_max_vec > 0, 2.5 * z_max_vec, 1.0)
        sigma = np.maximum(sigma, sigma_min)

        mu = np.full(B, initial_mu, dtype=float)

        # Initial residual via GEMM: (M, N) @ (N, B) - (M, B) -> (M, B)
        R = self.A @ Z - Y
        # Fidelity per patch: 0.5 * sum(R^2, axis=0) -> (B,)
        fid = 0.5 * np.sum(R * R, axis=0)

        # Per-patch convergence tracking
        converged = np.zeros(B, dtype=bool)

        for iteration in range(max_iter):
            # Check if all patches converged
            if np.all(converged):
                break

            # 1. Support determination
            support_threshold = support_threshold_multiplier * sigma  # (B,)
            if iteration % support_reopen_interval == 0:
                all_active = True
                active_mask = np.ones((self.N, B), dtype=bool)
            else:
                active_mask = np.abs(Z) > support_threshold[np.newaxis, :]
                # Empty support fallback
                any_act = active_mask.any(axis=0)
                if not np.all(any_act):
                    active_mask[:, ~any_act] = True
                all_active = bool(active_mask.all())

            # 2. Shared evaluation: W = exp(Z^2 * neg_inv_2sigma2)
            neg_inv_2sigma2 = -0.5 / (sigma * sigma)  # (B,)
            inv_sigma2 = 1.0 / (sigma * sigma)  # (B,)

            # (N, B) exponent
            exp_arg = (Z * Z) * neg_inv_2sigma2[np.newaxis, :]
            W = np.exp(exp_arg)

            sparse_val = -self.lambda_reg * np.sum(W, axis=0)  # (B,)
            cost_current = fid + sparse_val  # (B,)

            # 3. Gradient calculation
            if all_active:
                # GEMM: (N, M) @ (M, B) -> (N, B)
                grad_fidelity = self.A_T @ R
                grad_sparsity = (Z * inv_sigma2[np.newaxis, :]) * W
                grad = grad_fidelity + self.lambda_reg * grad_sparsity
            else:
                # Sparse fallback per patch
                grad = np.empty((self.N, B), dtype=float)
                for b in range(B):
                    act_b = active_mask[:, b]
                    if act_b.all():
                        grad_fidelity_b = self.A_T @ R[:, b]
                        grad_sparsity_b = (Z[:, b] * inv_sigma2[b]) * W[:, b]
                        grad[:, b] = grad_fidelity_b + self.lambda_reg * grad_sparsity_b
                    else:
                        A_act = self.A[:, act_b]
                        z_act = Z[act_b, b]
                        res_act = A_act @ z_act - Y[:, b]
                        grad_fidelity_b = A_act.T @ res_act
                        w_act = W[act_b, b]
                        grad_sparsity_b = (z_act * inv_sigma2[b]) * w_act
                        grad_b = grad_fidelity_b + self.lambda_reg * grad_sparsity_b
                        grad[act_b, b] = grad_b
                        grad[~act_b, b] = 0.0

            D = -grad  # (N, B)
            grad_dot_d = -np.sum(grad * grad, axis=0)  # (B,)

            # 4. A_d projection via single GEMM
            if all_active:
                A_d = self.A @ D  # (M, B)
            else:
                A_d = np.empty((self.M, B), dtype=float)
                for b in range(B):
                    act_b = active_mask[:, b]
                    A_d[:, b] = self.A[:, act_b] @ D[act_b, b]

            # 5. Fast primary Armijo candidate step (mu)
            R_cand = R + mu[np.newaxis, :] * A_d  # (M, B)
            fid_cand = 0.5 * np.sum(R_cand * R_cand, axis=0)  # (B,)

            Z_cand = Z + mu[np.newaxis, :] * D  # (N, B)
            exp_cand = (Z_cand * Z_cand) * neg_inv_2sigma2[np.newaxis, :]
            W_cand = np.exp(exp_cand)
            sparse_cand = -self.lambda_reg * np.sum(W_cand, axis=0)  # (B,)

            cost_cand = fid_cand + sparse_cand  # (B,)

            armijo_threshold = cost_current + armijo_c * mu * grad_dot_d
            primary_accepted = cost_cand <= armijo_threshold  # (B,)

            # For patches that accepted on primary candidate
            best_Z = Z_cand.copy()
            accepted_R = R_cand.copy()
            accepted_fid = fid_cand.copy()
            accepted_mu = mu.copy()

            # For any patches that did NOT accept on primary step, fallback independently
            if not np.all(primary_accepted):
                for b in np.where(~primary_accepted)[0]:
                    # Exact single-patch backtracking fallback
                    d_b = D[:, b]
                    z_b = Z[:, b]
                    r_b = R[:, b]
                    a_d_b = A_d[:, b]
                    g_dot_d_b = grad_dot_d[b]
                    c_curr_b = cost_current[b]
                    neg_inv_b = neg_inv_2sigma2[b]
                    w_b = W[:, b]
                    act_b = active_mask[:, b]

                    step_b = mu[b]
                    succ_b = False

                    if use_midpoint:
                        step_mid = step_b * 0.5
                        r_mid = r_b + step_mid * a_d_b
                        fid_mid = 0.5 * float(np.dot(r_mid, r_mid))
                        z_mid = z_b + step_mid * d_b
                        w_mid = np.exp((z_mid * z_mid) * neg_inv_b)
                        if act_b.all():
                            sparse_mid = -self.lambda_reg * float(w_mid.sum())
                        else:
                            sparse_mid = -self.lambda_reg * float(w_mid[act_b].sum() + w_b[~act_b].sum())
                        if (fid_mid + sparse_mid) <= c_curr_b + armijo_c * step_mid * g_dot_d_b:
                            best_Z[:, b] = z_mid
                            accepted_R[:, b] = r_mid
                            accepted_fid[b] = fid_mid
                            accepted_mu[b] = step_mid
                            succ_b = True

                    if not succ_b:
                        step_b *= beta_decay
                        for bt in range(1, max_backtracks):
                            r_c = r_b + step_b * a_d_b
                            fid_c = 0.5 * float(np.dot(r_c, r_c))
                            z_c = z_b + step_b * d_b
                            w_c = np.exp((z_c * z_c) * neg_inv_b)
                            if act_b.all():
                                sparse_c = -self.lambda_reg * float(w_c.sum())
                            else:
                                sparse_c = -self.lambda_reg * float(w_c[act_b].sum() + w_b[~act_b].sum())
                            if (fid_c + sparse_c) <= c_curr_b + armijo_c * step_b * g_dot_d_b:
                                best_Z[:, b] = z_c
                                accepted_R[:, b] = r_c
                                accepted_fid[b] = fid_c
                                accepted_mu[b] = step_b
                                succ_b = True
                                break

                            if use_midpoint:
                                step_mid = step_b * 0.5
                                r_mid = r_b + step_mid * a_d_b
                                fid_mid = 0.5 * float(np.dot(r_mid, r_mid))
                                z_mid = z_b + step_mid * d_b
                                w_mid = np.exp((z_mid * z_mid) * neg_inv_b)
                                if act_b.all():
                                    sparse_mid = -self.lambda_reg * float(w_mid.sum())
                                else:
                                    sparse_mid = -self.lambda_reg * float(w_mid[act_b].sum() + w_b[~act_b].sum())
                                if (fid_mid + sparse_mid) <= c_curr_b + armijo_c * step_mid * g_dot_d_b:
                                    best_Z[:, b] = z_mid
                                    accepted_R[:, b] = r_mid
                                    accepted_fid[b] = fid_mid
                                    accepted_mu[b] = step_mid
                                    succ_b = True
                                    break

                            step_b *= beta_decay

                    if not succ_b:
                        mu[b] *= 0.5
                        continue

            # Update step for accepted patches
            Z_new = best_Z
            diff_Z = Z_new - Z
            norm_dZ = np.sqrt(np.sum(diff_Z * diff_Z, axis=0))
            norm_Z = np.sqrt(np.sum(Z * Z, axis=0))
            rel_change = norm_dZ / (norm_Z + 1e-8)

            Z = Z_new
            R = accepted_R
            fid = accepted_fid
            mu = accepted_mu

            # Convergence condition per patch
            newly_converged = (rel_change < self.tol) & (sigma <= sigma_min)
            converged |= newly_converged

            sigma = np.maximum(sigma * decrease_factor, sigma_min)

        return Z


def test_batched_solver():
    print("Testing Batched ASL-SR-DPT Solver (Candidate H)...")
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "final_config.json")
    import json
    with open(config_path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "BSD68")
    images = load_bsd68(data_dir)
    test_img = images["test001"]["image"]

    base_seed = cfg.get("base_seed", 20260908)
    M, N = 38, 64
    A_std = generate_random_sensing(M, N, seed=base_seed)
    noisy_img = add_awgn(test_img, sigma_noise=15.0, seed=base_seed)

    records = extract_patches(noisy_img, patch_size=8, stride=2)[:500]
    noisy_patches = [r["patch"] for r in records]
    measurements = np.array([A_std @ dct_patch(p).reshape(-1) for p in noisy_patches])

    s_single = HybridSparseSolverV7Optimized(A_std, lambda_reg=0.1, tol=1e-5)
    s_batch = BatchedHybridSparseSolverV7(A_std, lambda_reg=0.1, tol=1e-5)

    test_sizes = [10, 100, 500]
    for count in test_sizes:
        print(f"\n--- Testing Batch Size B = {count} ---")
        Y_sub = measurements[:count].T  # (M, count)

        # Single patch loop
        t0 = time.perf_counter()
        Z_single = np.column_stack([s_single.denoise_patch(measurements[i]) for i in range(count)])
        t_single = time.perf_counter() - t0

        # Batched solver
        t0 = time.perf_counter()
        Z_batched = s_batch.denoise_batch(Y_sub)
        t_batched = time.perf_counter() - t0

        diff_z = np.max(np.abs(Z_single - Z_batched))
        speedup = t_single / t_batched
        reduction = (t_single - t_batched) / t_single * 100.0

        print(f"  Single patch: {t_single:.4f} s ({(t_single/count)*1000:.3f} ms/patch)")
        print(f"  Batched:      {t_batched:.4f} s ({(t_batched/count)*1000:.3f} ms/patch)")
        print(f"  Speedup:      {speedup:.2f}x ({reduction:.1f}% reduction)")
        print(f"  Max z diff:   {diff_z:.3e} (threshold: 1e-10) -> {'PASS' if diff_z < 1e-10 else 'FAIL'}")


if __name__ == "__main__":
    test_batched_solver()
