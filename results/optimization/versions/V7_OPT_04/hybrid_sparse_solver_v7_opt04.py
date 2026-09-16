"""
ASL-SR-DPT Solver - Version V7_OPT_04
Optimization Candidate D: Precompute C-Contiguous Transpose A_T

Precomputes self.A_T = np.ascontiguousarray(self.A.T) once during initialization.
Memory cost: 64 x 38 x 8 bytes = 19.45 KiB (resides fully within L1 data cache).
Replaces non-contiguous transpose view attribute access self.A.T with contiguous
row-major BLAS matrix-vector multiplication in the gradient calculation.
"""

import time
import numpy as np


class HybridSparseSolverV7Optimized:
    """
    ASL-SR-DPT Sparse Recovery Solver - V7_OPT_04.
    Candidate D: Precomputed C-contiguous transpose A_T.
    """

    def __init__(self, A, lambda_reg=0.1, tol=1e-5):
        self.A = np.asarray(A, dtype=float)
        if self.A.ndim != 2:
            raise ValueError("A must be a 2D matrix.")
        self.M, self.N = self.A.shape
        if self.M >= self.N:
            raise ValueError("A must have fewer rows than columns (M < N).")

        self.lambda_reg = float(lambda_reg)
        self.tol = float(tol)

        if self.lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative.")
        if self.tol <= 0:
            raise ValueError("tol must be positive.")

        self.P_init = np.linalg.pinv(self.A)
        self.all_active_mask = np.ones(self.N, dtype=bool)

        # Candidate D: Precompute C-contiguous transpose for L1 cache-optimal matrix-vector streaming
        self.A_T = np.ascontiguousarray(self.A.T)

    def _validate_inputs(self, y, sigma):
        """Public validation helper preserving API compatibility."""
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != self.M:
            raise ValueError(f"y must have length {self.M}, got {y.size}.")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or Inf values.")
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be a finite positive number.")
        return y

    def _calc_objective(self, z, y, sigma, return_components=False):
        """Public objective evaluation: L(z) = 0.5||Az-y||_2^2 - lambda * sum(exp(-z_i^2/(2 sigma^2)))."""
        z = np.asarray(z, dtype=float)
        y = self._validate_inputs(y, sigma)

        residual = self.A @ z - y
        fidelity = 0.5 * np.dot(residual, residual)

        exponent = -(z * z) / (2.0 * sigma * sigma)
        sparsity = -self.lambda_reg * np.sum(np.exp(exponent))

        value = fidelity + sparsity
        if not np.isfinite(value):
            raise FloatingPointError("Objective evaluation produced NaN or Inf.")
        if return_components:
            return float(value), float(fidelity), float(sparsity)
        return float(value)

    def _calc_gradient(self, z, y, sigma, active_indices):
        """Public gradient evaluation restricted to the active coefficients."""
        z = np.asarray(z, dtype=float)
        y = self._validate_inputs(y, sigma)

        active_indices = np.asarray(active_indices, dtype=bool)
        if active_indices.size != self.N:
            raise ValueError("active_indices has the wrong length.")

        A_active = self.A[:, active_indices]
        z_active = z[active_indices]

        residual = A_active @ z_active - y
        grad_fidelity = A_active.T @ residual

        exponent = -(z_active * z_active) / (2.0 * sigma * sigma)
        grad_sparsity = (z_active / (sigma * sigma)) * np.exp(exponent)

        grad = grad_fidelity + self.lambda_reg * grad_sparsity

        if not np.all(np.isfinite(grad)):
            raise FloatingPointError("Gradient evaluation produced NaN or Inf.")
        return grad

    def denoise_patch(
        self,
        y,
        sigma_init=None,
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
        return_diagnostics=False,
        detailed_profile=False,
        init_method="pinv",
    ):
        """
        Recover a sparse coefficient vector from a measurement vector using V7_OPT_04.
        """
        # 1. Outer boundary input validation
        y = self._validate_inputs(y, sigma=1.0)

        if sigma_min <= 0:
            raise ValueError("sigma_min must be positive.")
        if not 0 < decrease_factor <= 1:
            raise ValueError("decrease_factor must be in (0, 1].")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if initial_mu <= 0:
            raise ValueError("initial_mu must be positive.")
        if not 0 < armijo_c < 1:
            raise ValueError("armijo_c must be in (0, 1).")
        if not 0 < beta_decay < 1:
            raise ValueError("beta_decay must be in (0, 1).")
        if max_backtracks <= 0:
            raise ValueError("max_backtracks must be positive.")
        if support_threshold_multiplier < 0:
            raise ValueError("support_threshold_multiplier must be non-negative.")
        if support_reopen_interval <= 0:
            raise ValueError("support_reopen_interval must be positive.")
        if not isinstance(use_midpoint, (bool, np.bool_)):
            raise ValueError("use_midpoint must be True or False.")

        # 2. Minimum-norm or cold-start initialization
        if init_method == "zeros":
            z = np.zeros(self.N, dtype=float)
        else:
            z = self.P_init @ y

        if sigma_init is None:
            z_max = float(np.max(np.abs(z)))
            sigma_init = 2.5 * z_max if z_max > 0 else 1.0

        sigma = float(max(sigma_init, sigma_min))
        mu = float(initial_mu)

        diagnostics = {
            "objective": [],
            "mu": [],
            "sigma": [],
            "accepted_steps": 0,
            "failed_line_searches": 0,
            "active_counts": [],
            "relative_change": [],
            "use_midpoint": bool(use_midpoint),
        }

        # Profiling accumulators
        t_obj_total = 0.0
        t_grad_total = 0.0
        t_cand_total = 0.0
        t_mid_total = 0.0
        t_mask_total = 0.0
        t_residual_total = 0.0

        count_obj = 0
        count_grad = 0
        count_cand = 0
        count_mid = 0
        count_backtracks = 0
        count_mask = 0
        count_matvec = 0
        count_exp = 0

        iteration_history = []
        sigmas_list = [float(sigma)]
        sigmas_at_min = 0

        # Precompute initial residual and fidelity once
        if detailed_profile:
            t_s = time.perf_counter()
        r = self.A @ z - y
        fid = 0.5 * float(np.dot(r, r))
        if detailed_profile:
            t_residual_total += time.perf_counter() - t_s
            count_matvec += 1

            inv_2sigma2_0 = 1.0 / (2.0 * sigma * sigma)
            w0 = np.exp(-(z * z) * inv_2sigma2_0)
            initial_state = {
                "initial_objective": fid - self.lambda_reg * float(np.sum(w0)),
                "initial_fidelity": fid,
                "initial_sparsity": -self.lambda_reg * float(np.sum(w0)),
                "initial_residual": float(np.linalg.norm(r)),
                "initial_coefficient_norm": float(np.linalg.norm(z)),
                "initial_active_count": int(np.count_nonzero(np.abs(z) > support_threshold_multiplier * sigma)),
                "initial_sigma": float(sigma),
            }

        stop_reason = "max_iter_reached"

        for iteration in range(max_iter):
            # 1. Active support determination with preallocated mask
            if detailed_profile:
                t_s = time.perf_counter()
            support_threshold = support_threshold_multiplier * sigma

            if iteration % support_reopen_interval == 0:
                active_indices = self.all_active_mask
                all_active = True
            else:
                active_indices = np.abs(z) > support_threshold
                if not active_indices.any():
                    active_indices = self.all_active_mask
                    all_active = True
                else:
                    all_active = bool(active_indices.all())

            if detailed_profile:
                t_mask_total += time.perf_counter() - t_s
                count_mask += 1

            # 2. Shared evaluation: compute w = exp(-z^2/(2*sigma^2)) once
            if detailed_profile:
                t_s = time.perf_counter()
            inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)
            inv_sigma2 = 1.0 / (sigma * sigma)
            w = np.exp(-(z * z) * inv_2sigma2)
            sparse_val = -self.lambda_reg * float(np.sum(w))
            cost_current = fid + sparse_val
            if detailed_profile:
                t_obj_total += time.perf_counter() - t_s
                count_obj += 1
                count_exp += self.N

            diagnostics["objective"].append(cost_current)
            diagnostics["sigma"].append(sigma)
            diagnostics["active_counts"].append(self.N if all_active else int(np.count_nonzero(active_indices)))

            # 3. Gradient evaluation reusing r, w and precomputed C-contiguous A_T
            if detailed_profile:
                t_s = time.perf_counter()
            if all_active:
                A_active = self.A
                z_active = z
                # Candidate D: Use C-contiguous self.A_T
                grad_fidelity = self.A_T @ r
                grad_sparsity = (z * inv_sigma2) * w
                grad = grad_fidelity + self.lambda_reg * grad_sparsity
                if detailed_profile:
                    count_matvec += 1
            else:
                A_active = self.A[:, active_indices]
                z_active = z[active_indices]
                res_act = A_active @ z_active - y
                grad_fidelity = A_active.T @ res_act
                w_active = w[active_indices]
                grad_sparsity = (z_active * inv_sigma2) * w_active
                grad = grad_fidelity + self.lambda_reg * grad_sparsity
                if detailed_profile:
                    count_matvec += 2

            if detailed_profile:
                t_grad_total += time.perf_counter() - t_s
                count_grad += 1

            d_active = -grad
            grad_dot_d = -float(np.dot(grad, grad))

            # 4. Precompute A_d for linear candidate evaluation
            if detailed_profile:
                t_s = time.perf_counter()
            A_d = A_active @ d_active
            if detailed_profile:
                count_matvec += 1

            step_size = mu
            step_size_initial = step_size
            line_search_success = False
            used_midpoint = False
            backtrack_iter = 0
            accepted_r = r
            accepted_fid = fid
            best_z_active = None

            for bt in range(max_backtracks):
                backtrack_iter = bt + 1
                if detailed_profile:
                    count_backtracks += 1

                # Linear residual update: r_cand = r + step_size * A_d
                r_cand = r + step_size * A_d
                fid_cand = 0.5 * float(np.dot(r_cand, r_cand))

                z_candidate_active = z_active + step_size * d_active
                if all_active:
                    w_cand = np.exp(-(z_candidate_active * z_candidate_active) * inv_2sigma2)
                    sparse_cand = -self.lambda_reg * float(np.sum(w_cand))
                    if detailed_profile:
                        count_exp += self.N
                else:
                    w_cand_act = np.exp(-(z_candidate_active * z_candidate_active) * inv_2sigma2)
                    sparse_cand = -self.lambda_reg * float(np.sum(w_cand_act) + np.sum(w[~active_indices]))
                    if detailed_profile:
                        count_exp += int(np.count_nonzero(active_indices))

                cost_candidate = fid_cand + sparse_cand

                if detailed_profile:
                    count_cand += 1
                    count_obj += 1

                if cost_candidate <= cost_current + armijo_c * step_size * grad_dot_d:
                    best_z_active = z_candidate_active
                    line_search_success = True
                    mu = step_size
                    diagnostics["mu"].append(step_size)
                    used_midpoint = False
                    accepted_r = r_cand
                    accepted_fid = fid_cand
                    break

                if use_midpoint:
                    step_mid = step_size * 0.5
                    r_mid = r + step_mid * A_d
                    fid_mid = 0.5 * float(np.dot(r_mid, r_mid))

                    z_mid_active = z_active + step_mid * d_active
                    if all_active:
                        w_mid = np.exp(-(z_mid_active * z_mid_active) * inv_2sigma2)
                        sparse_mid = -self.lambda_reg * float(np.sum(w_mid))
                        if detailed_profile:
                            count_exp += self.N
                    else:
                        w_mid_act = np.exp(-(z_mid_active * z_mid_active) * inv_2sigma2)
                        sparse_mid = -self.lambda_reg * float(np.sum(w_mid_act) + np.sum(w[~active_indices]))
                        if detailed_profile:
                            count_exp += int(np.count_nonzero(active_indices))

                    cost_mid = fid_mid + sparse_mid

                    if detailed_profile:
                        count_mid += 1
                        count_obj += 1

                    if cost_mid <= cost_current + armijo_c * step_mid * grad_dot_d:
                        best_z_active = z_mid_active
                        line_search_success = True
                        mu = step_mid
                        diagnostics["mu"].append(mu)
                        used_midpoint = True
                        accepted_r = r_mid
                        accepted_fid = fid_mid
                        break

                step_size *= beta_decay

            if not line_search_success:
                diagnostics["failed_line_searches"] += 1
                diagnostics["mu"].append(step_size)
                mu *= 0.5
                diagnostics["relative_change"].append(0.0)

                if detailed_profile:
                    cur_res = float(np.linalg.norm(self.A @ z - y))
                    iteration_history.append({
                        "iteration": iteration + 1,
                        "active_count": self.N if all_active else int(np.count_nonzero(active_indices)),
                        "active_ratio": (self.N if all_active else float(np.count_nonzero(active_indices))) / float(self.N),
                        "sigma": float(sigma),
                        "initial_mu": float(step_size_initial),
                        "accepted_mu": 0.0,
                        "mu": float(step_size_initial),
                        "backtracks": backtrack_iter,
                        "relative_change": 0.0,
                        "objective": float(cost_current),
                        "fidelity": float(fid),
                        "sparsity": float(sparse_val),
                        "residual": cur_res,
                        "accepted_or_failed": "failed",
                        "midpoint_accepted": False,
                    })
                    sigmas_list.append(float(sigma))
                    if sigma <= sigma_min + 1e-12:
                        sigmas_at_min += 1
                continue

            # Full-support step assignment
            if all_active:
                z_new = best_z_active
            else:
                z_new = z.copy()
                z_new[active_indices] = best_z_active

            rel_change = np.linalg.norm(z_new - z) / (np.linalg.norm(z) + 1e-8)
            diagnostics["relative_change"].append(float(rel_change))
            diagnostics["accepted_steps"] += 1

            z = z_new
            r = accepted_r
            fid = accepted_fid

            if detailed_profile:
                cur_res = float(np.linalg.norm(r))
                iteration_history.append({
                    "iteration": iteration + 1,
                    "active_count": self.N if all_active else int(np.count_nonzero(active_indices)),
                    "active_ratio": (self.N if all_active else float(np.count_nonzero(active_indices))) / float(self.N),
                    "sigma": float(sigma),
                    "initial_mu": float(step_size_initial),
                    "accepted_mu": float(mu),
                    "mu": float(mu),
                    "backtracks": backtrack_iter - 1,
                    "relative_change": float(rel_change),
                    "objective": float(cost_current),
                    "fidelity": float(fid),
                    "sparsity": float(sparse_val),
                    "residual": cur_res,
                    "accepted_or_failed": "accepted_midpoint" if used_midpoint else "accepted",
                    "midpoint_accepted": bool(used_midpoint),
                })
                sigmas_list.append(float(sigma))
                if sigma <= sigma_min + 1e-12:
                    sigmas_at_min += 1

            # Stop condition
            if rel_change < self.tol and sigma <= sigma_min:
                stop_reason = "converged_at_sigma_min"
                break

            sigma = max(sigma * decrease_factor, sigma_min)

        if detailed_profile:
            t_s = time.perf_counter()
        final_residual_norm = float(np.linalg.norm(self.A @ z - y))
        if detailed_profile:
            t_residual_total += time.perf_counter() - t_s

        diagnostics["final_residual"] = final_residual_norm
        diagnostics["final_sigma"] = float(sigma)
        diagnostics["iterations"] = len(diagnostics["objective"])
        diagnostics["stop_reason"] = stop_reason

        if diagnostics["iterations"] > 0:
            diagnostics["mean_active_count"] = float(np.mean(diagnostics["active_counts"]))
            diagnostics["active_support_ratio"] = (
                float(np.sum(diagnostics["active_counts"]))
                / (diagnostics["iterations"] * self.N)
            )

        if detailed_profile:
            diagnostics["time_breakdown"] = {
                "solver_objective_evaluation": t_obj_total,
                "solver_gradient_evaluation": t_grad_total,
                "solver_armijo_candidate_checks": t_cand_total,
                "solver_midpoint_evaluation": t_mid_total,
                "solver_support_mask_calculation": t_mask_total,
                "solver_final_residual_calc": t_residual_total,
            }
            diagnostics["operation_counts"] = {
                "iterations": diagnostics["iterations"],
                "accepted_steps": diagnostics["accepted_steps"],
                "backtracking_attempts": count_backtracks,
                "failed_line_searches": diagnostics["failed_line_searches"],
                "objective_evaluations": count_obj,
                "gradient_evaluations": count_grad,
                "candidate_evaluations": count_cand,
                "midpoint_evaluations": count_mid,
                "support_mask_calculations": count_mask,
                "matrix_vector_multiplications": count_matvec,
                "exponential_evaluations": count_exp,
            }
            diagnostics["iteration_history"] = iteration_history
            diagnostics["sigma_history"] = {
                "initial_sigma": float(sigmas_list[0]),
                "final_sigma": float(sigmas_list[-1]),
                "number_of_sigma_updates": len(sigmas_list) - 1,
                "number_of_iterations_at_sigma_min": sigmas_at_min,
                "max_iter_hit": stop_reason == "max_iter_reached",
                "converged_at_sigma_min": stop_reason == "converged_at_sigma_min",
            }
            diagnostics["initial_state"] = initial_state
            diagnostics["detailed_profile"] = {
                "time_breakdown": diagnostics["time_breakdown"],
                "operation_counts": diagnostics["operation_counts"],
                "iteration_history": diagnostics["iteration_history"],
                "sigma_history": diagnostics["sigma_history"],
                "initial_state": diagnostics["initial_state"],
            }

        if return_diagnostics:
            return z, diagnostics
        return z
