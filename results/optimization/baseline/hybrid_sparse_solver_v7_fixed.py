import shutil
import time
import numpy as np
import scipy.fftpack as fftpack
import scipy.linalg as linalg


class HybridSparseSolverV7:
    """
    ASL-SR-DPT sparse-recovery solver.

    Core behavior:
      1. Moore-Penrose pseudoinverse is used only for initialization.
      2. Nonconvex objective uses a smoothed L0-style Gaussian surrogate.
      3. Active-support gradient updates are accepted through Armijo backtracking.
      4. Failed line searches do not trigger convergence and do not decay sigma.
      5. Sigma continuation occurs only after an accepted update.
      6. Support is fully reopened every 3 iterations.
      7. Optional diagnostics expose iteration-level data required for benchmarking.
    """

    def __init__(self, A, lambda_reg=0.1, tol=1e-5, seed=42):
        self.A = np.asarray(A, dtype=float)
        if self.A.ndim != 2:
            raise ValueError("A must be a 2D array.")
        if not np.all(np.isfinite(self.A)):
            raise ValueError("A contains NaN or Inf values.")

        self.M, self.N = self.A.shape
        self.lambda_reg = float(lambda_reg)
        self.tol = float(tol)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if self.lambda_reg < 0:
            raise ValueError("lambda_reg must be non-negative.")
        if self.tol <= 0:
            raise ValueError("tol must be positive.")

        # np.linalg.pinv uses an SVD-based Moore-Penrose pseudoinverse.
        # This is computed once when the solver object is created and is used
        # only for the minimum-norm initialization of each patch.
        self.P_init = np.linalg.pinv(self.A)

    def _validate_inputs(self, y, sigma):
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size != self.M:
            raise ValueError(f"y must have length {self.M}, got {y.size}.")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or Inf values.")
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be a finite positive number.")
        return y

    def _calc_objective(self, z, y, sigma, return_components=False):
        """L(z) = 0.5||Az-y||_2^2 - lambda * sum(exp(-z_i^2/(2 sigma^2)))."""
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
        """Gradient restricted to the active coefficients."""
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
        Recover a sparse coefficient vector from a measurement vector.

        The default settings retain the current V7 values with sigma_min=0.01.

        When return_diagnostics=True, a second dictionary is returned with:
          objective, mu, sigma, accepted_steps, failed_line_searches,
          active_counts, relative_change, final_residual, and stop_reason.
        When detailed_profile=True, comprehensive profiling data is attached:
          time_breakdown, operation_counts, iteration_history, sigma_history,
          active_support_summary, and initial_state.
        """
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

        iteration_history = []
        sigmas_list = [float(sigma)]
        sigmas_at_min = 0

        if detailed_profile:
            z0_val, z0_fid, z0_sparse = self._calc_objective(z, y, sigma, return_components=True)
            z0_res = float(np.linalg.norm(self.A @ z - y))
            initial_state = {
                "initial_objective": z0_val,
                "initial_fidelity": z0_fid,
                "initial_sparsity": z0_sparse,
                "initial_residual": z0_res,
                "initial_coefficient_norm": float(np.linalg.norm(z)),
                "initial_active_count": int(np.count_nonzero(np.abs(z) > support_threshold_multiplier * sigma)),
                "initial_sigma": float(sigma),
            }

        stop_reason = "max_iter_reached"

        for iteration in range(max_iter):
            # Active support calculation
            if detailed_profile:
                t_s = time.perf_counter()
            support_threshold = support_threshold_multiplier * sigma

            # Periodic support reopening.
            if iteration % support_reopen_interval == 0:
                active_indices = np.ones(self.N, dtype=bool)
            else:
                active_indices = np.abs(z) > support_threshold

            if not np.any(active_indices):
                active_indices = np.ones(self.N, dtype=bool)

            if detailed_profile:
                t_mask_total += time.perf_counter() - t_s
                count_mask += 1

            # Objective evaluation
            if detailed_profile:
                t_s = time.perf_counter()
                cost_current, cur_fid, cur_sparse = self._calc_objective(z, y, sigma, return_components=True)
                t_obj_total += time.perf_counter() - t_s
                count_obj += 1
            else:
                cost_current = self._calc_objective(z, y, sigma)

            # Gradient evaluation
            if detailed_profile:
                t_s = time.perf_counter()
            grad = self._calc_gradient(z, y, sigma, active_indices)
            if detailed_profile:
                t_grad_total += time.perf_counter() - t_s
                count_grad += 1

            d_active = -grad
            grad_dot_d = -float(np.dot(grad, grad))

            diagnostics["objective"].append(cost_current)
            diagnostics["sigma"].append(sigma)
            diagnostics["active_counts"].append(int(np.count_nonzero(active_indices)))

            step_size = mu
            step_size_initial = step_size
            best_z_active = z[active_indices].copy()
            line_search_success = False
            used_midpoint = False
            backtrack_iter = 0

            for bt in range(max_backtracks):
                backtrack_iter = bt + 1
                if detailed_profile:
                    count_backtracks += 1

                z_candidate_active = z[active_indices] + step_size * d_active
                z_candidate = z.copy()
                z_candidate[active_indices] = z_candidate_active

                if detailed_profile:
                    t_s = time.perf_counter()
                    cost_candidate = self._calc_objective(z_candidate, y, sigma)
                    t_cand_total += time.perf_counter() - t_s
                    count_cand += 1
                    count_obj += 1
                else:
                    cost_candidate = self._calc_objective(z_candidate, y, sigma)

                if cost_candidate <= cost_current + armijo_c * step_size * grad_dot_d:
                    best_z_active = z_candidate_active
                    line_search_success = True
                    mu = step_size
                    diagnostics["mu"].append(step_size)
                    used_midpoint = False
                    break

                if use_midpoint:
                    # Optional half-step sufficient-descent candidate.
                    # Disabling this branch produces the midpoint-off ablation.
                    z_mid_active = (z[active_indices] + z_candidate_active) / 2.0
                    z_mid = z.copy()
                    z_mid[active_indices] = z_mid_active

                    if detailed_profile:
                        t_s = time.perf_counter()
                        cost_mid = self._calc_objective(z_mid, y, sigma)
                        t_mid_total += time.perf_counter() - t_s
                        count_mid += 1
                        count_obj += 1
                    else:
                        cost_mid = self._calc_objective(z_mid, y, sigma)

                    if cost_mid <= cost_current + armijo_c * (step_size / 2.0) * grad_dot_d:
                        best_z_active = z_mid_active
                        line_search_success = True
                        mu = step_size * 0.5
                        diagnostics["mu"].append(mu)
                        used_midpoint = True
                        break

                step_size *= beta_decay

            if not line_search_success:
                diagnostics["failed_line_searches"] += 1
                diagnostics["mu"].append(step_size)
                mu *= 0.5

                # Failed line searches are not convergence.
                # z remains unchanged and sigma is intentionally frozen.
                diagnostics["relative_change"].append(0.0)

                if detailed_profile:
                    cur_res = float(np.linalg.norm(self.A @ z - y))
                    iteration_history.append({
                        "iteration": iteration + 1,
                        "active_count": int(np.count_nonzero(active_indices)),
                        "active_ratio": float(np.count_nonzero(active_indices)) / float(self.N),
                        "sigma": float(sigma),
                        "initial_mu": float(step_size_initial),
                        "accepted_mu": 0.0,
                        "mu": float(step_size_initial),
                        "backtracks": backtrack_iter,
                        "relative_change": 0.0,
                        "objective": float(cost_current),
                        "fidelity": float(cur_fid),
                        "sparsity": float(cur_sparse),
                        "residual": cur_res,
                        "accepted_or_failed": "failed",
                        "midpoint_accepted": False,
                    })
                    sigmas_list.append(float(sigma))
                    if sigma <= sigma_min + 1e-12:
                        sigmas_at_min += 1
                continue

            z_new = z.copy()
            z_new[active_indices] = best_z_active

            rel_change = np.linalg.norm(z_new - z) / (np.linalg.norm(z) + 1e-8)
            diagnostics["relative_change"].append(float(rel_change))
            diagnostics["accepted_steps"] += 1

            z = z_new

            if detailed_profile:
                cur_res = float(np.linalg.norm(self.A @ z - y))
                iteration_history.append({
                    "iteration": iteration + 1,
                    "active_count": int(np.count_nonzero(active_indices)),
                    "active_ratio": float(np.count_nonzero(active_indices)) / float(self.N),
                    "sigma": float(sigma),
                    "initial_mu": float(step_size_initial),
                    "accepted_mu": float(mu),
                    "mu": float(mu),
                    "backtracks": backtrack_iter - 1,
                    "relative_change": float(rel_change),
                    "objective": float(cost_current),
                    "fidelity": float(cur_fid),
                    "sparsity": float(cur_sparse),
                    "residual": cur_res,
                    "accepted_or_failed": "accepted_midpoint" if used_midpoint else "accepted",
                    "midpoint_accepted": bool(used_midpoint),
                })
                sigmas_list.append(float(sigma))
                if sigma <= sigma_min + 1e-12:
                    sigmas_at_min += 1

            # Stop only after a successful update AND after sigma has reached
            # the specified continuation floor.
            if rel_change < self.tol and sigma <= sigma_min:
                stop_reason = "converged_at_sigma_min"
                break

            # Continuation occurs only after an accepted step.
            sigma = max(sigma * decrease_factor, sigma_min)

        if detailed_profile:
            t_s = time.perf_counter()
        residual = self.A @ z - y
        res_norm = float(np.linalg.norm(residual))
        if detailed_profile:
            t_residual_total += time.perf_counter() - t_s

        diagnostics["final_residual"] = res_norm
        diagnostics["final_sigma"] = float(sigma)
        diagnostics["iterations"] = len(diagnostics["objective"])
        diagnostics["stop_reason"] = stop_reason

        if diagnostics["iterations"] > 0:
            diagnostics["mean_active_count"] = float(np.mean(diagnostics["active_counts"]))
            diagnostics["active_support_ratio"] = (
                float(np.sum(diagnostics["active_counts"]))
                / (diagnostics["iterations"] * self.N)
            )
        else:
            diagnostics["mean_active_count"] = 0.0
            diagnostics["active_support_ratio"] = 0.0

        if detailed_profile:
            diagnostics["time_breakdown"] = {
                "objective_time": t_obj_total,
                "gradient_time": t_grad_total,
                "candidate_time": t_cand_total,
                "midpoint_time": t_mid_total,
                "support_mask_time": t_mask_total,
                "residual_time": t_residual_total,
            }
            diagnostics["operation_counts"] = {
                "objective_evaluations": count_obj,
                "gradient_evaluations": count_grad,
                "candidate_evaluations": count_cand,
                "midpoint_evaluations": count_mid,
                "backtracking_attempts": count_backtracks,
                "accepted_steps": diagnostics["accepted_steps"],
                "failed_line_searches": diagnostics["failed_line_searches"],
                "support_mask_calculations": count_mask,
            }
            diagnostics["iteration_history"] = iteration_history
            diagnostics["sigma_history"] = {
                "initial_sigma": float(sigma_init),
                "final_sigma": float(sigma),
                "sigma_at_each_iteration": sigmas_list,
                "number_of_sigma_updates": diagnostics["accepted_steps"],
                "number_of_iterations_at_sigma_min": sigmas_at_min,
                "max_iter_hit": stop_reason == "max_iter_reached",
                "converged_at_sigma_min": stop_reason == "converged_at_sigma_min",
            }
            diagnostics["initial_state"] = initial_state

        if return_diagnostics:
            return z, diagnostics
        return z


def precompute_omp(A):
    """
    Precompute column L2 norms for a fixed sensing matrix A.
    Call this once when A is reused across patches to isolate pure iterative solve time.
    """
    A = np.asarray(A, dtype=float)
    col_norms = np.linalg.norm(A, axis=0)
    col_norms = np.where(col_norms < 1e-8, 1.0, col_norms)
    return col_norms


def run_omp(
    y,
    A,
    relative_residual_tol=1e-5,
    max_coefficients=None,
    col_norms=None,
    return_diagnostics=False,
):
    """Column-normalized OMP with scale-independent residual stopping."""
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    M, N = A.shape
    if y.size != M:
        raise ValueError(f"y must have length {M}, got {y.size}.")
    if max_coefficients is None:
        max_coefficients = M
    if max_coefficients <= 0:
        diagnostics = {
            "iterations": 0,
            "final_residual": float(np.linalg.norm(y)),
            "stop_reason": "max_coefficients_zero",
        }
        if return_diagnostics:
            return np.zeros(N), diagnostics
        return np.zeros(N)

    residual = y.copy()
    support = []
    theta = np.zeros(N)

    if col_norms is None:
        col_norms = precompute_omp(A)
    else:
        col_norms = np.asarray(col_norms, dtype=float)

    y_norm = max(np.linalg.norm(y), 1e-8)
    theta_sub = np.zeros(0)
    stop_reason = "max_coefficients_reached"

    for _ in range(min(max_coefficients, N)):
        projections = (A.T @ residual) / col_norms
        best_idx = int(np.argmax(np.abs(projections)))

        if best_idx in support:
            stop_reason = "duplicate_atom_selected"
            break

        support.append(best_idx)

        A_sub = A[:, support]
        theta_sub, _, _, _ = np.linalg.lstsq(A_sub, y, rcond=None)
        residual = y - A_sub @ theta_sub

        if np.linalg.norm(residual) / y_norm < relative_residual_tol:
            stop_reason = "residual_tol_reached"
            break

    if support:
        theta[support] = theta_sub

    diagnostics = {
        "iterations": len(support),
        "final_residual": float(np.linalg.norm(residual)),
        "stop_reason": stop_reason,
    }

    if return_diagnostics:
        return theta, diagnostics
    return theta


def precompute_lasso_admm(A, rho=1.0):
    """
    Precompute the Cholesky factorization for a fixed sensing matrix A.

    Call this once when A is reused across patches. This lets benchmark code
    report factorization/setup time separately from iterative solve time.
    """
    A = np.asarray(A, dtype=float)
    M, N = A.shape

    if rho <= 0:
        raise ValueError("rho must be positive.")

    normal_matrix = A.T @ A + rho * np.eye(N)
    L = linalg.cholesky(normal_matrix, lower=True)
    return L


def run_lasso_admm(
    y,
    A,
    lambda_lasso=0.01,
    rho=1.0,
    tol=1e-4,
    max_iter=100,
    L=None,
    return_diagnostics=False,
):
    """
    Convex LASSO-ADMM.

    Pass a precomputed Cholesky factor L to exclude factorization from
    iterative solve timing when A is reused.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    M, N = A.shape

    if y.size != M:
        raise ValueError(f"y must have length {M}, got {y.size}.")
    if lambda_lasso < 0:
        raise ValueError("lambda_lasso must be non-negative.")
    if rho <= 0:
        raise ValueError("rho must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    if L is None:
        L = precompute_lasso_admm(A, rho=rho)

    z = np.zeros(N)
    u = np.zeros(N)
    Aty = A.T @ y

    diagnostics = {
        "primal_residual": [],
        "dual_residual": [],
    }

    stop_reason = "max_iter_reached"

    for _ in range(max_iter):
        z_prev = z.copy()

        rhs = Aty + rho * z - u
        tmp = linalg.solve_triangular(L, rhs, lower=True)
        x = linalg.solve_triangular(L.T, tmp, lower=False)

        v = x + u / rho
        z = np.sign(v) * np.maximum(np.abs(v) - lambda_lasso / rho, 0.0)

        u = u + rho * (x - z)

        r_primal = float(np.linalg.norm(x - z))
        r_dual = float(rho * np.linalg.norm(z - z_prev))

        diagnostics["primal_residual"].append(r_primal)
        diagnostics["dual_residual"].append(r_dual)

        if r_primal < tol and r_dual < tol:
            stop_reason = "residuals_tol_reached"
            break

    diagnostics["iterations"] = len(diagnostics["primal_residual"])
    diagnostics["final_primal_residual"] = diagnostics["primal_residual"][-1] if diagnostics["primal_residual"] else 0.0
    diagnostics["final_dual_residual"] = diagnostics["dual_residual"][-1] if diagnostics["dual_residual"] else 0.0
    diagnostics["stop_reason"] = stop_reason

    if return_diagnostics:
        return z, diagnostics
    return z


def dct_2d(patch):
    patch = np.asarray(patch, dtype=float)
    if patch.ndim != 2:
        raise ValueError("patch must be a 2D array.")
    return fftpack.dct(
        fftpack.dct(patch.T, norm="ortho").T,
        norm="ortho",
    )


def idct_2d(coefficients):
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 2:
        raise ValueError("coefficients must be a 2D array.")
    return fftpack.idct(
        fftpack.idct(coefficients.T, norm="ortho").T,
        norm="ortho",
    )
