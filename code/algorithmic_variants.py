"""
Algorithmic Variants Module for ASL-SR-DPT.
Implements isolated, strictly controlled algorithmic variants for thesis investigation:

- V7_OPT_BASE: Reference implementation-optimized V7 solver.
- V7_A1_ADAPTIVE_EXIT: Adaptive consecutive-stability early termination.
- V7_A2_SCALE_COUPLED_LAMBDA: Scale-coupled surrogate regularization lambda(sigma) = lambda_0 * sigma^2.
- V7_A3_SUPPORT_THRESHOLD: Sensitivity study of active support threshold multiplier.
- V7_A4_CONTINUATION: Sensitivity study of geometric continuation schedule decay factor.
- V7_A5_TWO_STAGE: Support identification followed by least-squares debiasing on support.
- V7_A6_DC_PRESERVATION: Dedicated AC recovery with exact DC preservation.
"""

import numpy as np


class SolverV7OptBase:
    """
    V7_OPT_BASE: Reference baseline solver using approved V7 mathematics
    and Category-A implementation optimizations.
    """

    def __init__(self, A, lambda_reg=0.1, tol=1e-5):
        self.A = np.asarray(A, dtype=float)
        self.M, self.N = self.A.shape
        self.lambda_reg = float(lambda_reg)
        self.tol = float(tol)

        self.P_init = np.linalg.pinv(self.A)
        self.all_active_mask = np.ones(self.N, dtype=bool)
        self.A_T = np.ascontiguousarray(self.A.T)

    def denoise_patch(
        self,
        y,
        initial_sigma=None,
        decrease_factor=0.95,
        sigma_min=0.01,
        max_iter=150,
        initial_mu=0.2,
        beta_decay=0.5,
        armijo_c=1e-4,
        max_backtracks=10,
        support_threshold_multiplier=1e-5,
        support_reopen_interval=3,
        use_midpoint=True,
        return_diagnostics=True,
    ):
        y = np.asarray(y, dtype=float).reshape(-1)
        z = self.P_init @ y

        z_max = float(np.max(np.abs(z)))
        if initial_sigma is None:
            sigma = 2.5 * z_max if z_max > 0 else 1.0
        else:
            sigma = float(initial_sigma)
        sigma = max(sigma, sigma_min)

        mu = float(initial_mu)

        diagnostics = {
            "objective": [],
            "fidelity": [],
            "sparsity": [],
            "sigma": [],
            "active_counts": [],
            "mu": [],
            "relative_change": [],
            "failed_line_searches": 0,
            "backtracking_steps": 0,
            "accepted_steps": 0,
            "midpoint_acceptances": 0,
            "gradient_audits": {},
        }

        r = self.A @ z - y
        fid = 0.5 * float(np.dot(r, r))
        stop_reason = "max_iter_reached"

        # Milestones for gradient audit
        audit_targets = [("init", None), ("sigma_1", 1.0), ("sigma_01", 0.1), ("sigma_001", 0.01)]
        audited = set()

        for iteration in range(max_iter):
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

            neg_inv_2sigma2 = -0.5 / (sigma * sigma)
            inv_sigma2 = 1.0 / (sigma * sigma)
            w = np.exp((z * z) * neg_inv_2sigma2)
            sparse_val = -self.lambda_reg * float(w.sum())
            cost_current = fid + sparse_val

            if return_diagnostics:
                diagnostics["objective"].append(cost_current)
                diagnostics["fidelity"].append(fid)
                diagnostics["sparsity"].append(sparse_val)
                diagnostics["sigma"].append(sigma)
                act_cnt = self.N if all_active else int(np.count_nonzero(active_indices))
                diagnostics["active_counts"].append(act_cnt)

            # Gradient evaluation
            if all_active:
                A_active = self.A
                z_active = z
                grad_fidelity = self.A_T @ r
                grad_sparsity = (z * inv_sigma2) * w
                grad = grad_fidelity + self.lambda_reg * grad_sparsity
            else:
                A_active = self.A[:, active_indices]
                z_active = z[active_indices]
                res_act = A_active @ z_active - y
                grad_fidelity = A_active.T @ res_act
                w_active = w[active_indices]
                grad_sparsity = (z_active * inv_sigma2) * w_active
                grad = grad_fidelity + self.lambda_reg * grad_sparsity

            # Audit gradient balance at milestones
            if return_diagnostics:
                if "init" not in audited:
                    audited.add("init")
                    self._record_gradient_audit(diagnostics, "init", sigma, grad_fidelity, self.lambda_reg * grad_sparsity)
                for tag, val in [("sigma_1", 1.0), ("sigma_01", 0.1), ("sigma_001", 0.01)]:
                    if tag not in audited and sigma <= val:
                        audited.add(tag)
                        self._record_gradient_audit(diagnostics, tag, sigma, grad_fidelity, self.lambda_reg * grad_sparsity)

            d_active = -grad
            grad_dot_d = -float(np.dot(grad, grad))

            A_d = A_active @ d_active

            step_size = mu
            line_search_success = False
            used_midpoint = False
            accepted_r = r
            accepted_fid = fid
            best_z_active = None

            # Primary candidate evaluation
            r_cand = r + step_size * A_d
            fid_cand = 0.5 * float(np.dot(r_cand, r_cand))
            z_candidate_active = z_active + step_size * d_active
            if all_active:
                w_cand = np.exp((z_candidate_active * z_candidate_active) * neg_inv_2sigma2)
                sparse_cand = -self.lambda_reg * float(w_cand.sum())
            else:
                w_cand_act = np.exp((z_candidate_active * z_candidate_active) * neg_inv_2sigma2)
                sparse_cand = -self.lambda_reg * float(w_cand_act.sum() + w[~active_indices].sum())

            if (fid_cand + sparse_cand) <= cost_current + armijo_c * step_size * grad_dot_d:
                best_z_active = z_candidate_active
                line_search_success = True
                mu = step_size
                if return_diagnostics:
                    diagnostics["mu"].append(step_size)
                accepted_r = r_cand
                accepted_fid = fid_cand
            else:
                # Midpoint evaluation on primary failure
                if use_midpoint:
                    step_mid = step_size * 0.5
                    r_mid = r + step_mid * A_d
                    fid_mid = 0.5 * float(np.dot(r_mid, r_mid))
                    z_mid_active = z_active + step_mid * d_active
                    if all_active:
                        w_mid = np.exp((z_mid_active * z_mid_active) * neg_inv_2sigma2)
                        sparse_mid = -self.lambda_reg * float(w_mid.sum())
                    else:
                        w_mid_act = np.exp((z_mid_active * z_mid_active) * neg_inv_2sigma2)
                        sparse_mid = -self.lambda_reg * float(w_mid_act.sum() + w[~active_indices].sum())

                    if (fid_mid + sparse_mid) <= cost_current + armijo_c * step_mid * grad_dot_d:
                        best_z_active = z_mid_active
                        line_search_success = True
                        mu = step_mid
                        if return_diagnostics:
                            diagnostics["mu"].append(mu)
                            diagnostics["midpoint_acceptances"] += 1
                        used_midpoint = True
                        accepted_r = r_mid
                        accepted_fid = fid_mid

                if not line_search_success:
                    step_size *= beta_decay
                    for bt in range(1, max_backtracks):
                        if return_diagnostics:
                            diagnostics["backtracking_steps"] += 1
                        r_cand = r + step_size * A_d
                        fid_cand = 0.5 * float(np.dot(r_cand, r_cand))
                        z_candidate_active = z_active + step_size * d_active
                        if all_active:
                            w_cand = np.exp((z_candidate_active * z_candidate_active) * neg_inv_2sigma2)
                            sparse_cand = -self.lambda_reg * float(w_cand.sum())
                        else:
                            w_cand_act = np.exp((z_candidate_active * z_candidate_active) * neg_inv_2sigma2)
                            sparse_cand = -self.lambda_reg * float(w_cand_act.sum() + w[~active_indices].sum())

                        if (fid_cand + sparse_cand) <= cost_current + armijo_c * step_size * grad_dot_d:
                            best_z_active = z_candidate_active
                            line_search_success = True
                            mu = step_size
                            if return_diagnostics:
                                diagnostics["mu"].append(mu)
                            accepted_r = r_cand
                            accepted_fid = fid_cand
                            break

                        if use_midpoint:
                            step_mid = step_size * 0.5
                            r_mid = r + step_mid * A_d
                            fid_mid = 0.5 * float(np.dot(r_mid, r_mid))
                            z_mid_active = z_active + step_mid * d_active
                            if all_active:
                                w_mid = np.exp((z_mid_active * z_mid_active) * neg_inv_2sigma2)
                                sparse_mid = -self.lambda_reg * float(w_mid.sum())
                            else:
                                w_mid_act = np.exp((z_mid_active * z_mid_active) * neg_inv_2sigma2)
                                sparse_mid = -self.lambda_reg * float(w_mid_act.sum() + w[~active_indices].sum())

                            if (fid_mid + sparse_mid) <= cost_current + armijo_c * step_mid * grad_dot_d:
                                best_z_active = z_mid_active
                                line_search_success = True
                                mu = step_mid
                                if return_diagnostics:
                                    diagnostics["mu"].append(mu)
                                    diagnostics["midpoint_acceptances"] += 1
                                used_midpoint = True
                                accepted_r = r_mid
                                accepted_fid = fid_mid
                                break

                        step_size *= beta_decay

            if not line_search_success:
                if return_diagnostics:
                    diagnostics["failed_line_searches"] += 1
                    diagnostics["mu"].append(step_size)
                    diagnostics["relative_change"].append(0.0)
                mu *= 0.5
                continue

            if all_active:
                z_new = best_z_active
            else:
                z_new = z.copy()
                z_new[active_indices] = best_z_active

            rel_change = float(np.linalg.norm(z_new - z) / (np.linalg.norm(z) + 1e-8))
            if return_diagnostics:
                diagnostics["relative_change"].append(rel_change)
                diagnostics["accepted_steps"] += 1

            z = z_new
            r = accepted_r
            fid = accepted_fid

            # Approved baseline stopping condition: both rel_change < tol and sigma <= sigma_min
            if self._check_stop_condition(rel_change, sigma, sigma_min, iteration):
                stop_reason = "converged"
                break

            sigma = max(sigma * decrease_factor, sigma_min)

        final_residual_norm = float(np.linalg.norm(self.A @ z - y))
        if return_diagnostics:
            diagnostics["final_residual"] = final_residual_norm
            diagnostics["final_sigma"] = float(sigma)
            diagnostics["iterations"] = len(diagnostics["objective"])
            diagnostics["stop_reason"] = stop_reason
            diagnostics["mean_active_count"] = float(np.mean(diagnostics["active_counts"])) if diagnostics["active_counts"] else self.N
            diagnostics["min_active_count"] = int(np.min(diagnostics["active_counts"])) if diagnostics["active_counts"] else self.N
            diagnostics["active_support_ratio"] = float(np.sum(diagnostics["active_counts"])) / (diagnostics["iterations"] * self.N) if diagnostics["iterations"] > 0 else 1.0
            diagnostics["mean_mu"] = float(np.mean(diagnostics["mu"])) if diagnostics["mu"] else mu
            return z, diagnostics
        return z

    def _check_stop_condition(self, rel_change, sigma, sigma_min, iteration):
        """Baseline stop condition: requires sigma <= sigma_min."""
        return rel_change < self.tol and sigma <= sigma_min

    def _record_gradient_audit(self, diagnostics, tag, sigma, g_fid, g_sparse):
        norm_fid = float(np.linalg.norm(g_fid))
        norm_sparse = float(np.linalg.norm(g_sparse))
        norm_total = float(np.linalg.norm(g_fid + g_sparse))
        ratio = norm_sparse / max(norm_fid, 1e-12)
        diagnostics["gradient_audits"][tag] = {
            "sigma": float(sigma),
            "norm_fidelity": norm_fid,
            "norm_sparsity": norm_sparse,
            "norm_total": norm_total,
            "gradient_ratio": ratio,
        }


class SolverV7A1AdaptiveExit(SolverV7OptBase):
    """
    Variant A1: Adaptive Early Exit.
    Allows convergence when the relative update remains < tol for K consecutive
    iterations, without requiring sigma <= sigma_min.
    """

    def __init__(self, A, lambda_reg=0.1, tol=1e-5, patience=2, min_iter=5):
        super().__init__(A, lambda_reg, tol)
        self.patience = int(patience)
        self.min_iter = int(min_iter)
        self.consecutive_stable = 0

    def _check_stop_condition(self, rel_change, sigma, sigma_min, iteration):
        # Traditional baseline check
        if rel_change < self.tol and sigma <= sigma_min:
            return True

        # Adaptive consecutive-stability early exit
        if iteration >= self.min_iter:
            if rel_change < self.tol:
                self.consecutive_stable += 1
                if self.consecutive_stable >= self.patience:
                    return True
            else:
                self.consecutive_stable = 0
        return False

    def denoise_patch(self, y, **kwargs):
        self.consecutive_stable = 0
        return super().denoise_patch(y, **kwargs)


class SolverV7A2ScaleCoupled(SolverV7OptBase):
    """
    Variant A2: Scale-Coupled Regularization.
    Sets lambda(sigma) = lambda_0 * sigma^2, ensuring bounded gradient magnitude
    across all sigma values.
    Analytical gradient: grad_sparsity = lambda_0 * z * exp(-z^2 / (2*sigma^2)).
    """

    def __init__(self, A, lambda_0=0.1, tol=1e-5):
        super().__init__(A, lambda_reg=lambda_0, tol=tol)
        self.lambda_0 = float(lambda_0)

    def denoise_patch(
        self,
        y,
        initial_sigma=None,
        decrease_factor=0.95,
        sigma_min=0.01,
        max_iter=150,
        initial_mu=0.2,
        beta_decay=0.5,
        armijo_c=1e-4,
        max_backtracks=10,
        support_threshold_multiplier=1e-5,
        support_reopen_interval=3,
        use_midpoint=True,
        return_diagnostics=True,
    ):
        y = np.asarray(y, dtype=float).reshape(-1)
        z = self.P_init @ y

        z_max = float(np.max(np.abs(z)))
        if initial_sigma is None:
            sigma = 2.5 * z_max if z_max > 0 else 1.0
        else:
            sigma = float(initial_sigma)
        sigma = max(sigma, sigma_min)

        mu = float(initial_mu)

        diagnostics = {
            "objective": [],
            "fidelity": [],
            "sparsity": [],
            "sigma": [],
            "active_counts": [],
            "mu": [],
            "relative_change": [],
            "failed_line_searches": 0,
            "backtracking_steps": 0,
            "accepted_steps": 0,
            "midpoint_acceptances": 0,
            "gradient_audits": {},
        }

        r = self.A @ z - y
        fid = 0.5 * float(np.dot(r, r))
        stop_reason = "max_iter_reached"

        audited = set()

        for iteration in range(max_iter):
            # Scale-coupled lambda: lambda(sigma) = lambda_0 * sigma^2
            curr_lambda = self.lambda_0 * (sigma * sigma)

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

            neg_inv_2sigma2 = -0.5 / (sigma * sigma)
            w = np.exp((z * z) * neg_inv_2sigma2)
            # Sparsity penalty with curr_lambda
            sparse_val = -curr_lambda * float(w.sum())
            cost_current = fid + sparse_val

            if return_diagnostics:
                diagnostics["objective"].append(cost_current)
                diagnostics["fidelity"].append(fid)
                diagnostics["sparsity"].append(sparse_val)
                diagnostics["sigma"].append(sigma)
                act_cnt = self.N if all_active else int(np.count_nonzero(active_indices))
                diagnostics["active_counts"].append(act_cnt)

            # Gradient evaluation:
            # Notice: curr_lambda * (z / sigma^2) * w = (lambda_0 * sigma^2) * (z / sigma^2) * w = lambda_0 * z * w
            if all_active:
                A_active = self.A
                z_active = z
                grad_fidelity = self.A_T @ r
                grad_sparsity = self.lambda_0 * (z * w)
                grad = grad_fidelity + grad_sparsity
            else:
                A_active = self.A[:, active_indices]
                z_active = z[active_indices]
                res_act = A_active @ z_active - y
                grad_fidelity = A_active.T @ res_act
                w_active = w[active_indices]
                grad_sparsity = self.lambda_0 * (z_active * w_active)
                grad = grad_fidelity + grad_sparsity

            # Audit gradient balance
            if return_diagnostics:
                if "init" not in audited:
                    audited.add("init")
                    self._record_gradient_audit(diagnostics, "init", sigma, grad_fidelity, grad_sparsity)
                for tag, val in [("sigma_1", 1.0), ("sigma_01", 0.1), ("sigma_001", 0.01)]:
                    if tag not in audited and sigma <= val:
                        audited.add(tag)
                        self._record_gradient_audit(diagnostics, tag, sigma, grad_fidelity, grad_sparsity)

            d_active = -grad
            grad_dot_d = -float(np.dot(grad, grad))

            A_d = A_active @ d_active

            step_size = mu
            line_search_success = False
            used_midpoint = False
            accepted_r = r
            accepted_fid = fid
            best_z_active = None

            # Primary candidate evaluation
            r_cand = r + step_size * A_d
            fid_cand = 0.5 * float(np.dot(r_cand, r_cand))
            z_candidate_active = z_active + step_size * d_active
            if all_active:
                w_cand = np.exp((z_candidate_active * z_candidate_active) * neg_inv_2sigma2)
                sparse_cand = -curr_lambda * float(w_cand.sum())
            else:
                w_cand_act = np.exp((z_candidate_active * z_candidate_active) * neg_inv_2sigma2)
                sparse_cand = -curr_lambda * float(w_cand_act.sum() + w[~active_indices].sum())

            if (fid_cand + sparse_cand) <= cost_current + armijo_c * step_size * grad_dot_d:
                best_z_active = z_candidate_active
                line_search_success = True
                mu = step_size
                if return_diagnostics:
                    diagnostics["mu"].append(step_size)
                accepted_r = r_cand
                accepted_fid = fid_cand
            else:
                if use_midpoint:
                    step_mid = step_size * 0.5
                    r_mid = r + step_mid * A_d
                    fid_mid = 0.5 * float(np.dot(r_mid, r_mid))
                    z_mid_active = z_active + step_mid * d_active
                    if all_active:
                        w_mid = np.exp((z_mid_active * z_mid_active) * neg_inv_2sigma2)
                        sparse_mid = -curr_lambda * float(w_mid.sum())
                    else:
                        w_mid_act = np.exp((z_mid_active * z_mid_active) * neg_inv_2sigma2)
                        sparse_mid = -curr_lambda * float(w_mid_act.sum() + w[~active_indices].sum())

                    if (fid_mid + sparse_mid) <= cost_current + armijo_c * step_mid * grad_dot_d:
                        best_z_active = z_mid_active
                        line_search_success = True
                        mu = step_mid
                        if return_diagnostics:
                            diagnostics["mu"].append(mu)
                            diagnostics["midpoint_acceptances"] += 1
                        used_midpoint = True
                        accepted_r = r_mid
                        accepted_fid = fid_mid

                if not line_search_success:
                    step_size *= beta_decay
                    for bt in range(1, max_backtracks):
                        if return_diagnostics:
                            diagnostics["backtracking_steps"] += 1
                        r_cand = r + step_size * A_d
                        fid_cand = 0.5 * float(np.dot(r_cand, r_cand))
                        z_candidate_active = z_active + step_size * d_active
                        if all_active:
                            w_cand = np.exp((z_candidate_active * z_candidate_active) * neg_inv_2sigma2)
                            sparse_cand = -curr_lambda * float(w_cand.sum())
                        else:
                            w_cand_act = np.exp((z_candidate_active * z_candidate_active) * neg_inv_2sigma2)
                            sparse_cand = -curr_lambda * float(w_cand_act.sum() + w[~active_indices].sum())

                        if (fid_cand + sparse_cand) <= cost_current + armijo_c * step_size * grad_dot_d:
                            best_z_active = z_candidate_active
                            line_search_success = True
                            mu = step_size
                            if return_diagnostics:
                                diagnostics["mu"].append(mu)
                            accepted_r = r_cand
                            accepted_fid = fid_cand
                            break

                        if use_midpoint:
                            step_mid = step_size * 0.5
                            r_mid = r + step_mid * A_d
                            fid_mid = 0.5 * float(np.dot(r_mid, r_mid))
                            z_mid_active = z_active + step_mid * d_active
                            if all_active:
                                w_mid = np.exp((z_mid_active * z_mid_active) * neg_inv_2sigma2)
                                sparse_mid = -curr_lambda * float(w_mid.sum())
                            else:
                                w_mid_act = np.exp((z_mid_active * z_mid_active) * neg_inv_2sigma2)
                                sparse_mid = -curr_lambda * float(w_mid_act.sum() + w[~active_indices].sum())

                            if (fid_mid + sparse_mid) <= cost_current + armijo_c * step_mid * grad_dot_d:
                                best_z_active = z_mid_active
                                line_search_success = True
                                mu = step_mid
                                if return_diagnostics:
                                    diagnostics["mu"].append(mu)
                                    diagnostics["midpoint_acceptances"] += 1
                                used_midpoint = True
                                accepted_r = r_mid
                                accepted_fid = fid_mid
                                break

                        step_size *= beta_decay

            if not line_search_success:
                if return_diagnostics:
                    diagnostics["failed_line_searches"] += 1
                    diagnostics["mu"].append(step_size)
                    diagnostics["relative_change"].append(0.0)
                mu *= 0.5
                continue

            if all_active:
                z_new = best_z_active
            else:
                z_new = z.copy()
                z_new[active_indices] = best_z_active

            rel_change = float(np.linalg.norm(z_new - z) / (np.linalg.norm(z) + 1e-8))
            if return_diagnostics:
                diagnostics["relative_change"].append(rel_change)
                diagnostics["accepted_steps"] += 1

            z = z_new
            r = accepted_r
            fid = accepted_fid

            if self._check_stop_condition(rel_change, sigma, sigma_min, iteration):
                stop_reason = "converged"
                break

            sigma = max(sigma * decrease_factor, sigma_min)

        final_residual_norm = float(np.linalg.norm(self.A @ z - y))
        if return_diagnostics:
            diagnostics["final_residual"] = final_residual_norm
            diagnostics["final_sigma"] = float(sigma)
            diagnostics["iterations"] = len(diagnostics["objective"])
            diagnostics["stop_reason"] = stop_reason
            diagnostics["mean_active_count"] = float(np.mean(diagnostics["active_counts"])) if diagnostics["active_counts"] else self.N
            diagnostics["min_active_count"] = int(np.min(diagnostics["active_counts"])) if diagnostics["active_counts"] else self.N
            diagnostics["active_support_ratio"] = float(np.sum(diagnostics["active_counts"])) / (diagnostics["iterations"] * self.N) if diagnostics["iterations"] > 0 else 1.0
            diagnostics["mean_mu"] = float(np.mean(diagnostics["mu"])) if diagnostics["mu"] else mu
            return z, diagnostics
        return z


class SolverV7A3SupportThreshold(SolverV7OptBase):
    """
    Variant A3: Active Support Threshold Sensitivity Study.
    Runs V7 with configurable support_threshold_multiplier (e.g., 1e-5, 1e-4, 1e-3, 1e-2).
    """

    def __init__(self, A, lambda_reg=0.1, tol=1e-5, threshold_multiplier=1e-5):
        super().__init__(A, lambda_reg, tol)
        self.threshold_multiplier = float(threshold_multiplier)

    def denoise_patch(self, y, **kwargs):
        kwargs["support_threshold_multiplier"] = self.threshold_multiplier
        return super().denoise_patch(y, **kwargs)


class SolverV7A4Continuation(SolverV7OptBase):
    """
    Variant A4: Continuation Schedule Study.
    Runs V7 with configurable geometric continuation decay (e.g. 0.90, 0.95, 0.98).
    """

    def __init__(self, A, lambda_reg=0.1, tol=1e-5, decrease_factor=0.95):
        super().__init__(A, lambda_reg, tol)
        self.decrease_factor = float(decrease_factor)

    def denoise_patch(self, y, **kwargs):
        kwargs["decrease_factor"] = self.decrease_factor
        return super().denoise_patch(y, **kwargs)


class SolverV7A5TwoStage(SolverV7OptBase):
    """
    Variant A5: Two-Stage Sparse Recovery.
    Stage 1: ASL-SR-DPT determines candidate sparse support set S.
    Stage 2: Debiasing via Least-Squares on support S:
             theta_S = argmin ||A_S theta_S - y||_2 = pinv(A_S) @ y.
    """

    def __init__(self, A, lambda_reg=0.1, tol=1e-5, support_prune_threshold=1e-3):
        super().__init__(A, lambda_reg, tol)
        self.support_prune_threshold = float(support_prune_threshold)

    def denoise_patch(self, y, return_diagnostics=True, **kwargs):
        kwargs.pop("return_diagnostics", None)
        # Stage 1: Standard ASL-SR-DPT solve
        z_stage1, diag1 = super().denoise_patch(y, return_diagnostics=True, **kwargs)

        # Stage 2: Identify support and debias via least squares
        support_mask = np.abs(z_stage1) > self.support_prune_threshold
        support_count = int(np.count_nonzero(support_mask))

        if support_count == 0 or support_count >= self.M:
            # Fallback to pseudoinverse or top-M if ill-conditioned
            if support_count == 0:
                support_mask = np.ones(self.N, dtype=bool)
                support_count = self.N
            else:
                top_indices = np.argsort(np.abs(z_stage1))[-self.M :]
                support_mask = np.zeros(self.N, dtype=bool)
                support_mask[top_indices] = True
                support_count = self.M

        A_supp = self.A[:, support_mask]
        y_vec = np.asarray(y, dtype=float).reshape(-1)
        z_supp = np.linalg.pinv(A_supp) @ y_vec

        z_stage2 = np.zeros(self.N, dtype=float)
        z_stage2[support_mask] = z_supp

        # Update diagnostics to reflect stage 2
        final_residual = float(np.linalg.norm(self.A @ z_stage2 - y_vec))
        diag1["final_residual"] = final_residual
        diag1["stage2_support_size"] = support_count
        diag1["stage1_residual"] = float(np.linalg.norm(self.A @ z_stage1 - y_vec))

        if return_diagnostics:
            return z_stage2, diag1
        return z_stage2


class SolverV7A6DCPreservation:
    """
    Variant A6: DC-Preserving Architecture.
    Separates the DC coefficient (spatial mean) and performs recovery exclusively
    on the 63 AC coefficients using A_AC in R^(37 x 63).
    Restores patch via [theta_DC; theta_AC].
    """

    def __init__(self, A_ac, lambda_reg=0.1, tol=1e-5):
        self.A_ac = np.asarray(A_ac, dtype=float)
        self.M_ac, self.N_ac = self.A_ac.shape  # 37 x 63
        self.solver_ac = SolverV7OptBase(self.A_ac, lambda_reg=lambda_reg, tol=tol)

    def denoise_patch_dc(self, y_ac, dc_val, return_diagnostics=True, **kwargs):
        kwargs.pop("return_diagnostics", None)
        z_ac, diag = self.solver_ac.denoise_patch(y_ac, return_diagnostics=True, **kwargs)

        z_full = np.empty(64, dtype=float)
        z_full[0] = dc_val
        z_full[1:] = z_ac

        y_vec = np.asarray(y_ac, dtype=float).reshape(-1)
        diag["ac_residual"] = float(np.linalg.norm(self.A_ac @ z_ac - y_vec))
        if return_diagnostics:
            return z_full, diag
        return z_full

