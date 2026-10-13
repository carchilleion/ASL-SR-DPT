"""
ASL-SR-DPT Final Benchmark: Statistical Analysis & Hypothesis Testing Suite (30,600 Evaluations)

Methodology:
1. Aggregates the 50 trials for each image-noise-method condition to yield
   exactly 68 independent image-level observations per method and noise level.
   (Patches are treated strictly as computational units, not independent statistical units).
2. Computes descriptive statistics (Mean, Std, Median, IQR, Min, Max).
3. Conducts paired two-sided Wilcoxon signed-rank tests at alpha = 0.05.
4. Applies Holm-Bonferroni step-down correction for family-wise error rate control.
5. Computes matched-pairs rank-biserial correlation (r) effect sizes and 95% bootstrap CIs.
6. Generates publication tables and high-resolution figures.

Outputs:
- results/final_benchmark/final_descriptive_summary.csv
- results/final_benchmark/final_inferential_tests.csv
- results/final_benchmark/final_benchmark_psnr_ssim_curves.png
- results/final_benchmark/final_benchmark_runtime_boxplots.png
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL_BENCHMARK_DIR = os.path.join(REPO_ROOT, "results", "final_benchmark")
MERGED_CSV_PATH = os.path.join(FINAL_BENCHMARK_DIR, "raw_results_merged.csv")
DESCRIPTIVE_CSV_PATH = os.path.join(FINAL_BENCHMARK_DIR, "final_descriptive_summary.csv")
INFERENTIAL_CSV_PATH = os.path.join(FINAL_BENCHMARK_DIR, "final_inferential_tests.csv")
PSNR_SSIM_PLOT_PATH = os.path.join(FINAL_BENCHMARK_DIR, "final_benchmark_psnr_ssim_curves.png")
RUNTIME_PLOT_PATH = os.path.join(FINAL_BENCHMARK_DIR, "final_benchmark_runtime_boxplots.png")

NOISE_LEVELS = [15.0, 25.0, 50.0]
METHODS_ORDER = [
    "Noise-Adaptive ASL-SR-DPT",
    "OMP",
    "LASSO-ADMM",
]


def bootstrap_ci(diff_array, n_boot=10000, alpha=0.05, seed=20260908):
    """Computes two-sided bootstrap confidence interval for the mean difference."""
    rng = np.random.default_rng(seed)
    n = len(diff_array)
    if n == 0:
        return (0.0, 0.0)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(diff_array[indices], axis=1)
    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)
    return float(np.percentile(boot_means, low_pct)), float(np.percentile(boot_means, high_pct))


def compute_rank_biserial(w_pos, w_neg):
    """Computes matched-pairs rank-biserial correlation effect size r."""
    total_w = w_pos + w_neg
    if total_w == 0:
        return 0.0
    return float((w_pos - w_neg) / total_w)


def run_statistical_analysis():
    print("=" * 80)
    print("ASL-SR-DPT Final Benchmark: Statistical Analysis & Hypothesis Testing")
    print("=" * 80)

    if not os.path.exists(MERGED_CSV_PATH):
        print(f"[Error] Merged CSV not found at: {MERGED_CSV_PATH}")
        return

    df = pd.read_csv(MERGED_CSV_PATH)
    print(f"Loaded merged benchmark records: {len(df)}")

    for c in ["psnr", "ssim", "mse", "runtime_seconds", "noise_sigma", "trial"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # -------------------------------------------------------------------------
    # 1. Aggregate 50 Trials per (image_id, noise_sigma, method)
    #    Yields exactly 68 independent image-level observations
    # -------------------------------------------------------------------------
    img_agg = df.groupby(["image_id", "noise_sigma", "method"]).agg(
        psnr_mean=("psnr", "mean"),
        ssim_mean=("ssim", "mean"),
        mse_mean=("mse", "mean"),
        runtime_mean=("runtime_seconds", "mean"),
    ).reset_index()

    print(f"Aggregated image-level independent observations: {len(img_agg)} (68 images x 3 noises x 3 methods = {68 * 3 * 3})")

    # -------------------------------------------------------------------------
    # 2. Descriptive Summary
    # -------------------------------------------------------------------------
    desc_rows = []
    for m in METHODS_ORDER:
        m_df = img_agg[img_agg["method"] == m]

        # Overall row
        desc_rows.append({
            "method": m,
            "noise_sigma": "ALL",
            "images_count": len(m_df["image_id"].unique()),
            "total_evals": len(df[df["method"] == m]),
            "psnr_mean": m_df["psnr_mean"].mean(),
            "psnr_std": m_df["psnr_mean"].std(),
            "psnr_median": m_df["psnr_mean"].median(),
            "psnr_iqr": m_df["psnr_mean"].quantile(0.75) - m_df["psnr_mean"].quantile(0.25),
            "ssim_mean": m_df["ssim_mean"].mean(),
            "ssim_std": m_df["ssim_mean"].std(),
            "ssim_median": m_df["ssim_mean"].median(),
            "ssim_iqr": m_df["ssim_mean"].quantile(0.75) - m_df["ssim_mean"].quantile(0.25),
            "runtime_mean": m_df["runtime_mean"].mean(),
            "runtime_median": m_df["runtime_mean"].median(),
        })

        # Per noise level
        for n in NOISE_LEVELS:
            mn_df = m_df[m_df["noise_sigma"] == n]
            desc_rows.append({
                "method": m,
                "noise_sigma": str(int(n)),
                "images_count": len(mn_df),
                "total_evals": len(df[(df["method"] == m) & (df["noise_sigma"] == n)]),
                "psnr_mean": mn_df["psnr_mean"].mean(),
                "psnr_std": mn_df["psnr_mean"].std(),
                "psnr_median": mn_df["psnr_mean"].median(),
                "psnr_iqr": mn_df["psnr_mean"].quantile(0.75) - mn_df["psnr_mean"].quantile(0.25),
                "ssim_mean": mn_df["ssim_mean"].mean(),
                "ssim_std": mn_df["ssim_mean"].std(),
                "ssim_median": mn_df["ssim_mean"].median(),
                "ssim_iqr": mn_df["ssim_mean"].quantile(0.75) - mn_df["ssim_mean"].quantile(0.25),
                "runtime_mean": mn_df["runtime_mean"].mean(),
                "runtime_median": mn_df["runtime_mean"].median(),
            })

    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(DESCRIPTIVE_CSV_PATH, index=False)
    print(f"[Saved] Descriptive Summary CSV -> {DESCRIPTIVE_CSV_PATH}")

    # -------------------------------------------------------------------------
    # 3. Inferential Statistical Hypothesis Testing
    #    Two-sided Wilcoxon signed-rank tests with Holm-Bonferroni correction
    # -------------------------------------------------------------------------
    test_rows = []
    comparisons = [
        ("Noise-Adaptive ASL-SR-DPT", "OMP"),
        ("Noise-Adaptive ASL-SR-DPT", "LASSO-ADMM"),
    ]

    for metric, metric_col in [("PSNR", "psnr_mean"), ("SSIM", "ssim_mean")]:
        for n in NOISE_LEVELS:
            for m_test, m_base in comparisons:
                sub_test = img_agg[(img_agg["method"] == m_test) & (img_agg["noise_sigma"] == n)].sort_values("image_id")
                sub_base = img_agg[(img_agg["method"] == m_base) & (img_agg["noise_sigma"] == n)].sort_values("image_id")

                v_test = sub_test[metric_col].values
                v_base = sub_base[metric_col].values
                diff = v_test - v_base

                # Wilcoxon signed-rank test
                res = stats.wilcoxon(v_test, v_base, alternative="two-sided")
                w_stat = float(res.statistic)
                p_val = float(res.pvalue)

                # Effect size rank biserial
                ranks = stats.rankdata(np.abs(diff))
                pos_ranks = np.sum(ranks[diff > 0])
                neg_ranks = np.sum(ranks[diff < 0])
                r_effect = compute_rank_biserial(pos_ranks, neg_ranks)

                # Bootstrap 95% CI
                ci_low, ci_high = bootstrap_ci(diff)

                test_rows.append({
                    "metric": metric,
                    "noise_sigma": float(n),
                    "comparison": f"{m_test} vs {m_base}",
                    "test_method": m_test,
                    "baseline_method": m_base,
                    "n_pairs": len(diff),
                    "mean_diff": float(np.mean(diff)),
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                    "wilcoxon_w": w_stat,
                    "raw_p_value": p_val,
                    "effect_size_r": r_effect,
                })

    test_df = pd.DataFrame(test_rows)

    # Apply Holm-Bonferroni correction per metric family
    for metric in ["PSNR", "SSIM"]:
        m_mask = test_df["metric"] == metric
        sub_indices = test_df[m_mask].sort_values("raw_p_value").index.tolist()
        m_total = len(sub_indices)

        p_adj_list = [0.0] * m_total
        sig_list = [False] * m_total
        running_max = 0.0

        for rank_k, idx in enumerate(sub_indices):
            raw_p = test_df.loc[idx, "raw_p_value"]
            factor = m_total - rank_k
            adj_p = min(1.0, raw_p * factor)
            # Monotonicity adjustment
            running_max = max(running_max, adj_p)
            test_df.loc[idx, "holm_bonferroni_p"] = running_max
            test_df.loc[idx, "statistically_significant"] = (running_max < 0.05)

    test_df.to_csv(INFERENTIAL_CSV_PATH, index=False)
    print(f"[Saved] Inferential Tests CSV -> {INFERENTIAL_CSV_PATH}")

    # -------------------------------------------------------------------------
    # 4. Generate Publication Figures
    # -------------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # PSNR Curve
    for m in METHODS_ORDER:
        sub = desc_df[(desc_df["method"] == m) & (desc_df["noise_sigma"] != "ALL")]
        noises = [float(x) for x in sub["noise_sigma"]]
        means = sub["psnr_mean"].values
        stds = sub["psnr_std"].values
        axes[0].errorbar(noises, means, yerr=stds, label=m, marker="o", capsize=4, linewidth=2)

    axes[0].set_title("Reconstruction Quality (PSNR) vs. Noise Regime", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(r"Noise Standard Deviation ($\sigma$)", fontsize=11)
    axes[0].set_ylabel("Mean PSNR (dB)", fontsize=11)
    axes[0].set_xticks(NOISE_LEVELS)
    axes[0].legend(loc="upper right", frameon=True)

    # SSIM Curve
    for m in METHODS_ORDER:
        sub = desc_df[(desc_df["method"] == m) & (desc_df["noise_sigma"] != "ALL")]
        noises = [float(x) for x in sub["noise_sigma"]]
        means = sub["ssim_mean"].values
        stds = sub["ssim_std"].values
        axes[1].errorbar(noises, means, yerr=stds, label=m, marker="s", capsize=4, linewidth=2)

    axes[1].set_title("Structural Similarity (SSIM) vs. Noise Regime", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(r"Noise Standard Deviation ($\sigma$)", fontsize=11)
    axes[1].set_ylabel("Mean SSIM", fontsize=11)
    axes[1].set_xticks(NOISE_LEVELS)
    axes[1].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(PSNR_SSIM_PLOT_PATH, dpi=200)
    plt.close()
    print(f"[Saved] PSNR/SSIM Plot -> {PSNR_SSIM_PLOT_PATH}")

    # Runtime Boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data = [img_agg[img_agg["method"] == m]["runtime_mean"].values for m in METHODS_ORDER]
    ax.boxplot(plot_data, labels=[m.replace("ASL-SR-DPT", "ASL") for m in METHODS_ORDER], patch_artist=True)
    ax.set_title("Execution Runtime Distribution Across 68 Images", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Full-Image Solve Time (seconds)", fontsize=11)
    plt.tight_layout()
    plt.savefig(RUNTIME_PLOT_PATH, dpi=200)
    plt.close()
    print(f"[Saved] Runtime Plot -> {RUNTIME_PLOT_PATH}")
    print("=" * 80)
    print("Statistical Analysis Complete.")
    print("=" * 80)


def main():
    run_statistical_analysis()


if __name__ == "__main__":
    main()
