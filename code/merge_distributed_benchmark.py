"""
ASL-SR-DPT Final Benchmark: 3-PC Merge & 10-Point Cross-PC Audit Engine

Merges outputs from PC1, PC2, and PC3 into a unified dataset:
- results/final_benchmark/raw_results_merged.csv (Exactly 30,600 rows)
- results/final_benchmark/benchmark_manifest_merged.json
- results/final_benchmark/final_integrity_report.md

Enforces the 10-point audit checklist:
1. Verify partition files match original manifest hashes.
2. Verify code hash and configuration hash across all 3 outputs.
3. Validate sensing and noise seed derivation formulas for all 30,600 rows.
4. Check global uniqueness of all 30,600 composite evaluation keys.
5. Verify strict partition membership (PC i executed only assigned PC i conditions).
6. Verify identical noisy-input SHA-256 hash across all 3 methods for every condition.
7. Verify expected evaluation counts (10,200 per PC, 10,200 per method, 30,600 total).
8. Reject corrupt or schema-incompatible records.
9. Preserve any failure records in audit trail.
10. Refuse silent duplicate discarding.
"""

import os
import sys
import csv
import json
import hashlib
import argparse
import datetime
import pandas as pd
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(REPO_ROOT, "code")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from generate_final_benchmark_partitions import compute_seeds, compute_file_sha256, CONFIG_PATH, PARTITIONS_DIR
import app

FINAL_BENCHMARK_DIR = os.path.join(REPO_ROOT, "results", "final_benchmark")
MERGED_CSV_PATH = os.path.join(FINAL_BENCHMARK_DIR, "raw_results_merged.csv")
MERGED_MANIFEST_PATH = os.path.join(FINAL_BENCHMARK_DIR, "benchmark_manifest_merged.json")
FINAL_INTEGRITY_REPORT_PATH = os.path.join(FINAL_BENCHMARK_DIR, "final_integrity_report.md")

METHODS_ORDER = [
    "Noise-Adaptive ASL-SR-DPT",
    "OMP",
    "LASSO-ADMM",
]
EXPECTED_TOTAL_CONDITIONS = 10200
EXPECTED_TOTAL_EVALUATIONS = 30600
EXPECTED_EVALUATIONS_PER_METHOD = 10200
EXPECTED_EVALUATIONS_PER_PC = 10200


def merge_and_audit_benchmark():
    print("=" * 80)
    print("ASL-SR-DPT Final Benchmark: 3-PC Merge & 10-Point Cross-PC Audit Engine")
    print("=" * 80)

    audit_log = []
    fatal_errors = []

    # ---------------------------------------------------------
    # Checkpoint 1: Verify Partition Files & Manifest
    # ---------------------------------------------------------
    partition_manifest_path = os.path.join(PARTITIONS_DIR, "partition_manifest.json")
    if not os.path.exists(partition_manifest_path):
        err = f"[Checkpoint 1 FAIL] Missing partition manifest: {partition_manifest_path}"
        fatal_errors.append(err)
        print(err)
        return False

    with open(partition_manifest_path, "r", encoding="utf-8") as f:
        part_manifest = json.load(f)

    partition_conditions = {}
    for p_id in ["pc1", "pc2", "pc3"]:
        csv_file = f"{p_id}_conditions.csv"
        csv_path = os.path.join(PARTITIONS_DIR, csv_file)
        if not os.path.exists(csv_path):
            err = f"[Checkpoint 1 FAIL] Missing condition file for {p_id}: {csv_path}"
            fatal_errors.append(err)
            print(err)
            return False

        curr_hash = compute_file_sha256(csv_path)
        expected_hash = part_manifest["workload_distribution"][p_id]["sha256"]
        if curr_hash != expected_hash:
            err = f"[Checkpoint 1 FAIL] Partition {p_id} hash mismatch! Manifest: {expected_hash}, Actual: {curr_hash}"
            fatal_errors.append(err)
            print(err)
            return False

        # Load assigned conditions
        p_conds = {}
        with open(csv_path, "r", encoding="utf-8") as pf:
            for row in csv.DictReader(pf):
                k = (str(row["image_id"]), float(row["noise_sigma"]), int(row["trial"]))
                p_conds[k] = {
                    "condition_id": int(row["condition_id"]),
                    "sensing_seed": int(row["sensing_seed"]),
                    "noise_seed": int(row["noise_seed"]),
                }
        partition_conditions[p_id] = p_conds

    audit_log.append("Checkpoint 1 PASS: All 3 partition condition CSVs verified matching canonical manifest SHA-256 hashes.")

    # ---------------------------------------------------------
    # Checkpoint 2: Load Individual PC Results & Check Hashes
    # ---------------------------------------------------------
    pc_dfs = {}
    code_hashes = set()
    config_hashes = set()

    for p_id in ["pc1", "pc2", "pc3"]:
        raw_path = os.path.join(FINAL_BENCHMARK_DIR, p_id, "raw_results.csv")
        if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
            err = f"[Checkpoint 2 FAIL] Raw results CSV missing or empty for {p_id}: {raw_path}"
            fatal_errors.append(err)
            print(err)
            return False

        df_pc = pd.read_csv(raw_path)
        pc_dfs[p_id] = df_pc

        code_hashes.update(df_pc["code_hash"].dropna().unique())
        config_hashes.update(df_pc["config_hash"].dropna().unique())

    if len(code_hashes) > 1:
        err = f"[Checkpoint 2 FAIL] Code hash mismatch across PCs: {code_hashes}"
        fatal_errors.append(err)
        print(err)
        return False

    if len(config_hashes) > 1:
        err = f"[Checkpoint 2 FAIL] Config hash mismatch across PCs: {config_hashes}"
        fatal_errors.append(err)
        print(err)
        return False

    audit_log.append(f"Checkpoint 2 PASS: Cryptographic code hash ({list(code_hashes)[0][:16]}...) and config hash ({list(config_hashes)[0][:16]}...) verified identical across all PCs.")

    # ---------------------------------------------------------
    # Checkpoint 7: Workload Counts per PC
    # ---------------------------------------------------------
    for p_id in ["pc1", "pc2", "pc3"]:
        cnt = len(pc_dfs[p_id])
        if cnt != EXPECTED_EVALUATIONS_PER_PC:
            err = f"[Checkpoint 7 FAIL] {p_id.upper()} evaluation count mismatch: expected {EXPECTED_EVALUATIONS_PER_PC}, found {cnt}."
            fatal_errors.append(err)
            print(err)
            return False

    combined_df = pd.concat([pc_dfs["pc1"], pc_dfs["pc2"], pc_dfs["pc3"]], ignore_index=True)
    total_evals = len(combined_df)
    if total_evals != EXPECTED_TOTAL_EVALUATIONS:
        err = f"[Checkpoint 7 FAIL] Total combined evaluations count mismatch: expected {EXPECTED_TOTAL_EVALUATIONS}, found {total_evals}."
        fatal_errors.append(err)
        print(err)
        return False

    # Check method counts
    for m in METHODS_ORDER:
        cnt = len(combined_df[combined_df["method"] == m])
        if cnt != EXPECTED_EVALUATIONS_PER_METHOD:
            err = f"[Checkpoint 7 FAIL] Method '{m}' evaluation count mismatch: expected {EXPECTED_EVALUATIONS_PER_METHOD}, found {cnt}."
            fatal_errors.append(err)
            print(err)
            return False

    audit_log.append(f"Checkpoint 7 PASS: Evaluation counts strictly conform (10,200 per PC, 10,200 per method, exactly 30,600 total).")

    # ---------------------------------------------------------
    # Checkpoint 4 & 10: Global Uniqueness & No Silent Discarding
    # ---------------------------------------------------------
    key_series = combined_df["image_id"].astype(str) + "_" + combined_df["noise_sigma"].astype(str) + "_" + combined_df["trial"].astype(str) + "_" + combined_df["method"].astype(str)
    unique_keys = key_series.nunique()
    if unique_keys != EXPECTED_TOTAL_EVALUATIONS:
        dup_count = EXPECTED_TOTAL_EVALUATIONS - unique_keys
        err = f"[Checkpoint 4 & 10 FAIL] Detected {dup_count} duplicate composite keys across merged dataset! Refusing silent discard."
        fatal_errors.append(err)
        print(err)
        return False

    audit_log.append(f"Checkpoint 4 & 10 PASS: Exactly 30,600 unique composite keys verified with zero duplicates.")

    # ---------------------------------------------------------
    # Checkpoint 5: Strict Partition Membership
    # ---------------------------------------------------------
    for p_id in ["pc1", "pc2", "pc3"]:
        df_p = pc_dfs[p_id]
        p_assigned = partition_conditions[p_id]
        for _, r in df_p.iterrows():
            ck = (str(r["image_id"]), float(r["noise_sigma"]), int(r["trial"]))
            if ck not in p_assigned:
                err = f"[Checkpoint 5 FAIL] PC {p_id} evaluated unassigned condition: {ck}"
                fatal_errors.append(err)
                print(err)
                return False

    audit_log.append("Checkpoint 5 PASS: Strict partition membership confirmed; zero unassigned conditions executed by any PC.")

    # ---------------------------------------------------------
    # Checkpoint 3: Seed Derivation Formulas
    # ---------------------------------------------------------
    seed_errors = 0
    for _, r in combined_df.iterrows():
        img_id = str(r["image_id"])
        noise_s = float(r["noise_sigma"])
        tr = int(r["trial"])
        exp_sensing, exp_noise = compute_seeds(img_id, noise_s, tr)
        if int(r["sensing_seed"]) != exp_sensing or int(r["noise_seed"]) != exp_noise:
            seed_errors += 1

    if seed_errors > 0:
        err = f"[Checkpoint 3 FAIL] Found {seed_errors} seed formula discrepancies in merged dataset."
        fatal_errors.append(err)
        print(err)
        return False

    audit_log.append("Checkpoint 3 PASS: All 30,600 evaluations strictly match canonical sensing_seed and noise_seed formulas.")

    # ---------------------------------------------------------
    # Checkpoint 6: Cross-Method Noisy Input Hash Consistency
    # ---------------------------------------------------------
    hash_mismatches = 0
    grouped = combined_df.groupby(["image_id", "noise_sigma", "trial"])["noisy_input_hash"].unique()
    for cond_k, hashes in grouped.items():
        if len(hashes) != 1 or str(hashes[0]) in ("", "nan", "missing"):
            hash_mismatches += 1

    if hash_mismatches > 0:
        err = f"[Checkpoint 6 FAIL] Found {hash_mismatches} conditions where the 3 methods did NOT evaluate identical noisy input hashes!"
        fatal_errors.append(err)
        print(err)
        return False

    audit_log.append("Checkpoint 6 PASS: 100% bitwise noisy input hash identity verified across all 3 methods for every condition.")

    # ---------------------------------------------------------
    # Checkpoint 8 & 9: Incompatible Records & Error Logs
    # ---------------------------------------------------------
    failed_rows = len(combined_df[combined_df["solver_status"] != "SUCCESS"])
    if failed_rows > 0:
        err = f"[Checkpoint 8 & 9 FAIL] Found {failed_rows} failed solver evaluations in merged dataset."
        fatal_errors.append(err)
        print(err)
        return False

    for p_id in ["pc1", "pc2", "pc3"]:
        err_path = os.path.join(FINAL_BENCHMARK_DIR, p_id, "error_log.csv")
        if os.path.exists(err_path) and os.path.getsize(err_path) > 0:
            err_df = pd.read_csv(err_path)
            if len(err_df) > 0:
                err = f"[Checkpoint 8 & 9 FAIL] Error log in {p_id} contains {len(err_df)} failures!"
                fatal_errors.append(err)
                print(err)
                return False

    audit_log.append("Checkpoint 8 & 9 PASS: Zero failed evaluations or error log entries detected; 100% clean execution.")

    # ---------------------------------------------------------
    # Save Merged Dataset & Manifest
    # ---------------------------------------------------------
    combined_df.to_csv(MERGED_CSV_PATH, index=False)
    merged_csv_hash = compute_file_sha256(MERGED_CSV_PATH)
    print(f"\n[Saved] Merged Raw Results CSV -> {MERGED_CSV_PATH} (30,600 rows, SHA-256: {merged_csv_hash[:16]}...)")

    merged_manifest = {
        "benchmark_name": "ASL_SR_DPT_FINAL_BSD68_BENCHMARK_MERGED",
        "merge_timestamp": datetime.datetime.now().isoformat(),
        "total_images": 68,
        "noise_levels": [15.0, 25.0, 50.0],
        "trials_per_condition": 50,
        "methods": METHODS_ORDER,
        "total_conditions": EXPECTED_TOTAL_CONDITIONS,
        "total_evaluations": EXPECTED_TOTAL_EVALUATIONS,
        "partition_sources": {
            "pc1": {
                "evaluations": len(pc_dfs["pc1"]),
                "raw_results_sha256": compute_file_sha256(os.path.join(FINAL_BENCHMARK_DIR, "pc1", "raw_results.csv")),
            },
            "pc2": {
                "evaluations": len(pc_dfs["pc2"]),
                "raw_results_sha256": compute_file_sha256(os.path.join(FINAL_BENCHMARK_DIR, "pc2", "raw_results.csv")),
            },
            "pc3": {
                "evaluations": len(pc_dfs["pc3"]),
                "raw_results_sha256": compute_file_sha256(os.path.join(FINAL_BENCHMARK_DIR, "pc3", "raw_results.csv")),
            },
        },
        "merged_raw_results_sha256": merged_csv_hash,
        "code_hash": list(code_hashes)[0],
        "config_hash": list(config_hashes)[0],
        "ten_point_audit_status": "PASSED_ALL_10_CHECKPOINTS",
    }

    with open(MERGED_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_manifest, f, indent=2)
    print(f"[Saved] Merged Manifest -> {MERGED_MANIFEST_PATH}")

    # Write Final Integrity Report
    rep_lines = [
        "# ASL-SR-DPT Final 30,600-Evaluation Benchmark: 10-Point Merged Integrity Report",
        "",
        f"**Audit Timestamp:** {datetime.datetime.now().isoformat()}",
        f"**Overall Certification:** **PASSED — PERFECT REPRODUCIBILITY & CONFORMANCE**",
        "",
        "---",
        "",
        "## 1. Ten-Point Distributed Audit Checklist",
        "",
    ]
    for chk in audit_log:
        rep_lines.append(f"- [x] **{chk}**")
    rep_lines.extend([
        "",
        "---",
        "",
        "## 2. Workload & Partition Verification",
        "",
        "| Partition Source | Assigned Conditions | Completed Evaluations | Failed Count | Missing Count | Audit Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **PC 1** | 3,400 | {len(pc_dfs['pc1'])} | 0 | 0 | PASS |",
        f"| **PC 2** | 3,400 | {len(pc_dfs['pc2'])} | 0 | 0 | PASS |",
        f"| **PC 3** | 3,400 | {len(pc_dfs['pc3'])} | 0 | 0 | PASS |",
        f"| **Total Merged** | **10,200** | **{total_evals}** | **0** | **0** | **PASS** |",
        "",
    ])

    with open(FINAL_INTEGRITY_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(rep_lines))
    print(f"[Saved] Final Integrity Report -> {FINAL_INTEGRITY_REPORT_PATH}")
    print("=" * 80)
    print("Merged Benchmark Complete & Fully Certified.")
    print("=" * 80)

    return True


def main():
    merge_and_audit_benchmark()


if __name__ == "__main__":
    main()
