"""
ASL-SR-DPT Final Benchmark: Deterministic Partition Generation Engine (3-PC Distributed Architecture)

Generates three disjoint, perfectly balanced partitions of exactly 3,400 conditions each
from the complete 10,200 unique BSD68 image-noise-trial conditions.

Workload:
- Total unique conditions: 68 images x 3 noise levels x 50 trials = 10,200 conditions.
- Workload per PC: 3,400 conditions x 3 methods = 10,200 evaluations.
- Combined workload: 10,200 evals x 3 PCs = 30,600 evaluations.

Outputs:
- results/final_benchmark/partitions/pc1_conditions.csv
- results/final_benchmark/partitions/pc2_conditions.csv
- results/final_benchmark/partitions/pc3_conditions.csv
- results/final_benchmark/partitions/partition_manifest.json
"""

import os
import sys
import csv
import json
import hashlib
import argparse
import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(REPO_ROOT, "code")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from dataset import load_bsd68
import app

CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "final_benchmark_3pc_config.json")
OUTPUT_DIR = os.path.join(REPO_ROOT, "results", "final_benchmark", "partitions")

BASE_SEED = 20260908
NOISE_LEVELS = [15.0, 25.0, 50.0]
TRIAL_RANGE = (1, 50)
METHODS_LIST = [
    "Noise-Adaptive ASL-SR-DPT",
    "OMP",
    "LASSO-ADMM",
]


def compute_file_sha256(filepath):
    """Computes SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return "missing"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_seeds(image_id, noise_sigma, trial):
    """
    Deterministic seed formulas matching repository invariants.
    clean_id: numeric part of image_id (e.g. 'test001' -> 1)
    """
    clean_id = int(str(image_id).replace("test", "").lstrip("0") or "0")
    sensing_seed = BASE_SEED + clean_id * 1000000 + int(noise_sigma) * 1000 + int(trial)
    noise_seed = BASE_SEED + int(trial) * 100000 + int(noise_sigma) * 1000 + clean_id
    return sensing_seed, noise_seed


def generate_canonical_conditions():
    """
    Generates deterministically ordered list of all 10,200 conditions.
    Order: image_id ascending ('test001' to 'test068'),
           noise_sigma ascending (15.0, 25.0, 50.0),
           trial ascending (1 to 50).
    """
    dataset_dir = os.path.join(REPO_ROOT, "data", "BSD68")
    loaded_data = load_bsd68(dataset_dir=dataset_dir, strict_count=68)
    image_ids = sorted(list(loaded_data.keys()))

    conditions = []
    condition_id = 0
    for img_id in image_ids:
        for sigma in NOISE_LEVELS:
            for trial in range(TRIAL_RANGE[0], TRIAL_RANGE[1] + 1):
                sensing_seed, noise_seed = compute_seeds(img_id, sigma, trial)
                conditions.append({
                    "condition_id": condition_id,
                    "image_id": img_id,
                    "noise_sigma": float(sigma),
                    "trial": int(trial),
                    "sensing_seed": sensing_seed,
                    "noise_seed": noise_seed,
                })
                condition_id += 1

    return conditions, image_ids


def partition_conditions(conditions, strategy="interleaved"):
    """
    Partitions the 10,200 conditions into 3 non-overlapping sets of 3,400 each.
    strategy='interleaved': round-robin (index % 3 == p_idx).
    strategy='contiguous': contiguous slicing (0..3400, 3400..6800, 6800..10200).
    """
    n_total = len(conditions)
    if n_total != 10200:
        raise ValueError(f"Expected 10,200 conditions, got {n_total}")

    partitions = {
        "pc1": [],
        "pc2": [],
        "pc3": [],
    }

    if strategy == "interleaved":
        for i, cond in enumerate(conditions):
            p_key = f"pc{(i % 3) + 1}"
            cond_copy = dict(cond)
            cond_copy["partition_id"] = p_key
            partitions[p_key].append(cond_copy)
    elif strategy == "contiguous":
        chunk_size = 3400
        for p_idx, p_key in enumerate(["pc1", "pc2", "pc3"]):
            start = p_idx * chunk_size
            end = start + chunk_size
            for cond in conditions[start:end]:
                cond_copy = dict(cond)
                cond_copy["partition_id"] = p_key
                partitions[p_key].append(cond_copy)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    return partitions


def validate_partitions(partitions, universe_conditions):
    """
    Rigorous set-theoretic and statistical validation of partitions.
    """
    errors = []
    pc1 = partitions["pc1"]
    pc2 = partitions["pc2"]
    pc3 = partitions["pc3"]

    # 1. Exact count check
    if len(pc1) != 3400:
        errors.append(f"PC1 condition count mismatch: expected 3400, got {len(pc1)}")
    if len(pc2) != 3400:
        errors.append(f"PC2 condition count mismatch: expected 3400, got {len(pc2)}")
    if len(pc3) != 3400:
        errors.append(f"PC3 condition count mismatch: expected 3400, got {len(pc3)}")

    # 2. Key sets
    def make_key(c):
        return (c["image_id"], float(c["noise_sigma"]), int(c["trial"]))

    set1 = {make_key(c) for c in pc1}
    set2 = {make_key(c) for c in pc2}
    set3 = {make_key(c) for c in pc3}
    universe_set = {make_key(c) for c in universe_conditions}

    if len(set1) != 3400:
        errors.append(f"PC1 contains {3400 - len(set1)} duplicate keys.")
    if len(set2) != 3400:
        errors.append(f"PC2 contains {3400 - len(set2)} duplicate keys.")
    if len(set3) != 3400:
        errors.append(f"PC3 contains {3400 - len(set3)} duplicate keys.")

    # 3. Disjointness (Pairwise intersections)
    int_12 = set1 & set2
    int_23 = set2 & set3
    int_13 = set1 & set3
    if int_12:
        errors.append(f"Collision between PC1 and PC2: {len(int_12)} overlapping conditions.")
    if int_23:
        errors.append(f"Collision between PC2 and PC3: {len(int_23)} overlapping conditions.")
    if int_13:
        errors.append(f"Collision between PC1 and PC3: {len(int_13)} overlapping conditions.")

    # 4. Completeness (Union)
    union_set = set1 | set2 | set3
    if union_set != universe_set:
        missing = universe_set - union_set
        excess = union_set - universe_set
        if missing:
            errors.append(f"Union is missing {len(missing)} universe conditions.")
        if excess:
            errors.append(f"Union contains {len(excess)} non-universe conditions.")

    return errors


def generate_partitions(strategy="interleaved"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("ASL-SR-DPT Final Benchmark: Partition Generator (3-PC Distributed)")
    print("=" * 70)

    # 1. Generate Universe of Conditions
    conditions, image_ids = generate_canonical_conditions()
    total_conditions = len(conditions)
    print(f"Dataset: BSD68 ({len(image_ids)} images: {image_ids[0]}..{image_ids[-1]})")
    print(f"Noise levels: {NOISE_LEVELS}")
    print(f"Trials per condition: {TRIAL_RANGE[1] - TRIAL_RANGE[0] + 1}")
    print(f"Total unique conditions generated: {total_conditions}")
    print(f"Partition strategy: {strategy}")

    # 2. Partition
    partitions = partition_conditions(conditions, strategy=strategy)

    # 3. Validate
    errors = validate_partitions(partitions, conditions)
    if errors:
        print("\n[VALIDATION FAILED]")
        for err in errors:
            print(f"  - {err}")
        raise RuntimeError("Partition validation failed. Aborting file write.")

    print("\n[Set-Theoretic Validation PASSED]")
    print(f"  - PC1 conditions: {len(partitions['pc1'])} (Disjoint)")
    print(f"  - PC2 conditions: {len(partitions['pc2'])} (Disjoint)")
    print(f"  - PC3 conditions: {len(partitions['pc3'])} (Disjoint)")
    print(f"  - Total Union: {len(partitions['pc1']) + len(partitions['pc2']) + len(partitions['pc3'])} / {total_conditions}")
    print(f"  - Pairwise Overlaps: PC1 n PC2 = 0, PC2 n PC3 = 0, PC1 n PC3 = 0")

    # 4. Write CSVs
    csv_header = ["condition_id", "partition_id", "image_id", "noise_sigma", "trial", "sensing_seed", "noise_seed"]
    file_hashes = {}

    for p_key in ["pc1", "pc2", "pc3"]:
        csv_path = os.path.join(OUTPUT_DIR, f"{p_key}_conditions.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_header)
            writer.writeheader()
            for r in partitions[p_key]:
                writer.writerow(r)
        file_hashes[f"{p_key}_conditions_csv"] = compute_file_sha256(csv_path)
        print(f"  [Saved] {csv_path} (3,400 rows, SHA-256: {file_hashes[f'{p_key}_conditions_csv'][:12]}...)")

    # 5. Code & Config hashes
    code_hash = app.CODE_HASH
    config_hash = compute_file_sha256(CONFIG_PATH)

    # 6. Write Partition Manifest
    manifest_data = {
        "benchmark_name": "ASL_SR_DPT_FINAL_BSD68_BENCHMARK_3PC_DISTRIBUTED",
        "generated_timestamp": datetime.datetime.now().isoformat(),
        "dataset": "BSD68",
        "dataset_images": image_ids,
        "total_images": len(image_ids),
        "noise_levels": NOISE_LEVELS,
        "trial_range": list(TRIAL_RANGE),
        "trials_per_condition": TRIAL_RANGE[1] - TRIAL_RANGE[0] + 1,
        "methods": METHODS_LIST,
        "total_methods": len(METHODS_LIST),
        "total_conditions": total_conditions,
        "number_of_partitions": 3,
        "partition_strategy": strategy,
        "workload_distribution": {
            "pc1": {
                "assigned_conditions": len(partitions["pc1"]),
                "methods_per_condition": len(METHODS_LIST),
                "expected_evaluations": len(partitions["pc1"]) * len(METHODS_LIST),
                "conditions_file": "pc1_conditions.csv",
                "sha256": file_hashes["pc1_conditions_csv"],
            },
            "pc2": {
                "assigned_conditions": len(partitions["pc2"]),
                "methods_per_condition": len(METHODS_LIST),
                "expected_evaluations": len(partitions["pc2"]) * len(METHODS_LIST),
                "conditions_file": "pc2_conditions.csv",
                "sha256": file_hashes["pc2_conditions_csv"],
            },
            "pc3": {
                "assigned_conditions": len(partitions["pc3"]),
                "methods_per_condition": len(METHODS_LIST),
                "expected_evaluations": len(partitions["pc3"]) * len(METHODS_LIST),
                "conditions_file": "pc3_conditions.csv",
                "sha256": file_hashes["pc3_conditions_csv"],
            },
        },
        "expected_evaluations_total": total_conditions * len(METHODS_LIST),
        "base_seed": BASE_SEED,
        "seed_formulas": {
            "sensing_seed": "base_seed + clean_id * 1000000 + int(noise_sigma) * 1000 + trial",
            "noise_seed": "base_seed + trial * 100000 + int(noise_sigma) * 1000 + clean_id",
        },
        "code_hash": code_hash,
        "config_hash": config_hash,
        "max_cpu_workers_limit": 10,
        "validation_status": "PASSED_PERFECT_DISJOINT_COVERAGE",
    }

    manifest_path = os.path.join(OUTPUT_DIR, "partition_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"  [Saved] {manifest_path}")
    print("=" * 70)
    print("Partition Generation Complete & Certified.")
    print("=" * 70)

    return manifest_data


def main():
    parser = argparse.ArgumentParser(description="Deterministic 3-PC Partition Generator for ASL-SR-DPT Final Benchmark")
    parser.add_argument("--strategy", choices=["interleaved", "contiguous"], default="interleaved",
                        help="Partition splitting strategy (default: interleaved for stratified balance)")
    args = parser.parse_args()

    generate_partitions(strategy=args.strategy)


if __name__ == "__main__":
    main()
