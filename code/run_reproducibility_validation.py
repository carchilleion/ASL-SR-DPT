"""
Static reproducibility validation script for Requirement 20:
Runs 1 reproducibility test (3 solvers, test001, sigma=15.0, trial=1, twice, full image).
Outputs exact array comparisons, SHA-256 hashes, and classifications.
"""

import sys
import os
import json
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app import (
    load_pilot_config,
    compute_codebase_integrity_hash,
    run_reproducibility_test,
    CONFIG_PATH,
)

def main():
    print("=" * 80)
    print("STARTING FULL-IMAGE DETERMINISTIC REPRODUCIBILITY VALIDATION (REQUIREMENT 20)")
    print("Image: test001 | Noise: sigma=15.0 | Trial: 1 | Complete 37,604 Patches")
    print("Solvers: ASL-SR-DPT, OMP, LASSO-ADMM (Executed Twice Each)")
    print("=" * 80)

    config, cfg_hash = load_pilot_config(CONFIG_PATH)
    code_hash, _ = compute_codebase_integrity_hash()

    print(f"Config Hash:   {cfg_hash}")
    print(f"Codebase Hash: {code_hash}")
    print("Starting duplicate executions across all 3 solvers...")

    t0 = time.time()
    results = run_reproducibility_test(config, cfg_hash, code_hash)
    elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print(f"REPRODUCIBILITY RESULTS SUMMARY (Total Duration: {elapsed:.2f} s)")
    print("=" * 80)

    all_passed = True
    for r in results:
        solver = r["solver"]
        classification = r["classification"]
        passed = r["passed"]
        all_passed = all_passed and passed

        print(f"\nSolver: {solver}")
        print(f"  Classification:     {classification} (Passed: {passed})")
        print(f"  Bytes Match:        {r['bytes_match']}")
        print(f"  Image SHA-256:      {r['image_sha256']}")
        print(f"  Coeff SHA-256:      {r['coeff_sha256']}")
        print(f"  Max Array Diff:     {r['max_image_diff']:.2e}")
        print(f"  PSNR Run 1 / Run 2: {r['psnr_run1']:.4f} dB / {r['psnr_run2']:.4f} dB (diff: {r['psnr_diff']:.2e})")
        print(f"  SSIM Diff:          {r['ssim_diff']:.2e}")
        print(f"  MSE Diff:           {r['mse_diff']:.2e}")
        print(f"  Residual Diff:      {r['res_diff']:.2e}")

    print("\n" + "=" * 80)
    print(f"OVERALL REPRODUCIBILITY STATUS: {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 80)

    # Save to a json artifact
    out_file = os.path.join(REPO_ROOT, "results", "reproducibility_validation_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config_hash": cfg_hash,
            "code_hash": code_hash,
            "elapsed_seconds": elapsed,
            "overall_passed": all_passed,
            "results": results,
        }, f, indent=2)
    print(f"Saved validation report to: {out_file}")

if __name__ == "__main__":
    main()
