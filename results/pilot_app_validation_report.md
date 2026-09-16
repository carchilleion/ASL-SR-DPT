# ASL-SR-DPT Pilot Benchmark Application Pre-Flight Validation Report

**Project:** Accelerated Single-Loop Sparse Recovery with Dynamic Parameter Tuning for Mobile Image Denoising  
**Application File:** [`app.py`](file:///c:/Users/Carlo%20Mendoza/OneDrive/Desktop/ASL-SR-DPT/app.py)  
**Configuration File:** [`configs/pilot_config.json`](file:///c:/Users/Carlo%20Mendoza/OneDrive/Desktop/ASL-SR-DPT/configs/pilot_config.json)  
**Date Generated:** 2026-09-16  
**Final Status:** **READY FOR PILOT**  

---

## 1. Executive Summary & Verdict

The ASL-SR-DPT pilot benchmark application has undergone a comprehensive academic integrity and engineering audit. All requirements specified for the distributed 450-evaluation pilot study have been fully satisfied, verified, and certified:

1. **Codebase Integrity Hash:** `app.py` and `requirements.txt` have been integrated into `RELEVANT_CODE_FILES`. The codebase hash now covers the application orchestration itself, ensuring that any modification to the execution logic changes `CODE_HASH`.
2. **Explicit Sensing-Seed Policy:** A unique, deterministic sensing seed is computed per experimental condition:
   $$\text{sensing\_seed} = \text{base\_seed} + \text{clean\_id} \times 1{,}000{,}000 + \text{int}(\text{noise\_sigma}) \times 1{,}000 + \text{int}(\text{trial})$$
3. **Strict ASL-SR-DPT Diagnostic Schema Validation:** Implemented per-batch and per-patch schema validation (`validate_dpt_batch_diagnostics`). Array lengths, positive iteration counts, and bounded active support sizes ($0 \le |S_k| \le 63$) are strictly validated before acceptance.
4. **Real Diagnostic Extraction:** All fabricated constants (e.g. 97.8, 0.0) have been permanently removed. Active support ratio ($R_{\text{active}} = \frac{\sum_k |S_k|}{63 K_{\text{total}}}$), mean active support count, accepted steps, failed line searches, and final sigma are computed from live patch diagnostics. Baselines (OMP, LASSO-ADMM) output strictly blank fields.
5. **Full-Image Reproducibility Certification (Gate 3):** All three solvers (`ASL-SR-DPT`, `OMP`, `LASSO-ADMM`) were executed twice on `test001` ($\sigma=15.0$, Trial 1) across all 37,604 patches. All three solvers achieved **`EXACT_REPRODUCTION`** (byte-identical array SHA-256 hashes, zero numerical divergence).
6. **Workload Disjointness:** Team A (Khevin: `test001`–`test005`) and Team B (Marc: `test006`–`test010`) are strictly non-overlapping ($225 + 225 = 450$ evaluations).
7. **Pre-Write Validation & Atomic Flushing:** Every completed evaluation row is validated against research invariants and flushed atomically to disk immediately upon completion.
8. **Automated Benchmark Restraint:** Benchmark execution remains halted and will NOT launch automatically.

---

## 2. Environment Certification

The execution environment is locked and certified for reproducible single-threaded timing:

| Parameter | Certified Value | Thesis / Application Requirement | Status |
| :--- | :--- | :--- | :--- |
| **Operating System** | Windows 11 Pro / Ubuntu 22.04 LTS (VMware) | Multi-platform compatible | Verified |
| **Python Version** | 3.12.9 | $\ge 3.10$ | **PASS** |
| **NumPy Version** | 2.2.3 | Validated linear algebra backend | **PASS** |
| **SciPy Version** | 1.15.3 | Explicitly reported in environment metadata | **PASS** |
| **Pandas Version** | 2.2.3 | Validated CSV export & merge | **PASS** |
| **Streamlit Version** | 1.42.0 | Reactive Web UI & audit controls | **PASS** |
| **Thread Pinning** | `OMP=1, OPENBLAS=1, MKL=1, NUMEXPR=1` | Single-threaded latency isolation | **PASS** |

---

## 3. Codebase Integrity & Configuration Hashes

### Configuration Digest
- **Config Path:** `configs/pilot_config.json`
- **Config SHA-256:** `5b7afb33830f7ce025bed2b5639f6b6eef01d7b2844fba1f3fe12cf8b962a27b`

### Codebase Digest
- **Aggregate Codebase SHA-256:** `875540736616f08f9bcc5f8db7ce7f2094e3912a5651f904d8b88d8dea536752`

### Individual Component Hashes
| Component / File Path | SHA-256 Digest | Status |
| :--- | :--- | :--- |
| `configs/pilot_config.json` | `5b7afb33830f7ce025bed2b5639f6b6eef01d7b2844fba1f3fe12cf8b962a27b` | Frozen |
| `code/hybrid_sparse_solver_v7_optimized.py` | `9dfe3bd66337e998b4da5085ed8bf5b675a36f340ba224fd59576e06a309de96` | Frozen |
| `code/hybrid_sparse_solver_v7_fixed.py` | `c6ccd84eeb8f00f389bc7c6234e354df4fac735739a5cabfcc00530357d7eeb1` | Frozen |
| `code/sensing.py` | `b145f8a644d385cc23e28450ea743473a2af81e3b465c999c753c7f4ac948a54` | Frozen |
| `code/reconstruction.py` | `87f706f6486c6bf0f91ab3dc9fd7c744f977ca7baab72ddb0aa09e4a2cbc5a12` | Frozen |
| `code/metrics.py` | `eaf8806c3c6ba80122f74b08ca51dc270195e867ced4dd96f40650320baa7324` | Frozen |
| `app.py` | `2a7f7a764d23a1d3d8467d15f441eb5df0ec65a38e6415f38332bf04ba112ad6` | Verified |
| `requirements.txt` | `64c2c79aad5434ae7465c7d2b69e6a4a141d5ca750a9cd71c0b26775ca0226ab` | Verified |

*Note: Any change to `app.py` or `requirements.txt` immediately changes `CODE_HASH`, preventing unverified modifications during execution.*

---

## 4. Workload Integrity & Team Allocation

| Metric | Team A (Khevin) | Team B (Marc) | Combined Pilot Total | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Assigned BSD68 Images** | `test001`, `test002`, `test003`, `test004`, `test005` (5) | `test006`, `test007`, `test008`, `test009`, `test010` (5) | 10 unique BSD68 images | **PASS** |
| **Noise Levels ($\sigma$)** | 15.0, 25.0, 50.0 (3) | 15.0, 25.0, 50.0 (3) | 15.0, 25.0, 50.0 (3) | **PASS** |
| **Randomized Trials** | 1, 2, 3, 4, 5 (5) | 1, 2, 3, 4, 5 (5) | 1, 2, 3, 4, 5 (5) | **PASS** |
| **Evaluated Solvers** | ASL-SR-DPT, OMP, LASSO-ADMM (3) | ASL-SR-DPT, OMP, LASSO-ADMM (3) | 3 matched solvers | **PASS** |
| **Evaluations per Team** | $5 \times 3 \times 5 \times 3 = \mathbf{225}$ | $5 \times 3 \times 5 \times 3 = \mathbf{225}$ | $\mathbf{450}$ total evaluations | **PASS** |
| **Key Overlap** | 0 | 0 | Exactly $\emptyset$ overlap | **PASS** |

---

## 5. Explicit Deterministic Seed Protocol

Deterministic seeding guarantees matched input conditions across solvers and exact reproducibility across machines:

1. **Sensing Seed Equation:**
   $$\text{sensing\_seed} = \text{base\_seed} + \text{clean\_id} \times 1{,}000{,}000 + \text{int}(\text{noise\_sigma}) \times 1{,}000 + \text{int}(\text{trial})$$
2. **Noise Seed Equation:**
   $$\text{noise\_seed} = \text{base\_seed} + \text{int}(\text{trial}) \times 100{,}000 + \text{int}(\text{noise\_sigma}) \times 1{,}000 + \text{clean\_id}$$
3. **Base Seed:** `20260908`

### Seed Samples
- `test001`, $\sigma=15$, Trial 1: `sensing_seed = 21275909`, `noise_seed = 20375909`
- `test005`, $\sigma=50$, Trial 5: `sensing_seed = 25310913`, `noise_seed = 20810913`
- `test010`, $\sigma=25$, Trial 3: `sensing_seed = 30285911`, `noise_seed = 20585918`

---

## 6. Diagnostic Schema & Invariant Enforcement

1. **ASL-SR-DPT Active Support Ratio:**
   $$R_{\text{active}} = \frac{\sum_{k} |S_k|}{63 \times K_{\text{total}}}$$
   where $K_{\text{total}} = \sum_p \text{iterations}_p$ is the total number of iterations executed across all 37,604 patches.
2. **Batch Diagnostic Schema Validation (`validate_dpt_batch_diagnostics`):**
   - Required diagnostic arrays: `patch_iterations`, `patch_active_counts`, `patch_accepted_steps`, `patch_failed_line_searches`, `patch_final_sigma`.
   - Length of each array strictly equals batch size ($B$).
   - For each patch: $\text{iterations} > 0$, $\text{len}(\text{patch\_active\_counts}) == \text{iterations}$.
   - For each iteration $k$: $0 \le |S_k| \le 63$.
   - $\text{accepted\_steps} \ge 0$, $\text{failed\_line\_searches} \ge 0$, $\text{final\_sigma} > 0$.
3. **Baselines Cleanliness:**
   - OMP and LASSO-ADMM output empty string `""` for $R_{\text{active}}$, mean active count, final sigma, accepted steps, and failed line searches.

---

## 7. Full-Image Reproducibility Gate Results (Gate 3)

Conducted on `test001` ($\sigma=15.0$, Trial 1) across all **37,604 patches** per solve, executed twice per solver.

- **Total Verification Duration:** 545.01 seconds (~9.1 minutes)
- **Patches Evaluated:** $37{,}604 \times 6 = 225{,}624$ patch solves

| Solver | Run 1 PSNR | Run 2 PSNR | $\Delta$ PSNR | Max Pixel Diff | Image SHA-256 Digest | Classification | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ASL-SR-DPT** | 21.2567 dB | 21.2567 dB | 0.00 dB | 0.00e+00 | `0c3c7cac33c1f0308d234a74ae0813e012485f1f885f25f83550da9d9d9cec69` | `EXACT_REPRODUCTION` | **PASS** |
| **OMP** | 24.4254 dB | 24.4254 dB | 0.00 dB | 0.00e+00 | `ed0225f49d59e225aa8457b3486b2f7fa81283e8708530a6c0c939abd721312e` | `EXACT_REPRODUCTION` | **PASS** |
| **LASSO-ADMM** | 21.5160 dB | 21.5160 dB | 0.00 dB | 0.00e+00 | `a56c66e6e81f38587f3a72d805ba08a2130a26f96001bda6775ec1ed80fd4abc` | `EXACT_REPRODUCTION` | **PASS** |

**Overall Gate 3 Status:** **PASSED (100% Exact Reproduction)**

---

## 8. Test Suite Verification

The full test suite was executed via `pytest`:
- `pytest tests/test_pilot_application.py`: **11 / 11 PASSED** (1.34s)
- `pytest tests/`: **34 / 34 PASSED** (1.83s)

Coverage includes config validation, seed determinism, team disjointness, CSV schema invariants, batch diagnostic schema rejection, solver numerical equivalence, resume behavior, and code integrity hashing.

---

## 9. Teammate Execution Guide

### Team A: Khevin
- **Assigned Workload:** `test001` through `test005` (225 evaluations)
- **Windows Command:**
  ```cmd
  run_team_a.bat
  ```
- **Linux / VMware Command:**
  ```bash
  chmod +x run_team_a.sh
  ./run_team_a.sh
  ```

### Team B: Marc
- **Assigned Workload:** `test006` through `test010` (225 evaluations)
- **Windows Command:**
  ```cmd
  run_team_b.bat
  ```
- **Linux / VMware Command:**
  ```bash
  chmod +x run_team_b.sh
  ./run_team_b.sh
  ```

### Central Merging & Audit (Post-Execution)
When both teammates have finished their 225 evaluations:
```bash
python merge_pilot_results.py
```
This utility verifies all 450 evaluation keys, performs cross-team uniqueness and seed audits, and generates `results/combined/final_pilot_audit.md`.

---

## 10. Final Readiness Declaration

```
================================================================================
FINAL BENCHMARK READINESS DECLARATION: READY FOR PILOT
================================================================================
The ASL-SR-DPT Pilot Benchmark Application is certified structurally sound,
mathematically valid, deterministic, fault-tolerant, and ready for deployment.
Benchmark workloads remain frozen and will NOT execute until initiated by the user.
================================================================================
```
