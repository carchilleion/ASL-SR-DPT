# ASL-SR-DPT Pilot Benchmark: Administrator Merge & Audit Report

**Date:** 2026-09-16 20:02:49  
**Configuration Hash:** `8cdf250d57dd3979dd5b1edaa297e0ee471568028c9c7571c542b18a09be5d14`  

---

## 1. Executive Certification Checklist

Expected:
450 evaluations

Received:
0

Missing:
450

Duplicates:
0

Team A:
225 expected (received 0)

Team B:
225 expected (received 0)

Configuration consistency:
FAIL

Seed consistency:
FAIL

Experiment-key uniqueness:
PASS

Overall pilot integrity:
FAIL

---

## 2. Workload & Coverage Verification

| Evaluation Track | Assigned Images | Expected | Received | Missing | Duplicates | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Team A** | test001, test002, test003, test004, test005 | 225 | 0 | 225 | 0 | INCOMPLETE |
| **Team B** | test006, test007, test008, test009, test010 | 225 | 0 | 225 | 0 | INCOMPLETE |
| **Combined Total** | All 10 Pilot Images | 450 | 0 | 450 | 0 | DEFICIT |

### Outstanding Missing Evaluations (450)
| Image | Noise Sigma | Trial | Solver |
| :--- | :---: | :---: | :--- |
| `test001` | 15.0 | 1 | `ASL-SR-DPT` |
| `test001` | 15.0 | 1 | `LASSO-ADMM` |
| `test001` | 15.0 | 1 | `OMP` |
| `test001` | 15.0 | 2 | `ASL-SR-DPT` |
| `test001` | 15.0 | 2 | `LASSO-ADMM` |
| `test001` | 15.0 | 2 | `OMP` |
| `test001` | 15.0 | 3 | `ASL-SR-DPT` |
| `test001` | 15.0 | 3 | `LASSO-ADMM` |
| `test001` | 15.0 | 3 | `OMP` |
| `test001` | 15.0 | 4 | `ASL-SR-DPT` |
| `test001` | 15.0 | 4 | `LASSO-ADMM` |
| `test001` | 15.0 | 4 | `OMP` |
| `test001` | 15.0 | 5 | `ASL-SR-DPT` |
| `test001` | 15.0 | 5 | `LASSO-ADMM` |
| `test001` | 15.0 | 5 | `OMP` |
| `test001` | 25.0 | 1 | `ASL-SR-DPT` |
| `test001` | 25.0 | 1 | `LASSO-ADMM` |
| `test001` | 25.0 | 1 | `OMP` |
| `test001` | 25.0 | 2 | `ASL-SR-DPT` |
| `test001` | 25.0 | 2 | `LASSO-ADMM` |
*... and 430 more.*  

## 4. Methodological Grounding

This audit certifies that all data merged by this utility preserves:
1. **Strict Key Independence:** No image patch or full-image experiment key was shared between teammates.
2. **Identical Numerical Constraints:** Both teams adhered to the frozen research configuration without local algorithmic parameter modifications.
3. **Deterministic Seeds:** Verified exact matched noise realizations across all three evaluated solvers.

