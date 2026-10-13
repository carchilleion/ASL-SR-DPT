# ASL-SR-DPT

**Accelerated Single-Loop Sparse Recovery with Dynamic Parameter Tuning for Mobile Image Denoising**

Official repository for the ASL-SR-DPT image denoising research project, featuring:
1. **Final 30,600-Evaluation Distributed Benchmark:** 3-PC distributed execution across the complete BSD68 dataset (68 images $\times$ 3 noise levels $\times$ 50 trials $\times$ 3 methods).
2. **Pilot 450-Evaluation Benchmark:** Preliminary reproducibility and parameter validation suite.

---

## Table of Contents

- [1. Final Benchmark Architecture (3-PC Distributed Execution)](#1-final-benchmark-architecture-3-pc-distributed-execution)
- [2. Workload Division & PC Assignments](#2-workload-division--pc-assignments)
- [3. System Requirements & Setup Guide](#3-system-requirements--setup-guide)
- [4. How to Run the Distributed Benchmark](#4-how-to-run-the-distributed-benchmark)
  - [Option A: Streamlit Control Center (GUI)](#option-a-streamlit-control-center-gui-recommended)
  - [Option B: Headless Terminal CLI](#option-b-headless-terminal-cli)
- [5. Engine Safeguards & Runtime Controls](#5-engine-safeguards--runtime-controls)
- [6. Teammate Git Collaboration Workflow](#6-teammate-git-collaboration-workflow)
- [7. Post-Benchmark: Validation, Merging & Analysis](#7-post-benchmark-validation-merging--analysis)
- [8. Pilot Benchmark Reference (450 Evaluations)](#8-pilot-benchmark-reference-450-evaluations)
- [9. Repository Layout](#9-repository-layout)

---

## 1. Final Benchmark Architecture (3-PC Distributed Execution)

The final benchmark validates the thesis hypothesis at scale across the full BSD68 benchmark:

- **Dataset:** BSD68 (All 68 standard test images: `test001.png` through `test068.png`)
- **Noise Levels ($\sigma$):** $15, 25, 50$ (Additive White Gaussian Noise)
- **Trials:** 1 through 50 (50 randomized Monte Carlo trials per condition)
- **Total Conditions:** $68 \text{ images} \times 3 \text{ noise levels} \times 50 \text{ trials} = \mathbf{10,200 \text{ unique conditions}}$
- **Evaluated Solvers (3 Methods per condition):**
  1. **Noise-Adaptive ASL-SR-DPT:** Adaptive regularization rule $\lambda(\sigma) = 0.01 \times (\sigma / 15)$ with DC preservation ($M_{\text{ac}}=37, N_{\text{ac}}=63$, $M_{\text{total}}=38, N_{\text{total}}=64$).
  2. **Orthogonal Matching Pursuit (OMP):** Standard full-patch greedy baseline ($M=38, N=64$).
  3. **LASSO-ADMM:** Standard full-patch $\ell_1$-regularized ADMM baseline ($M=38, N=64$).
- **Total Evaluations:** $10,200 \text{ conditions} \times 3 \text{ methods} = \mathbf{30,600 \text{ full-image solver evaluations}}$.

---

## 2. Workload Division & PC Assignments

To ensure exact fairness and equal hardware load, the 10,200 conditions are deterministically partitioned into three strictly non-overlapping workloads:

| Assigned PC | Operator | Assigned Conditions | Methods / Condition | Total Evaluations | Assigned Partition File | Output Directory |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **PC 1** | Carlo (Lead) | **3,400** | 3 | **10,200** | `results/final_benchmark/partitions/pc1_conditions.csv` | `results/final_benchmark/pc1/` |
| **PC 2** | Teammate 1 | **3,400** | 3 | **10,200** | `results/final_benchmark/partitions/pc2_conditions.csv` | `results/final_benchmark/pc2/` |
| **PC 3** | Teammate 2 | **3,400** | 3 | **10,200** | `results/final_benchmark/partitions/pc3_conditions.csv` | `results/final_benchmark/pc3/` |
| **TOTAL** | *Merged Dataset* | **10,200** | 3 | **30,600** | `results/final_benchmark/partitions/partition_manifest.json` | `results/final_benchmark/raw_results_merged.csv` |

### Deterministic Integrity Guarantees:
- **Disjoint & Exhaustive:** $\text{PC}_1 \cap \text{PC}_2 = \emptyset$, $\text{PC}_2 \cap \text{PC}_3 = \emptyset$, $\text{PC}_1 \cap \text{PC}_3 = \emptyset$, and $\bigcup_{i=1}^3 \text{PC}_i = 10,200$.
- **Bitwise Noise Parity:** For any $(image, \sigma, trial)$, all 3 solvers operate on the bitwise identical noisy image.
- **Deterministic Seeding:**
  - $\text{sensing\_seed} = 20260908 + \text{clean\_id} \times 1,000,000 + \sigma \times 1,000 + trial$
  - $\text{noise\_seed} = 20260908 + trial \times 100,000 + \sigma \times 1,000 + \text{clean\_id}$

---

## 3. System Requirements & Setup Guide

Each PC operates as an independent execution node.

### System Requirements:
- **OS:** Windows 10/11, Ubuntu Linux (20.04/22.04/24.04), or macOS
- **CPU:** Multi-core x86_64 CPU (up to 10 logical CPU workers utilized)
- **RAM:** Minimum 8 GB (16 GB recommended)
- **Disk Space:** ~5 GB free disk space (to store up to 10,200 reconstructed PNG images and logs)
- **Python:** Python 3.10, 3.11, or 3.12 (64-bit)

### Installation Steps (Run on Each PC):

```bash
# 1. Clone the repository
git clone https://github.com/carchilleion/ASL-SR-DPT.git
cd ASL-SR-DPT

# 2. Create and activate a virtual environment
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
python -m venv venv
.\venv\Scripts\activate.bat

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. How to Run the Distributed Benchmark

### Option A: Streamlit Control Center (GUI) *(Recommended)*

The interactive dashboard provides real-time progress bars, estimated time of arrival (ETA), live image preview, speed metrics, and responsive Pause/Resume controls.

#### To Launch:
- **Windows:** Double-click `launch_distributed_app.bat` or run:
  ```powershell
  streamlit run distributed_benchmark_app.py --server.port 8504
  ```
- **Linux / macOS:** Execute `run_distributed_app.sh` or run:
  ```bash
  chmod +x run_distributed_app.sh
  ./run_distributed_app.sh
  ```

#### In the Web Browser (`http://localhost:8504`):
1. **Select Assigned Partition:** Choose **`PC 1 (Conditions 1-3400)`**, **`PC 2 (Conditions 3401-6800)`**, or **`PC 3 (Conditions 6801-10200)`** matching your designated machine.
2. **Configure Workers:** The slider allows up to 10 CPU workers (hard-clamped at 10 by the execution engine).
3. **Image Saving Policy:** Defaults to **`Save All 10,200 Images (Warning: ~2GB per PC)`** into `results/final_benchmark/pcX/reconstructions/`.
4. **Click `▶️ Start Benchmark Workload`**:
   - The engine spawns a persistent background worker pool.
   - Real-time updates display progress ($X / 10,200$), active evaluation rate (evals/min), and live noisy/reconstruction image previews.
5. **Pause & Resume:** Click **`⏸️ Pause Benchmark`** at any point. The engine gracefully drains in-flight evaluations and flushes all CSV records. Click **`▶️ Start Benchmark Workload`** to seamlessly resume.

---

### Option B: Headless Terminal CLI

For headless servers, SSH sessions, or background batch execution without a browser:

```bash
# Run for PC 1:
python code/run_pc_benchmark.py --partition-id pc1 --max-cpu-workers 10 --save-images all

# Run for PC 2:
python code/run_pc_benchmark.py --partition-id pc2 --max-cpu-workers 10 --save-images all

# Run for PC 3:
python code/run_pc_benchmark.py --partition-id pc3 --max-cpu-workers 10 --save-images all
```

#### Available CLI Arguments:
- `--partition-id`: `pc1`, `pc2`, or `pc3` (Required).
- `--max-cpu-workers`: Number of parallel worker processes (Default: `10`, hard-capped at 10).
- `--save-images`: `all` (default, saves all 10,200 PNG images), `trial1`, `previews`, or `none`.
- `--max-retries`: Maximum retry attempts per failed evaluation before permanent failure logging (Default: `2`).
- `--max-consecutive-failures`: Failure circuit breaker threshold before automatic abort (Default: `5`).
- `--dry-run`: Pre-flight check that validates the partition queue and hash configuration without running solvers.

---

## 5. Engine Safeguards & Runtime Controls

The execution engine (`code/run_pc_benchmark.py`) incorporates six core production safeguards:

1. **Hard 10-Worker Limit:** Worker allocation is strictly clamped:
   $$\text{actual\_workers} = \max(1, \min(\text{requested}, 10, \text{available\_cpus}))$$
2. **Single-Thread Linear Algebra:** Worker processes enforce `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` to prevent CPU core oversubscription and ensure strictly comparable runtime measurements.
3. **Persistent Process Pool & Sliding Window:** Reuses a persistent `ProcessPoolExecutor` with a sliding target of $2 \times \text{actual\_workers}$ in-flight tasks, eliminating process spawn overhead.
4. **Crash Recovery & Hash Audit on Resume:** Before resuming, the engine audits existing records against current code SHA-256 and config SHA-256 hashes. Incompatible or corrupted rows are quarantined into `results/final_benchmark/pcX/raw_results_quarantine_<timestamp>.csv` rather than silently accepted.
5. **Lossless Pause Drain:** Graceful pause requests finish all in-flight patch evaluations before stopping, ensuring no partial computations are discarded.
6. **Failure Circuit Breaker:** Failed evaluations are retried up to 2 times. If 5 consecutive evaluations fail, execution halts to protect against corrupted environments.

---

## 6. Teammate Git Collaboration Workflow

Since the 3 PCs are independent and offline relative to each other, synchronization occurs through GitHub. Because each PC writes exclusively to its own directory (`pc1/`, `pc2/`, or `pc3/`), **there are zero git merge conflicts**.

### For Teammates (PC 2 and PC 3):

```bash
# 1. Pull the latest code and certified partitions before running
git pull origin main

# 2. Run your designated benchmark (via GUI or CLI) until all 10,200 evaluations complete
python code/run_pc_benchmark.py --partition-id pc2 --max-cpu-workers 10 --save-images all
# (Or use pc3 on PC 3)

# 3. Validate your output integrity
python code/validate_pc_benchmark.py --pc pc2

# 4. Stage, commit, and push your partition results
git add results/final_benchmark/pc2/
git commit -m "feat(benchmark): complete PC2 partition results (10,200 evals)"
git push origin main
```

### For Carlo / Lead (PC 1):

```bash
# 1. Run PC 1 benchmark
python code/run_pc_benchmark.py --partition-id pc1 --max-cpu-workers 10 --save-images all
python code/validate_pc_benchmark.py --pc pc1

# 2. Pull teammates' completed partitions from GitHub
git pull origin main

# 3. Validate all three partitions simultaneously
python code/validate_pc_benchmark.py --pc all
```

---

## 7. Post-Benchmark: Validation, Merging & Analysis

Once PC 1, PC 2, and PC 3 partitions are completed and pulled into the lead machine:

### Step 7.1: Execute 10-Point Cross-PC Audit & Merge
```bash
python code/merge_distributed_benchmark.py
```
This engine enforces:
1. Exact partition manifest hash verification.
2. Code and configuration hash consistency across all 3 outputs.
3. Seed derivation formula validation for all 30,600 rows.
4. Global uniqueness of all 30,600 composite evaluation keys.
5. Strict partition boundaries (no PC evaluated out-of-partition tasks).
6. Bitwise noisy-input SHA-256 parity across all 3 methods per condition.
7. Exact counts: 10,200 per PC, 10,200 per method, 30,600 total.
8. Rejection of corrupt or incomplete records.
9. Failure audit preservation.
10. Refusal of silent duplicate discarding.

**Outputs Produced:**
- `results/final_benchmark/raw_results_merged.csv` (Exactly 30,600 rows)
- `results/final_benchmark/benchmark_manifest_merged.json`
- `results/final_benchmark/final_integrity_report.md`

### Step 7.2: Generate Master Statistical Analysis & Publication Figures
```bash
python code/analyze_final_benchmark_30600.py
```
**Statistical Methodology:**
- Aggregates 50 trials per condition into 68 independent image-level observations.
- Computes Mean, Std, Median, IQR, Min, Max for PSNR, SSIM, and Runtime.
- Performs two-sided Wilcoxon signed-rank tests at $\alpha = 0.05$ with Holm-Bonferroni correction.
- Calculates matched-pairs rank-biserial correlation ($r$) effect sizes with 95% bootstrap confidence intervals.
- Generates publication-ready Pareto plots, PSNR/SSIM curves, and runtime boxplots in `results/final_benchmark/`.

---

## 8. Pilot Benchmark Reference (450 Evaluations)

For rapid parameter exploration or pre-benchmark testing, the 450-evaluation pilot benchmark remains available:

- **Images:** `test001` through `test010` (10 images)
- **Noise Levels:** $\sigma = 15, 25, 50$
- **Trials:** 1 through 5 (5 trials)
- **Workload:** $10 \times 3 \times 5 \times 3 = 450 \text{ evaluations}$

### To Run the Pilot:
```bash
# Via GUI:
streamlit run app.py --server.port 8501

# Or via dedicated pilot script:
python code/run_new_variant_pilot_450.py
```

---

## 9. Repository Layout

```
ASL-SR-DPT/
├── distributed_benchmark_app.py         # Streamlit GUI control center for PC1, PC2, PC3
├── launch_distributed_app.bat          # 1-click Windows launcher for distributed GUI
├── run_distributed_app.sh              # 1-click Linux/macOS launcher for distributed GUI
├── app.py                              # Legacy pilot benchmark Streamlit application
├── requirements.txt                    # Pinned Python package dependencies
├── configs/
│   ├── final_benchmark_3pc_config.json # Final 3-PC benchmark master configuration
│   └── pilot_config.json               # Pilot benchmark configuration
├── code/
│   ├── run_pc_benchmark.py             # Core distributed benchmark runner (10 workers, pause/resume)
│   ├── validate_pc_benchmark.py        # Single-PC partition validation engine
│   ├── merge_distributed_benchmark.py  # 3-PC cross-audit & 30,600-row merge engine
│   ├── analyze_final_benchmark_30600.py# Master statistical analysis & publication figures
│   └── generate_final_benchmark_partitions.py # Deterministic partition generator
├── data/
│   └── BSD68/                          # Complete BSD68 dataset (test001.png - test068.png)
└── results/
    └── final_benchmark/
        ├── partitions/                 # Certified partition assignments
        │   ├── partition_manifest.json
        │   ├── pc1_conditions.csv      # 3,400 conditions for PC 1
        │   ├── pc2_conditions.csv      # 3,400 conditions for PC 2
        │   └── pc3_conditions.csv      # 3,400 conditions for PC 3
        ├── pc1/                        # PC 1 results directory (10,200 evals)
        │   ├── raw_results.csv
        │   ├── error_log.csv
        │   ├── benchmark_manifest.json
        │   └── reconstructions/        # Saved image PNGs for PC 1
        ├── pc2/                        # PC 2 results directory (10,200 evals)
        ├── pc3/                        # PC 3 results directory (10,200 evals)
        ├── raw_results_merged.csv      # Unified 30,600-evaluation master dataset
        ├── benchmark_manifest_merged.json
        └── final_integrity_report.md   # 10-point cross-PC audit report
```
