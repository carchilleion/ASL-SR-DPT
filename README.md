# ASL-SR-DPT

**Accelerated Single-Loop Sparse Recovery with Dynamic Parameter Tuning for Mobile Image Denoising**

Official standalone pilot-benchmark application designed for distributed execution across two research teammates:
- **Team A (Khevin):** 5 BSD68 images (`test001`–`test005`) × 3 noise levels × 5 trials × 3 solvers = **225 evaluations**
- **Team B (Marc):** 5 BSD68 images (`test006`–`test010`) × 3 noise levels × 5 trials × 3 solvers = **225 evaluations**
- **Combined:** $225 + 225 = \mathbf{450\text{ evaluations}}$

Evaluated Solvers:
1. **`ASL-SR-DPT`**: Frozen A6 architecture with DC preservation ($M_{\text{ac}}=37, N_{\text{ac}}=63$)
2. **`OMP`**: Standard full-patch Orthogonal Matching Pursuit ($M=38, N=64$)
3. **`LASSO-ADMM`**: Standard full-patch $\ell_1$-ADMM ($M=38, N=64$)

---

## 1. Workload Division & Member Assignment

The 450-evaluation pilot benchmark is split into two **strictly non-overlapping** workloads:

| Workload Track | Teammate | Assigned BSD68 Images | Noise Regimes ($\sigma$) | Trials | Solvers | Total Evaluations | Output Directory |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| **TEAM A** | **Khevin** | `test001`, `test002`, `test003`, `test004`, `test005` | $15, 25, 50$ | $1, 2, 3, 4, 5$ | ASL-SR-DPT, OMP, LASSO-ADMM | **225** | `results/team_a/` |
| **TEAM B** | **Marc** | `test006`, `test007`, `test008`, `test009`, `test010` | $15, 25, 50$ | $1, 2, 3, 4, 5$ | ASL-SR-DPT, OMP, LASSO-ADMM | **225** | `results/team_b/` |
| **COMBINED** | *Central Merge* | All 10 Pilot Images (`test001`–`test010`) | $15, 25, 50$ | $1, 2, 3, 4, 5$ | All 3 Solvers | **450** | `results/pilot_combined/` |

### Key Protocol Guarantees:
- **Zero Overlapping Keys:** Teammates process mutually disjoint image sets. No patches are split; full images are reconstructed end-to-end.
- **Deterministic Reproducibility:** Seeds for sensing matrices and noise realizations are strictly computed from the experiment key:
  - $$\text{sensing\_seed} = \text{base\_seed} + \text{trial}$$
  - $$\text{noise\_seed} = \text{base\_seed} + \text{trial} \times 100000 + \sigma \times 1000 + \text{clean\_id}$$
- **Bitwise Noise Parity:** In every condition $(image, \sigma, trial)$, all 3 solvers operate on the bitwise identical noisy image.
- **Resume Safety:** If interrupted, the app skips already completed keys loaded from CSV.
- **Strict Single-Threading:** CPU libraries are pinned to 1 thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, etc.) to guarantee fair, isolated per-patch runtime measurements.

---

## 2. Running in VMware Workstation Pro with Ubuntu (Step-by-Step Guide)

This guide walks **Khevin (Team A)** and **Marc (Team B)** through running the benchmark inside an **Ubuntu Linux Virtual Machine** using **VMware Workstation Pro**.

### Step 2.1: VMware VM Setup
1. **Download Ubuntu ISO:** Download **Ubuntu 22.04 LTS** or **Ubuntu 24.04 LTS Desktop 64-bit ISO** from [ubuntu.com](https://ubuntu.com/download/desktop).
2. **Create New Virtual Machine in VMware Workstation Pro:**
   - Click **File > New Virtual Machine...**
   - Select **Typical (recommended)** > click **Next**.
   - Select **Installer disc image file (iso)** and browse to your Ubuntu ISO.
   - Enter your name, username, and password in the **Easy Install** prompt.
   - **Virtual Machine Name:** e.g., `Ubuntu-ASL-SR-DPT-TeamA` or `TeamB`.
   - **Disk Capacity:** Set **Maximum disk size** to at least **25–30 GB** (Split or single file).
   - In **Customize Hardware**:
     - **Memory:** Allocate at least **4 GB RAM** (recommended: **8 GB** if your host has 16 GB+).
     - **Processors:** Assign at least **2 to 4 CPU cores** (ensure the VM has responsive GUI performance).
     - **Network Adapter:** Set to **NAT** (default, shares host internet connection).
   - Click **Finish** and let Ubuntu complete installation.

3. **Install Open VM Tools (for clipboard sharing & file drag-and-drop):**
   Once logged into the Ubuntu desktop, open a terminal (`Ctrl + Alt + T`) and run:
   ```bash
   sudo apt update
   sudo apt install -y open-vm-tools-desktop
   ```
   *(Restart the VM if prompted to enable clipboard and shared folders).*

---

### Step 2.2: Install Ubuntu System Dependencies
Inside the Ubuntu terminal, run:
```bash
sudo apt update && sudo apt install -y git python3 python3-pip python3-venv build-essential
```

---

### Step 2.3: Clone the GitHub Repository
Clone the repository to your Ubuntu home directory:
```bash
cd ~
git clone https://github.com/carchilleion/ASL-SR-DPT.git
cd ASL-SR-DPT
```

---

### Step 2.4: Create Python Virtual Environment & Install Requirements
```bash
# 1. Create a dedicated virtual environment
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate

# 3. Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 2.5: Verify Setup with Automated Test Suite
Run the automated test suite to confirm configuration integrity and deterministic seeding:
```bash
pytest tests/test_pilot_application.py
```
*Expected output: All 5 tests pass (`5 passed in ~1s`).*

---

### Step 2.6: Launch the Pilot Benchmark Application

#### For Khevin (Team A):
Make the launcher executable and start:
```bash
chmod +x run_team_a.sh
./run_team_a.sh
```
*Alternatively:*
```bash
python3 -m streamlit run app.py -- --team "Team A"
```

#### For Marc (Team B):
Make the launcher executable and start:
```bash
chmod +x run_team_b.sh
./run_team_b.sh
```
*Alternatively:*
```bash
python3 -m streamlit run app.py -- --team "Team B"
```

The Streamlit web dashboard will automatically open in Ubuntu's Firefox browser at:
```
http://localhost:8501
```

*(Note: If you prefer to view the dashboard from your Windows host browser, run `ip addr` in Ubuntu to find your VM's IP address, e.g. `192.168.x.x`, and open `http://<VM-IP>:8501` in Chrome/Edge on Windows).*

---

## 3. Step-by-Step Pilot Workflow in the Streamlit GUI

### Step 3.1: Pass the 5 Pre-Run Safety Gates
In the left sidebar or **Tab 2 ("🛡️ Pilot Safety Gates")**:
1. **Gate 1 — Environment Gate:** Confirms Python version and single-thread locking (`OMP=1, MKL=1`).
2. **Gate 2 — Config Integrity Gate:** Validates SHA-256 hash of `configs/pilot_config.json`.
3. **Gate 3 — Deterministic Reproducibility Gate:** Click **"Run Reproducibility Check"**. The app executes reference condition `(test001, σ=15, Trial 1)` twice to verify exact $\Delta = 0.0$ reproducibility. *(Use "Fast Pre-flight Check" for instant verification).*
4. **Gate 4 — Workload Verification Gate:** Confirms all 5 assigned BSD68 images are present in `data/BSD68/`.
5. **Gate 5 — Benchmark Start Gate:** Check the confirmation box to unlock execution.

### Step 3.2: Execute the Benchmark
Navigate to **Tab 1 ("📊 Live Benchmark Dashboard")**:
1. Click **"▶️ Start Benchmark Workload"**.
2. Monitor live metrics:
   - Progress bar (0/225 to 225/225) and ETA.
   - Real-time noisy image and spatial reconstruction preview.
   - Live metrics table (PSNR, SSIM, MSE, Solve Time, ms/patch, Residuals).
3. If you need to stop, click **"⏹️ Pause / Stop"**. You can resume at any time; completed records are never re-run.

### Step 3.3: Export Results Package
When all 225 evaluations are finished:
1. Navigate to **Tab 4 ("📦 Export & Merge Center")**.
2. Click **"Generate Teammate ZIP Package"**.
3. Download the certified archive:
   - Khevin: `asl_sr_dpt_pilot_team_a.zip`
   - Marc: `asl_sr_dpt_pilot_team_b.zip`
4. Send your ZIP archive or `results/team_*/raw/pilot_raw_results_team_*.csv` to Carlo for central merging.

---

## 4. Alternative: Running Directly on Windows PC

If you prefer to run natively on Windows without VMware:
1. Install Python 3.10–3.12 (64-bit) from [python.org](https://www.python.org/).
2. In PowerShell / Command Prompt:
   ```powershell
   git clone https://github.com/carchilleion/ASL-SR-DPT.git
   cd ASL-SR-DPT
   pip install -r requirements.txt
   ```
3. Use the one-click Windows launchers:
   - **For Khevin (Team A):** Double-click `run_team_a.bat` (or run `.\run_team_a.ps1`).
   - **For Marc (Team B):** Double-click `run_team_b.bat` (or run `.\run_team_b.ps1`).

---

## 5. Central Merging & Master Integrity Audit

Once Khevin (Team A) and Marc (Team B) submit their results, run the central merge utility:

```bash
python3 merge_pilot_results.py \
    --team-a results/team_a/raw/pilot_raw_results_team_a.csv \
    --team-b results/team_b/raw/pilot_raw_results_team_b.csv \
    --team-a-failures results/team_a/failures/pilot_failures.csv \
    --team-b-failures results/team_b/failures/pilot_failures.csv \
    --config configs/pilot_config.json \
    --output results/pilot_combined
```

### Verified Merged Outputs in `results/pilot_combined/`:
- `pilot_combined_raw.csv`: Master 450-evaluation table.
- `pilot_combined_summary.csv`: Stratified summary statistics (PSNR, SSIM, solve time, residuals).
- `pilot_combined_failures.csv`: Consolidated failure log (0 failures in clean run).
- `pilot_combined_manifest.json`: SHA-256 provenance manifest.
- `final_pilot_audit.md`: Formal verification report proving 450/450 unique evaluations, 0 duplicates, and full seed/config consistency.

---

## 6. Repository Layout

```
ASL-SR-DPT/
├── app.py                            # Streamlit interactive pilot application
├── merge_pilot_results.py            # Central merge and master audit utility
├── run_team_a.sh / run_team_b.sh     # Linux launchers for Ubuntu (Khevin / Marc)
├── run_team_a.bat / run_team_b.bat   # Windows batch launchers (Khevin / Marc)
├── run_team_a.ps1 / run_team_b.ps1   # Windows PowerShell launchers
├── requirements.txt                  # Python dependencies
├── configs/
│   └── pilot_config.json             # Master frozen configuration & SHA-256 hash
├── data/
│   └── BSD68/                        # Benchmark images (test001.png - test068.png)
├── tests/
│   └── test_pilot_application.py     # Verification tests
└── results/
    ├── team_a/                       # Khevin's output directory (225 evaluations)
    │   ├── raw/pilot_raw_results_team_a.csv
    │   ├── failures/pilot_failures.csv
    │   └── summaries/team_report.md
    ├── team_b/                       # Marc's output directory (225 evaluations)
    │   ├── raw/pilot_raw_results_team_b.csv
    │   ├── failures/pilot_failures.csv
    │   └── summaries/team_report.md
    └── pilot_combined/               # Master merged results (450 evaluations)
        ├── pilot_combined_raw.csv
        ├── pilot_combined_summary.csv
        ├── pilot_combined_manifest.json
        └── final_pilot_audit.md
```
