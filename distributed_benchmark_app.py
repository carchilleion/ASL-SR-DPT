"""
ASL-SR-DPT 3-PC Distributed Final Benchmark Interactive Control Dashboard
Streamlit application supporting PC1 to PC3 execution, real-time telemetry, pause, and resumption.

Features:
- Partition selection: PC 1, PC 2, PC 3, and Merged Consolidation Overview.
- Start, Graceful Pause, and Seamless Resumption controls via sentinel flag and subprocess lifecycle management.
- Real-time progress monitoring (completed, remaining, rate, ETA, throughput).
- Live terminal log streaming with auto-scroll monospace console.
- Partition condition inspector (preview assigned 3,400 conditions per PC).
- Single-PC validation suite runner and integrity report viewer.
- 10-Point Cross-PC Merging and Wilcoxon Statistical Analysis center.
"""

import os
import sys

# 1. Thread isolation
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import json
import csv
import io
import zipfile
import platform
import datetime
import shutil
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import psutil

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(REPO_ROOT, "code")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from code.generate_final_benchmark_partitions import compute_file_sha256
from code.run_pc_benchmark import load_completed_keys, get_worker_configuration
import app

BENCHMARK_DIR = os.path.join(REPO_ROOT, "results", "final_benchmark")
PARTITIONS_DIR = os.path.join(BENCHMARK_DIR, "partitions")
MERGED_CSV_PATH = os.path.join(BENCHMARK_DIR, "raw_results_merged.csv")
MERGED_MANIFEST_PATH = os.path.join(BENCHMARK_DIR, "benchmark_manifest_merged.json")
FINAL_REPORT_PATH = os.path.join(BENCHMARK_DIR, "final_integrity_report.md")

# Streamlit Page Setup
st.set_page_config(
    page_title="ASL-SR-DPT Distributed Benchmark Control Center",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Glass Dark CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.025em;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 0.98rem;
        color: #94a3b8;
        margin-bottom: 18px;
        font-weight: 400;
    }
    .glass-card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        backdrop-filter: blur(12px);
    }
    .glass-metric-val {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    .glass-metric-lbl {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 600;
        margin-top: 4px;
    }
    .status-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.05em;
    }
    .badge-running {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-paused {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-complete {
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .badge-idle {
        background: rgba(148, 163, 184, 0.18);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.35);
    }
    .terminal-window {
        background: #0b0f19;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.80rem;
        line-height: 1.45;
        color: #e2e8f0;
        max-height: 380px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.6);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Process & Lifecycle Management Helpers
# -----------------------------------------------------------------------------
def get_pc_dir(partition_id):
    return os.path.join(BENCHMARK_DIR, partition_id)


def get_pid_file(partition_id):
    return os.path.join(get_pc_dir(partition_id), ".runner_pid")


def get_pause_file(partition_id):
    return os.path.join(get_pc_dir(partition_id), ".pause_requested")


def is_partition_running(partition_id):
    pid_path = get_pid_file(partition_id)
    if not os.path.exists(pid_path):
        return False
    try:
        with open(pid_path, "r", encoding="utf-8") as f:
            pid_str = f.read().strip()
        if not pid_str:
            return False
        pid = int(pid_str)
        if psutil.pid_exists(pid):
            proc = psutil.Process(pid)
            if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                return True
    except Exception:
        pass
    return False


def get_partition_pid(partition_id):
    if is_partition_running(partition_id):
        try:
            with open(get_pid_file(partition_id), "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except Exception:
            return None
    return None


def get_partition_status(partition_id):
    if is_partition_running(partition_id):
        return "RUNNING"
    if os.path.exists(get_pause_file(partition_id)):
        return "PAUSED"
    raw_path = os.path.join(get_pc_dir(partition_id), "raw_results.csv")
    if os.path.exists(raw_path):
        keys = load_completed_keys(raw_path)
        if len(keys) >= 10200:
            return "COMPLETED"
        elif len(keys) > 0:
            return "INCOMPLETE"
    return "IDLE"


def start_partition_runner(partition_id, max_workers, save_images="all"):
    """Launch partition runner subprocess in background."""
    if is_partition_running(partition_id):
        return True

    # Remove pause file if resuming
    pause_f = get_pause_file(partition_id)
    if os.path.exists(pause_f):
        try:
            os.remove(pause_f)
        except Exception:
            pass

    pc_dir = get_pc_dir(partition_id)
    os.makedirs(pc_dir, exist_ok=True)
    live_log = os.path.join(pc_dir, "live_terminal.log")
    log_file = open(live_log, "a", encoding="utf-8")

    clamped_workers = max(1, min(int(max_workers), 10, os.cpu_count() or 1))
    runner_script = os.path.join(CODE_DIR, "run_pc_benchmark.py")
    cmd = [
        sys.executable,
        "-u",
        runner_script,
        "--partition-id",
        partition_id,
        "--max-cpu-workers",
        str(clamped_workers),
        "--save-images",
        str(save_images),
        "--max-retries",
        "2",
        "--max-consecutive-failures",
        "5",
    ]

    creation_flags = 0
    if platform.system() == "Windows":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
    )

    try:
        with open(get_pid_file(partition_id), "w", encoding="utf-8") as pf:
            pf.write(str(proc.pid))
    except Exception:
        pass

    time.sleep(0.5)
    return True


def pause_partition_runner(partition_id):
    """Signal partition runner to pause gracefully via .pause_requested."""
    pause_f = get_pause_file(partition_id)
    os.makedirs(os.path.dirname(pause_f), exist_ok=True)
    try:
        with open(pause_f, "w", encoding="utf-8") as f:
            f.write(f"pause_requested_at={datetime.datetime.now().isoformat()}\n")
    except Exception:
        pass


def stop_partition_runner(partition_id):
    """Terminate the active runner process cleanly."""
    pid = get_partition_pid(partition_id)
    if pid and psutil.pid_exists(pid):
        try:
            p = psutil.Process(pid)
            p.terminate()
            p.wait(timeout=3)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    pid_f = get_pid_file(partition_id)
    if os.path.exists(pid_f):
        try:
            os.remove(pid_f)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Sidebar: Navigation & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🖥️ Distributed Benchmark")
    st.markdown("ASL-SR-DPT Final 30,600-Evaluation Suite")
    st.markdown("---")

    selected_mode = st.radio(
        "Active Node Selection",
        [
            "PC 1 Partition (10,200 evals)",
            "PC 2 Partition (10,200 evals)",
            "PC 3 Partition (10,200 evals)",
            "Merged 3-PC Overview (30,600 evals)",
        ],
        index=0,
    )

    partition_map = {
        "PC 1 Partition (10,200 evals)": "pc1",
        "PC 2 Partition (10,200 evals)": "pc2",
        "PC 3 Partition (10,200 evals)": "pc3",
        "Merged 3-PC Overview (30,600 evals)": "merged",
    }
    active_partition = partition_map[selected_mode]

    st.markdown("---")
    st.markdown("### ⚙️ Hardware Limits")
    avail_cpus = os.cpu_count() or 1
    max_allowed = min(10, avail_cpus)
    worker_slider = st.slider(
        "MAX_CPU_WORKERS",
        min_value=1,
        max_value=max_allowed,
        value=max_allowed,
        help="Hard ceiling on worker processes. Single-thread BLAS isolation enforced per worker.",
    )

    st.markdown(f"**Logical CPUs Detected:** `{avail_cpus}`")
    st.markdown(f"**Max Worker Limit:** `{max_allowed}`")
    st.markdown("---")

    st.markdown("### 🖼️ Image Artifacts")
    image_save_option = st.selectbox(
        "Image Saving Policy",
        ["all", "trial1", "previews", "none"],
        format_func=lambda x: {
            "all": "💾 Save All 10,200 Images (Warning: ~2GB per PC) [Default]",
            "trial1": "📸 Save All Trial 1 Images Only",
            "previews": "🔍 Key Previews Only (test001, test004, test008)",
            "none": "🚫 Metrics Only (No PNGs)",
        }.get(x, x),
        index=0,
        help="Controls whether reconstructed PNG images are saved to results/final_benchmark/<pc>/reconstructions/.",
    )
    st.markdown("---")

    st.markdown("### 📋 Quick Links")
    st.markdown("- [Canonical Config](file:///c:/Users/Carlo%20Mendoza/OneDrive/Desktop/ASL-SR-DPT/configs/final_benchmark_3pc_config.json)")
    st.markdown("- [Partitions Directory](file:///c:/Users/Carlo%20Mendoza/OneDrive/Desktop/ASL-SR-DPT/results/final_benchmark/partitions/)")


# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">ASL-SR-DPT Final Benchmark Control Center</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Distributed 3-PC Architecture &bull; 68 BSD68 Images &bull; 3 Noise Regimes &bull; 50 Trials &bull; 30,600 Total Solver Evaluations</div>', unsafe_allow_html=True)


# =============================================================================
# View 1: Single Partition Runner (PC1, PC2, PC3)
# =============================================================================
if active_partition in ("pc1", "pc2", "pc3"):
    p_id = active_partition
    p_name = p_id.upper()
    pc_dir = get_pc_dir(p_id)
    raw_csv = os.path.join(pc_dir, "raw_results.csv")
    manifest_f = os.path.join(pc_dir, "benchmark_manifest.json")
    live_log = os.path.join(pc_dir, "live_terminal.log")
    report_md = os.path.join(pc_dir, "integrity_report.md")

    # Partition condition CSV
    part_csv = os.path.join(PARTITIONS_DIR, f"{p_id}_conditions.csv")
    assigned_conds_count = 3400
    expected_evals_count = 10200

    # Auto-refreshing telemetry fragment
    @st.fragment(run_every=2)
    def render_partition_dashboard():
        status = get_partition_status(p_id)
        running = (status == "RUNNING")
        pid = get_partition_pid(p_id)

        # Header Status Bar
        badge_cls = "badge-running" if status == "RUNNING" else (
            "badge-paused" if status == "PAUSED" else (
                "badge-complete" if status == "COMPLETED" else "badge-idle"
            )
        )

        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h3 style="margin: 0; color: #f8fafc;">Node: {p_name} Partition Control</h3>
            <div>
                <span class="status-badge {badge_cls}">STATUS: {status} {'(PID: ' + str(pid) + ')' if pid else ''}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Controls Row
        c_btn1, c_btn2, c_btn3, c_btn4 = st.columns([1.5, 1.5, 1.5, 3])
        with c_btn1:
            if not running:
                btn_lbl = "▶️ Resume Execution" if status == "PAUSED" else "▶️ Start Benchmark"
                if st.button(btn_lbl, type="primary", use_container_width=True):
                    start_partition_runner(p_id, worker_slider, save_images=image_save_option)
                    st.rerun()
            else:
                st.button("▶️ Running...", disabled=True, use_container_width=True)

        with c_btn2:
            if running:
                if st.button("⏸️ Pause Execution", type="secondary", use_container_width=True):
                    pause_partition_runner(p_id)
                    st.toast(f"Pause signal sent to {p_name}. Runner will pause after current wave.")
                    st.rerun()
            else:
                st.button("⏸️ Pause", disabled=True, use_container_width=True)

        with c_btn3:
            if running:
                if st.button("🛑 Force Stop", use_container_width=True):
                    stop_partition_runner(p_id)
                    st.rerun()
            else:
                st.button("🛑 Stop", disabled=True, use_container_width=True)

        with c_btn4:
            st.caption(f"Allocated: **3,400 conditions** &bull; Target: **10,200 evaluations** &bull; Workers: **{worker_slider}**")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Telemetry metrics
        completed_count = 0
        if os.path.exists(raw_csv):
            completed_count = len(load_completed_keys(raw_csv))

        remaining_count = max(0, expected_evals_count - completed_count)
        pct = (completed_count / expected_evals_count) * 100.0 if expected_evals_count > 0 else 0.0

        elapsed_str, rate_str, eta_str = "0.0s", "0.0", "N/A"
        if os.path.exists(manifest_f):
            try:
                with open(manifest_f, "r", encoding="utf-8") as mf:
                    mdata = json.load(mf)
                    elapsed_str = f"{mdata.get('elapsed_seconds', 0.0):.1f}s"
                    rate_str = f"{mdata.get('evaluations_per_minute', 0.0):.1f}"
                    rem_s = mdata.get("estimated_remaining_seconds", 0.0)
                    eta_str = f"{rem_s / 3600.0:.1f}h" if rem_s >= 3600 else f"{rem_s / 60.0:.1f}m"
            except Exception:
                pass

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="glass-metric-val">{completed_count:,}</div>
                <div class="glass-metric-lbl">Completed Valid Evals</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="glass-card">
                <div class="glass-metric-val">{remaining_count:,}</div>
                <div class="glass-metric-lbl">Remaining Tasks</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="glass-card">
                <div class="glass-metric-val">{pct:.1f}%</div>
                <div class="glass-metric-lbl">Progress</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="glass-card">
                <div class="glass-metric-val">{rate_str}</div>
                <div class="glass-metric-lbl">Throughput (Evals/Min)</div>
            </div>
            """, unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
            <div class="glass-card">
                <div class="glass-metric-val">{eta_str}</div>
                <div class="glass-metric-lbl">Estimated ETA</div>
            </div>
            """, unsafe_allow_html=True)

        st.progress(min(1.0, completed_count / expected_evals_count))

        # Terminal & Log Box
        st.markdown(f"#### 📟 {p_name} Real-Time Terminal Log")
        log_text = "No log output recorded yet."
        if os.path.exists(live_log):
            try:
                with open(live_log, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    log_text = "".join(lines[-40:]) if lines else "Waiting for events..."
            except Exception as e:
                log_text = f"Error reading log: {e}"

        st.markdown(f'<div class="terminal-window">{log_text}</div>', unsafe_allow_html=True)

    render_partition_dashboard()

    # Tabs for detailed inspection
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Assigned Conditions", "🖼️ Image Gallery", "🛡️ Validation & Audit", "💾 Export Data"])

    with tab1:
        st.markdown(f"#### Conditions Assigned to {p_name} (Total: 3,400)")
        if os.path.exists(part_csv):
            df_part = pd.read_csv(part_csv)
            st.dataframe(df_part.head(100), use_container_width=True)
            st.caption(f"Showing first 100 of {len(df_part)} conditions from {part_csv}")
        else:
            st.warning("Partition conditions CSV not found. Please run partition generator.")

    with tab2:
        st.markdown(f"#### Reconstructed Images Gallery ({p_name})")
        recon_dir = os.path.join(pc_dir, "reconstructions")
        if os.path.exists(recon_dir):
            all_pngs = [f for f in os.listdir(recon_dir) if f.endswith(".png")]
            if all_pngs:
                st.success(f"Found {len(all_pngs)} saved PNG images in `{pc_dir}/reconstructions/`.")
                # Group by image_id
                img_stems = sorted(list(set(f.split("_")[0] for f in all_pngs if f.startswith("test"))))
                if img_stems:
                    sel_img = st.selectbox("Select Image to Inspect", img_stems, index=0)
                    sel_pngs = [f for f in all_pngs if f.startswith(sel_img)]

                    clean_f = f"{sel_img}_clean.png"
                    clean_path = os.path.join(recon_dir, clean_f)

                    # Show across noise levels
                    for n_val in [15, 25, 50]:
                        st.markdown(f"##### Noise $\\sigma = {n_val}$ (Trial 1)")
                        c_cols = st.columns(5)
                        noisy_f = f"{sel_img}_sigma{n_val}_trial1_noisy.png"
                        asl_f = f"{sel_img}_sigma{n_val}_trial1_noise-adaptive_asl-sr-dpt.png"
                        omp_f = f"{sel_img}_sigma{n_val}_trial1_omp.png"
                        admm_f = f"{sel_img}_sigma{n_val}_trial1_lasso-admm.png"

                        with c_cols[0]:
                            if os.path.exists(clean_path):
                                st.image(clean_path, caption="Clean Ground Truth", use_column_width=True)
                            else:
                                st.caption("Clean image pending")

                        with c_cols[1]:
                            p = os.path.join(recon_dir, noisy_f)
                            if os.path.exists(p):
                                st.image(p, caption=f"Noisy Input (sigma={n_val})", use_column_width=True)
                            else:
                                st.caption("Noisy pending")

                        with c_cols[2]:
                            p = os.path.join(recon_dir, asl_f)
                            if os.path.exists(p):
                                st.image(p, caption="Noise-Adaptive ASL", use_column_width=True)
                            else:
                                st.caption("ASL pending")

                        with c_cols[3]:
                            p = os.path.join(recon_dir, omp_f)
                            if os.path.exists(p):
                                st.image(p, caption="Canonical OMP", use_column_width=True)
                            else:
                                st.caption("OMP pending")

                        with c_cols[4]:
                            p = os.path.join(recon_dir, admm_f)
                            if os.path.exists(p):
                                st.image(p, caption="LASSO-ADMM", use_column_width=True)
                            else:
                                st.caption("ADMM pending")
            else:
                st.info("No reconstructed PNGs saved yet. Images will appear here automatically as evaluations run (governed by the Image Saving Policy in the sidebar).")
        else:
            st.info("Reconstructions folder will be initialized on first run.")

    with tab3:
        st.markdown(f"#### Single-Node Integrity Validation ({p_name})")
        if st.button(f"🔍 Run {p_name} Validation Audit", key=f"val_{p_id}"):
            with st.spinner(f"Auditing {p_name} records..."):
                val_script = os.path.join(CODE_DIR, "validate_pc_benchmark.py")
                res = subprocess.run([sys.executable, val_script, "--partition-id", p_id], capture_output=True, text=True)
                st.code(res.stdout + ("\n" + res.stderr if res.stderr else ""), language="bash")

        if os.path.exists(report_md):
            with open(report_md, "r", encoding="utf-8") as f:
                st.markdown(f.read())

    with tab4:
        st.markdown(f"#### Export {p_name} Artifacts")
        if os.path.exists(raw_csv):
            with open(raw_csv, "rb") as f:
                st.download_button(
                    f"⬇️ Download {p_name} raw_results.csv",
                    data=f.read(),
                    file_name=f"{p_id}_raw_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.info("No raw results available to download yet.")


# =============================================================================
# View 2: Merged 3-PC Overview
# =============================================================================
else:
    st.markdown("### 🌐 Distributed Benchmark Fleet Overview")
    st.markdown("Consolidated status across all 3 nodes (PC 1, PC 2, PC 3) totaling **30,600 evaluations**.")

    # Status Cards across the 3 PCs
    pc_cols = st.columns(3)
    pc_data = []

    for idx, pc_key in enumerate(["pc1", "pc2", "pc3"]):
        status = get_partition_status(pc_key)
        pid = get_partition_pid(pc_key)
        raw_p = os.path.join(get_pc_dir(pc_key), "raw_results.csv")
        c_count = len(load_completed_keys(raw_p)) if os.path.exists(raw_p) else 0

        badge_cls = "badge-running" if status == "RUNNING" else (
            "badge-paused" if status == "PAUSED" else (
                "badge-complete" if status == "COMPLETED" else "badge-idle"
            )
        )

        with pc_cols[idx]:
            st.markdown(f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: #f8fafc;">{pc_key.upper()} Node</h4>
                    <span class="status-badge {badge_cls}">{status}</span>
                </div>
                <div style="margin-top: 14px;">
                    <div class="glass-metric-val">{c_count:,} <span style="font-size: 1rem; color: #94a3b8;">/ 10,200</span></div>
                    <div class="glass-metric-lbl">Completed Evaluations ({c_count / 10200 * 100:.1f}%)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(1.0, c_count / 10200))

    st.markdown("---")

    # Consolidation Actions
    m_tab1, m_tab2, m_tab3 = st.tabs(["🧩 Merge & 10-Point Audit", "📈 Inferential Hypothesis Testing", "📦 Full Export Archive"])

    with m_tab1:
        st.markdown("#### Combine PC1, PC2, and PC3 Datasets")
        st.markdown("Applies the strict 10-point cross-PC audit before producing `raw_results_merged.csv` (30,600 records).")

        if st.button("🚀 Execute 10-Point Cross-PC Merge", type="primary", use_container_width=True):
            with st.spinner("Executing cryptographic audit and dataset merge..."):
                merge_script = os.path.join(CODE_DIR, "merge_distributed_benchmark.py")
                res = subprocess.run([sys.executable, merge_script], capture_output=True, text=True)
                st.code(res.stdout + ("\n" + res.stderr if res.stderr else ""), language="bash")

        if os.path.exists(FINAL_REPORT_PATH):
            st.markdown("---")
            with open(FINAL_REPORT_PATH, "r", encoding="utf-8") as f:
                st.markdown(f.read())

    with m_tab2:
        st.markdown("#### Wilcoxon Signed-Rank Inferential Analysis")
        st.markdown("Aggregates 50 trials into 68 independent image-level observations and tests for statistical significance with Holm-Bonferroni correction.")

        if st.button("📊 Run Paired Statistical Tests", type="primary", use_container_width=True):
            with st.spinner("Computing paired Wilcoxon tests, bootstrap CIs, and figures..."):
                stat_script = os.path.join(CODE_DIR, "analyze_final_benchmark_30600.py")
                res = subprocess.run([sys.executable, stat_script], capture_output=True, text=True)
                st.code(res.stdout + ("\n" + res.stderr if res.stderr else ""), language="bash")

        # Display figures if generated
        psnr_plot = os.path.join(BENCHMARK_DIR, "final_benchmark_psnr_ssim_curves.png")
        time_plot = os.path.join(BENCHMARK_DIR, "final_benchmark_runtime_boxplots.png")
        test_csv = os.path.join(BENCHMARK_DIR, "final_inferential_tests.csv")

        if os.path.exists(test_csv):
            st.markdown("##### Inferential Test Results Table")
            df_test = pd.read_csv(test_csv)
            st.dataframe(df_test, use_container_width=True)

        if os.path.exists(psnr_plot) and os.path.exists(time_plot):
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                st.image(psnr_plot, caption="Reconstruction Quality (PSNR/SSIM) vs. Noise Regime", use_column_width=True)
            with c_p2:
                st.image(time_plot, caption="Runtime Distribution Across 68 Images", use_column_width=True)

    with m_tab3:
        st.markdown("#### Complete Benchmark Archive")
        if os.path.exists(MERGED_CSV_PATH):
            with open(MERGED_CSV_PATH, "rb") as f:
                st.download_button(
                    "⬇️ Download Merged raw_results_merged.csv (30,600 rows)",
                    data=f.read(),
                    file_name="raw_results_merged.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.info("Merged CSV will be available for download once PC1, PC2, and PC3 are merged.")
