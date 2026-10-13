"""
ASL-SR-DPT Final BSD68 Benchmark Standalone Interactive Application
Comprehensive control dashboard for the 51,000-evaluation final thesis benchmark.

Features:
- Configurable MAX_CPU_WORKERS with system core detection and thread isolation.
- Full 51,000-evaluation scope and granular trial/image batch selection.
- Start, Graceful Pause, and Seamless Resumption with atomic CSV checkpointing.
- Live progress monitoring (completed, remaining, ETA, throughput).
- Live metrics analytics and interactive performance charts.
- Automated Wilcoxon paired hypothesis testing with Holm-Bonferroni correction.
- High-resolution visual comparison previews (Compact & Full 9x7).
- One-click export center for raw CSVs, reports, and complete ZIP archives.
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

from code.dataset import load_bsd68, get_bsd68_filenames
from code.final_68_benchmark_runner import (
    run_benchmark,
    get_worker_configuration,
    load_completed_keys,
    EXPECTED_TOTAL_EVALS,
    RAW_CSV_PATH,
    MANIFEST_PATH,
    PAUSE_FILE_PATH,
    OUTPUT_DIR,
    RECON_DIR,
    NOISE_LEVELS,
    TOTAL_TRIALS,
    METHODS_ORDER,
    LIVE_LOG_PATH,
    LIVE_PREVIEW_RECON,
    LIVE_PREVIEW_CLEAN,
    LIVE_PREVIEW_NOISY,
    LIVE_PREVIEW_META,
    PID_FILE_PATH,
)
from code.final_68_analysis import run_analysis
import app

# Set Streamlit page configuration
st.set_page_config(
    page_title="ASL-SR-DPT Final Benchmark Control Center",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Glass Minimalist CSS
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
        letter-spacing: -0.02em;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-bottom: 18px;
        font-weight: 400;
    }
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
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
        margin-bottom: 4px;
    }
    .pill-active {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(34, 197, 94, 0.20);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.40);
        box-shadow: 0 0 10px rgba(34, 197, 94, 0.25);
    }
    .pill-paused {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(234, 179, 8, 0.20);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.40);
    }
    .pill-idle {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
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
        max-height: 340px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.6);
    }
    .terminal-prompt {
        color: #22c55e;
        font-weight: 600;
    }
    .live-badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 12px;
    }
    .live-badge {
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 0.80rem;
        color: #f1f5f9;
        font-family: 'JetBrains Mono', monospace;
    }
    .live-badge b {
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)


def is_runner_active():
    """Check whether the benchmark runner subprocess is actively running."""
    if not os.path.exists(PID_FILE_PATH):
        return False
    try:
        with open(PID_FILE_PATH, "r", encoding="utf-8") as pf:
            pid_str = pf.read().strip()
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


def get_active_runner_pid():
    """Return active PID integer or None."""
    if is_runner_active():
        try:
            with open(PID_FILE_PATH, "r", encoding="utf-8") as pf:
                return int(pf.read().strip())
        except Exception:
            return None
    return None


def start_runner_subprocess(max_workers, trial_range, noise_filter):
    """Launch benchmark runner in detached background process."""
    if is_runner_active():
        return True

    if os.path.exists(PAUSE_FILE_PATH):
        try:
            os.remove(PAUSE_FILE_PATH)
        except Exception:
            pass

    runner_script = os.path.join(CODE_DIR, "final_68_benchmark_runner.py")
    cmd = [sys.executable, "-u", runner_script]

    if max_workers:
        cmd.extend(["--max-cpu-workers", str(max_workers)])
    if trial_range:
        if isinstance(trial_range, (tuple, list)):
            cmd.extend(["--trials", f"{trial_range[0]}-{trial_range[1]}"])
        else:
            cmd.extend(["--trials", str(trial_range)])
    if noise_filter:
        cmd.extend(["--noises", ",".join(str(int(n) if float(n).is_integer() else str(n)) for n in noise_filter)])
    cmd.append("--confirm-full")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_f = open(LIVE_LOG_PATH, "a", encoding="utf-8")

    creation_flags = 0
    if platform.system() == "Windows":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
    )

    with open(PID_FILE_PATH, "w", encoding="utf-8") as pf:
        pf.write(str(proc.pid))

    return True


def pause_runner_subprocess():
    """Signal background runner to halt safely at next task boundary."""
    with open(PAUSE_FILE_PATH, "w", encoding="utf-8") as pf:
        pf.write("PAUSE")


def get_manifest():
    """Read live manifest or construct defaults."""
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "expected_evaluations": EXPECTED_TOTAL_EVALS,
        "completed_evaluations": 0,
        "remaining_evaluations": EXPECTED_TOTAL_EVALS,
        "percent_complete": 0.0,
        "evaluations_per_minute": 0.0,
        "status": "NOT_STARTED",
    }


def get_recent_terminal_lines(max_lines=100):
    """Read the tail of the live terminal log file."""
    if not os.path.exists(LIVE_LOG_PATH):
        return "[Awaiting benchmark startup. Click 'Start / Resume Benchmark' to begin...]"
    try:
        with open(LIVE_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            if not lines:
                return "[Terminal log buffer empty. Waiting for runner events...]"
            tail = lines[-max_lines:]
            return "".join(tail)
    except Exception as exc:
        return f"[Error reading terminal log: {exc}]"


@st.fragment(run_every="2s")
def render_realtime_monitor(actual_w, custom_trials, selected_noises):
    """
    Auto-refreshing real-time fragment executing every 2 seconds.
    Renders status pills, KPI cards, live test preview, terminal log, and controls.
    """
    is_running = is_runner_active()
    active_pid = get_active_runner_pid()
    manifest = get_manifest()

    completed_keys = load_completed_keys(RAW_CSV_PATH)
    completed_count = len(completed_keys)
    remaining_count = max(0, EXPECTED_TOTAL_EVALS - completed_count)
    pct = min(100.0, (completed_count / EXPECTED_TOTAL_EVALS) * 100.0)

    rate_min = manifest.get("evaluations_per_minute", 0.0)
    eta_sec = manifest.get("estimated_remaining_seconds", 0.0)
    if eta_sec > 3600:
        eta_display = f"{eta_sec / 3600:.1f} hrs"
    elif eta_sec > 60:
        eta_display = f"{eta_sec / 60:.1f} mins"
    elif eta_sec > 0:
        eta_display = f"{eta_sec:.0f} secs"
    else:
        eta_display = "—"

    # Status Pill
    if is_running:
        status_pill = f'<span class="pill-active">● RUNNING (PID {active_pid})</span>'
    elif os.path.exists(PAUSE_FILE_PATH):
        status_pill = '<span class="pill-paused">⏸ PAUSED (Boundary Safe)</span>'
    elif completed_count >= EXPECTED_TOTAL_EVALS:
        status_pill = '<span class="pill-active">✔ COMPLETED (51,000 / 51,000)</span>'
    else:
        status_pill = '<span class="pill-idle">○ READY / IDLE</span>'

    # Top Status Bar
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
        <div><b>Engine State:</b> {status_pill}</div>
        <div style="color: #94a3b8; font-size: 0.85rem;">Thesis Workload: <b>68 Images × 3 Noise Levels × 50 Trials × 5 Solvers = 51,000 Evaluations</b></div>
    </div>
    """, unsafe_allow_html=True)

    # 5 KPI METRIC CARDS
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="glass-metric-lbl">Total Expected</div>
            <div class="glass-metric-val">{EXPECTED_TOTAL_EVALS:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="glass-metric-lbl">Completed Valid</div>
            <div class="glass-metric-val" style="color: #38bdf8;">{completed_count:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="glass-metric-lbl">Remaining</div>
            <div class="glass-metric-val" style="color: #f59e0b;">{remaining_count:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="glass-card">
            <div class="glass-metric-lbl">Completion Progress</div>
            <div class="glass-metric-val" style="color: #4ade80;">{pct:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m5:
        st.markdown(f"""
        <div class="glass-card">
            <div class="glass-metric-lbl">Speed / ETA</div>
            <div class="glass-metric-val" style="color: #a78bfa; font-size: 1.45rem;">
                {rate_min:.1f} <span style="font-size: 0.8rem; color: #94a3b8;">ev/m</span>
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">ETA: <b style="color: #f8fafc;">{eta_display}</b></div>
        </div>
        """, unsafe_allow_html=True)

    # Progress bar
    st.progress(pct / 100.0)

    # EXECUTION ACTION BUTTONS
    st.markdown("### ⚡ Execution Controls")
    btn_c1, btn_c2, btn_c3, btn_c4 = st.columns([2, 2, 2, 2])

    with btn_c1:
        if st.button("🚀 Start / Resume Benchmark", type="primary", disabled=is_running, use_container_width=True):
            start_runner_subprocess(actual_w, custom_trials, selected_noises)
            st.rerun()

    with btn_c2:
        if st.button("⏸ Pause Benchmark (Safe Boundary)", type="secondary", disabled=not is_running, use_container_width=True):
            pause_runner_subprocess()
            st.warning("Pause requested. Active worker processes will complete and flush to disk.")
            st.rerun()

    with btn_c3:
        if st.button("📊 Run Statistical Analysis & Reports", use_container_width=True):
            with st.spinner("Executing paired Wilcoxon tests, Holm-Bonferroni corrections, and generating reports..."):
                run_analysis()
            st.success("Analysis and markdown reports successfully updated in results/final_benchmark/!")
            st.rerun()

    with btn_c4:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    st.markdown("---")

    # REAL-TIME TEST RECONSTRUCTION SECTION
    st.markdown("### 🔴 Real-Time Test Reconstruction (Live Update on Every Completed Test)")

    meta_file = LIVE_PREVIEW_META
    has_meta = os.path.exists(meta_file)
    meta = {}
    if has_meta:
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    has_recon = os.path.exists(LIVE_PREVIEW_RECON)
    has_clean = os.path.exists(LIVE_PREVIEW_CLEAN)
    has_noisy = os.path.exists(LIVE_PREVIEW_NOISY)

    if has_meta and meta and has_recon:
        img_id = meta.get("image_id", "—")
        sigma = meta.get("noise_sigma", "—")
        trial = meta.get("trial", "—")
        method = meta.get("method", "—")
        psnr_val = meta.get("psnr", 0.0)
        ssim_val = meta.get("ssim", 0.0)
        mse_val = meta.get("mse", 0.0)
        runtime = meta.get("runtime_seconds", 0.0)
        speed = meta.get("evals_per_minute", 0.0)
        eta_str = meta.get("eta_string", "—")

        # Telemetry badge bar
        st.markdown(f"""
        <div class="live-badge-bar">
            <span class="live-badge">Image: <b>{img_id}</b></span>
            <span class="live-badge">Noise: <b>σ={sigma}</b></span>
            <span class="live-badge">Trial: <b>{trial}</b></span>
            <span class="live-badge">Method: <b>{method}</b></span>
            <span class="live-badge" style="border-color: #22c55e;">PSNR: <b style="color: #4ade80;">{psnr_val:.2f} dB</b></span>
            <span class="live-badge" style="border-color: #38bdf8;">SSIM: <b style="color: #38bdf8;">{ssim_val:.4f}</b></span>
            <span class="live-badge">MSE: <b>{mse_val:.6f}</b></span>
            <span class="live-badge">Solve Time: <b>{runtime:.2f}s</b></span>
            <span class="live-badge">Speed: <b>{speed:.1f} ev/m</b></span>
            <span class="live-badge">ETA: <b>{eta_str}</b></span>
        </div>
        """, unsafe_allow_html=True)

        prev_c1, prev_c2, prev_c3 = st.columns(3)
        with prev_c1:
            st.markdown(f"**1. Clean Reference Image ($x$)** `[{img_id}]`")
            if has_clean:
                st.image(LIVE_PREVIEW_CLEAN, use_container_width=True)
            else:
                st.info("Clean reference image loading...")

        with prev_c2:
            st.markdown(f"**2. Noisy AWGN Input ($y$)** `[σ={sigma}]`")
            if has_noisy:
                st.image(LIVE_PREVIEW_NOISY, use_container_width=True)
            else:
                st.info("Noisy input image loading...")

        with prev_c3:
            st.markdown(fr"**3. Reconstructed Output ($\hat{{x}}$)** `[{method}]`")
            st.image(LIVE_PREVIEW_RECON, use_container_width=True)
    else:
        st.info("Waiting for first test reconstruction to finish. The real-time preview will appear here automatically.")

    st.markdown("---")

    # REAL-TIME TERMINAL LOG
    st.markdown("### 🖥️ Real-Time Terminal Log (Live Stream)")
    log_content = get_recent_terminal_lines(max_lines=120)

    # Render terminal inside pre block
    st.markdown(f"""
    <div class="terminal-window">
{log_content}
    </div>
    """, unsafe_allow_html=True)

    col_log1, col_log2 = st.columns([6, 2])
    with col_log2:
        if os.path.exists(LIVE_LOG_PATH):
            with open(LIVE_LOG_PATH, "rb") as lf:
                st.download_button(
                    "📥 Download Complete Log File",
                    data=lf.read(),
                    file_name="asl_benchmark_live_terminal.log",
                    mime="text/plain",
                    use_container_width=True,
                )


def main():
    avail_cpus = os.cpu_count() or 1
    default_workers = max(1, avail_cpus - 1)
    code_hash = app.CODE_HASH

    # SIDEBAR: Worker & Execution Scope Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Benchmark Control")
        st.markdown(f"""
        <div class="glass-card" style="padding: 12px 16px;">
            <div class="glass-metric-lbl">Host CPU Architecture</div>
            <div style="font-size: 0.85rem; color: #f8fafc; font-weight: 600;">{platform.processor() or platform.machine()}</div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">Logical Cores: <b style="color: #60a5fa;">{avail_cpus}</b></div>
            <div style="font-size: 0.78rem; color: #94a3b8;">Code Hash: <code style="color: #a78bfa;">{code_hash[:10]}...</code></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### CPU Worker Allocation")
        worker_slider = st.slider(
            "MAX_CPU_WORKERS",
            min_value=1,
            max_value=avail_cpus,
            value=default_workers,
            step=1,
            help=f"Limits maximum parallel worker processes. Default: N-1 ({default_workers}) to preserve system responsiveness.",
        )
        req_w, avail_w, actual_w = get_worker_configuration(worker_slider)
        st.caption(f"Configured: {req_w} requested | {actual_w} active processes")

        st.markdown("---")
        st.markdown("#### Execution Scope")
        scope_mode = st.selectbox(
            "Select Evaluation Scope:",
            options=[
                "Full Benchmark (51,000 Evaluations)",
                "Batch: Trials 1 to 5 (5,100 Evaluations)",
                "Batch: Trials 6 to 10 (5,100 Evaluations)",
                "Trial 1 Benchmark Sweep (1,020 Evaluations)",
                "Custom Trial Range",
            ],
            index=0,
        )

        custom_trials = None
        if scope_mode == "Custom Trial Range":
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                t_start = st.number_input("Start Trial", min_value=1, max_value=50, value=1)
            with col_t2:
                t_end = st.number_input("End Trial", min_value=1, max_value=50, value=5)
            custom_trials = (int(t_start), int(t_end))
        elif scope_mode == "Batch: Trials 1 to 5 (5,100 Evaluations)":
            custom_trials = (1, 5)
        elif scope_mode == "Batch: Trials 6 to 10 (5,100 Evaluations)":
            custom_trials = (6, 10)
        elif scope_mode == "Trial 1 Benchmark Sweep (1,020 Evaluations)":
            custom_trials = 1
        else:
            custom_trials = (1, 50)

        st.markdown("---")
        st.markdown("#### Noise Level Filter")
        all_noises = [15.0, 25.0, 50.0]
        selected_noises = st.multiselect("Select Noise Regimes (AWGN σ):", options=all_noises, default=all_noises)

    # MAIN HEADER
    st.markdown('<div class="main-title">ASL-SR-DPT Final BSD68 Benchmark Control Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Real-Time 51,000-Evaluation Production Benchmarking System | Thesis Sparse Recovery Architecture</div>',
        unsafe_allow_html=True,
    )

    # RENDER AUTO-REFRESHING REAL-TIME MONITOR FRAGMENT
    render_realtime_monitor(actual_w, custom_trials, selected_noises)

    # TABS FOR DATA, ANALYTICS, PREVIEWS & EXPORTS
    st.markdown("---")
    tab_data, tab_stats, tab_previews, tab_reports, tab_export = st.tabs([
        "📋 Live Evaluation Records",
        "📈 Statistical Analysis & Performance",
        "🖼️ Visual Result Previews & Inspector",
        "📄 Benchmark Reports",
        "📦 Export & Download Center",
    ])

    # TAB 1: Live Evaluation Records
    with tab_data:
        st.subheader("Raw Evaluation Records")
        if os.path.exists(RAW_CSV_PATH) and os.path.getsize(RAW_CSV_PATH) > 0:
            df_raw = pd.read_csv(RAW_CSV_PATH)
            st.write(f"Displaying latest records from total **{len(df_raw):,}** entries:")
            st.dataframe(df_raw.tail(25), use_container_width=True)

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("##### Records by Method")
                st.write(df_raw["method"].value_counts())
            with col_d2:
                st.markdown("##### Records by Noise Level")
                st.write(df_raw["noise_sigma"].value_counts())
        else:
            st.info("No evaluation records exist yet. Click 'Start / Resume Benchmark' to begin.")

    # TAB 2: Statistical Analysis
    with tab_stats:
        st.subheader("Performance Breakdown & Hypothesis Testing")
        desc_path = os.path.join(OUTPUT_DIR, "descriptive_statistics.csv")
        pair_path = os.path.join(OUTPUT_DIR, "pairwise_statistics.csv")

        if os.path.exists(desc_path):
            df_desc = pd.read_csv(desc_path)
            st.markdown("#### Descriptive Statistics (Mean, 95% Confidence Intervals, Runtimes)")
            st.dataframe(df_desc, use_container_width=True)

        if os.path.exists(pair_path):
            df_pair = pd.read_csv(pair_path)
            st.markdown("#### Paired Inferential Statistics (Wilcoxon Signed-Rank with Holm-Bonferroni Correction, N=68)")
            st.dataframe(df_pair, use_container_width=True)

        # Performance Figures
        st.markdown("#### Performance Visualizations")
        f_col1, f_col2 = st.columns(2)
        psnr_fig = os.path.join(REPO_ROOT, "results", "pilot_final_comparison", "psnr_by_noise.png")
        ssim_fig = os.path.join(REPO_ROOT, "results", "pilot_final_comparison", "ssim_by_noise.png")
        with f_col1:
            if os.path.exists(psnr_fig):
                st.image(psnr_fig, caption="PSNR vs. Noise Level Across BSD68 Dataset", use_container_width=True)
        with f_col2:
            if os.path.exists(ssim_fig):
                st.image(ssim_fig, caption="SSIM vs. Noise Level Across BSD68 Dataset", use_container_width=True)

    # TAB 3: Visual Result Previews & Inspector
    with tab_previews:
        st.subheader("Reconstruction Image Previews & Custom Test Inspector")

        # Custom Completed Test Inspector
        if os.path.exists(RECON_DIR):
            recon_files = [f for f in os.listdir(RECON_DIR) if f.endswith(".png") and not f.startswith("temp_")]
            if recon_files:
                st.markdown("##### 🔍 Inspect Specific Completed Test Reconstruction")
                selected_file = st.selectbox("Select completed reconstruction from archive:", options=sorted(recon_files))
                if selected_file:
                    sel_p = os.path.join(RECON_DIR, selected_file)
                    st.image(sel_p, caption=f"Reconstruction: {selected_file}", use_container_width=True)

        st.markdown("---")
        st.markdown("##### 1. Compact Benchmark Preview (test001, test004, test008 at σ=15, Trial 1)")
        p_compact = os.path.join(OUTPUT_DIR, "final_result_preview_compact.png")
        if not os.path.exists(p_compact):
            p_compact = os.path.join(REPO_ROOT, "results", "pilot_final_comparison", "final_result_preview_compact.png")
        if os.path.exists(p_compact):
            st.image(p_compact, use_container_width=True)

        st.markdown("##### 2. Complete 9-Condition Grid (All 3 Images × 3 Noise Levels)")
        p_full = os.path.join(OUTPUT_DIR, "final_result_preview.png")
        if not os.path.exists(p_full):
            p_full = os.path.join(REPO_ROOT, "results", "pilot_final_comparison", "final_result_preview.png")
        if os.path.exists(p_full):
            st.image(p_full, use_container_width=True)

    # TAB 4: Benchmark Reports
    with tab_reports:
        st.subheader("Generated Benchmark Reports")
        rep_report = os.path.join(OUTPUT_DIR, "final_benchmark_report.md")
        rep_integrity = os.path.join(OUTPUT_DIR, "benchmark_integrity_report.md")

        sub_r1, sub_r2 = st.tabs(["Final Benchmark Report", "Benchmark Integrity Report"])
        with sub_r1:
            if os.path.exists(rep_report):
                with open(rep_report, "r", encoding="utf-8") as rf:
                    st.markdown(rf.read())
            else:
                st.info("Click 'Run Statistical Analysis & Reports' to generate.")
        with sub_r2:
            if os.path.exists(rep_integrity):
                with open(rep_integrity, "r", encoding="utf-8") as rf:
                    st.markdown(rf.read())
            else:
                st.info("Click 'Run Statistical Analysis & Reports' to generate.")

    # TAB 5: Export Center
    with tab_export:
        st.subheader("Results Export & Archive Generation")
        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.markdown("##### Individual File Downloads")
            if os.path.exists(RAW_CSV_PATH):
                with open(RAW_CSV_PATH, "rb") as f:
                    st.download_button(
                        label="Download raw_results.csv",
                        data=f.read(),
                        file_name="raw_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            agg_p = os.path.join(OUTPUT_DIR, "aggregated_image_level.csv")
            if os.path.exists(agg_p):
                with open(agg_p, "rb") as f:
                    st.download_button(
                        label="Download aggregated_image_level.csv",
                        data=f.read(),
                        file_name="aggregated_image_level.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            pair_p = os.path.join(OUTPUT_DIR, "pairwise_statistics.csv")
            if os.path.exists(pair_p):
                with open(pair_p, "rb") as f:
                    st.download_button(
                        label="Download pairwise_statistics.csv",
                        data=f.read(),
                        file_name="pairwise_statistics.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        with col_e2:
            st.markdown("##### Comprehensive ZIP Package")
            st.write("Packages raw results, aggregated datasets, statistics, reports, and manifests into a single archive.")
            if st.button("Build Final Benchmark ZIP Archive", use_container_width=True):
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for root, _, files in os.walk(OUTPUT_DIR):
                        for file in files:
                            full_p = os.path.join(root, file)
                            rel_p = os.path.relpath(full_p, OUTPUT_DIR)
                            zf.write(full_p, arcname=rel_p)
                st.download_button(
                    label="Download final_benchmark_bundle.zip",
                    data=zip_buf.getvalue(),
                    file_name="asl_sr_dpt_final_benchmark_bundle.zip",
                    mime="application/zip",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
