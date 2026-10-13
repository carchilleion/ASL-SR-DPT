"""
ASL-SR-DPT 450-Evaluation Pilot Benchmark Interactive Application
Streamlit control dashboard and real-time telemetry console for the Noise-Adaptive ASL-SR-DPT pilot benchmark.

Features:
- Configurable MAX_CPU_WORKERS with system core clamping and single-thread isolation.
- Full 450-evaluation workload scope (10 images x 3 noises x 5 trials x 3 methods).
- Start, Graceful Pause, and Resumption controls with atomic CSV checkpointing.
- Live progress monitoring (completed, remaining, rate, ETA, throughput).
- Live test reconstruction previews and streaming monospace terminal log.
- Automated descriptive analytics and convergence diagnostics.
- Pre-run validation & reproducibility audit reports.
- One-click export center for raw results, summaries, and complete pilot archives.
"""

import os
import sys

# 1. Enforce strict single-threading per process before NumPy/BLAS imports
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
from code.run_new_variant_pilot_450 import (
    get_worker_configuration,
    load_completed_keys,
    EXPECTED_TOTAL_EVALS,
    RAW_CSV_PATH,
    ERROR_CSV_PATH,
    MANIFEST_PATH,
    PAUSE_FILE_PATH,
    OUTPUT_DIR,
    RECON_DIR,
    NOISE_LEVELS,
    PILOT_TRIALS,
    PILOT_IMAGES,
    METHODS_ORDER,
    LIVE_LOG_PATH,
    LIVE_PREVIEW_RECON,
    LIVE_PREVIEW_CLEAN,
    LIVE_PREVIEW_NOISY,
    LIVE_PREVIEW_META,
    PID_FILE_PATH,
)
from code.analyze_new_variant_pilot_450 import run_pilot_analysis, SUMMARY_CSV_PATH, RUNTIME_CSV_PATH, INTEGRITY_MD_PATH
import app

# Set Streamlit page configuration
st.set_page_config(
    page_title="ASL-SR-DPT 450-Evaluation Pilot Benchmark Control Center",
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
        letter-spacing: -0.025em;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 18px;
        font-weight: 400;
    }
    .glass-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.12);
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
        color: #fde047;
        border: 1px solid rgba(234, 179, 8, 0.40);
    }
    .pill-idle {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(148, 163, 184, 0.15);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.30);
    }
    .terminal-box {
        background-color: #050811;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px 16px;
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
    if is_runner_active():
        try:
            with open(PID_FILE_PATH, "r", encoding="utf-8") as pf:
                return int(pf.read().strip())
        except Exception:
            return None
    return None


def start_runner_subprocess(max_workers):
    """Launch benchmark runner in detached background process."""
    if is_runner_active():
        return True

    if os.path.exists(PAUSE_FILE_PATH):
        try:
            os.remove(PAUSE_FILE_PATH)
        except Exception:
            pass

    runner_script = os.path.join(CODE_DIR, "run_new_variant_pilot_450.py")
    cmd = [sys.executable, "-u", runner_script, "--confirm-full"]
    if max_workers:
        cmd.extend(["--max-cpu-workers", str(max_workers)])

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
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "expected_evaluations": EXPECTED_TOTAL_EVALS,
        "completed_valid_evaluations": 0,
        "failed_evaluations": 0,
        "remaining_evaluations": EXPECTED_TOTAL_EVALS,
        "percent_complete": 0.0,
        "evaluations_per_minute": 0.0,
        "status": "NOT_STARTED",
    }


def get_recent_terminal_lines(max_lines=100):
    if not os.path.exists(LIVE_LOG_PATH):
        return "No execution logs recorded yet. Start pilot to begin streaming."
    try:
        with open(LIVE_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        tail = lines[-max_lines:] if len(lines) > max_lines else lines
        return "".join(tail)
    except Exception as exc:
        return f"Error reading log file: {exc}"


# ==============================================================================
# Sidebar Configuration & Execution Controls
# ==============================================================================
with st.sidebar:
    st.markdown("### 🔬 Execution Controls")
    avail_cpus = os.cpu_count() or 1
    req_workers_default = max(1, avail_cpus - 1)

    max_workers_sel = st.slider(
        "MAX_CPU_WORKERS",
        min_value=1,
        max_value=avail_cpus,
        value=req_workers_default,
        help="System logical cores available. Configures worker clamping and thread isolation.",
    )

    st.markdown("---")
    active = is_runner_active()
    active_pid = get_active_runner_pid()

    if active:
        st.markdown(f'<span class="pill-active">● RUNNER ACTIVE (PID {active_pid})</span>', unsafe_allow_html=True)
        if st.button("⏸ Graceful Pause", use_container_width=True):
            pause_runner_subprocess()
            st.warning("Pause signal dispatched. In-flight tasks will complete cleanly.")
            time.sleep(1)
            st.rerun()
    else:
        if os.path.exists(PAUSE_FILE_PATH):
            st.markdown('<span class="pill-paused">⏸ PAUSED</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill-idle">○ IDLE</span>', unsafe_allow_html=True)

        if st.button("🚀 Start / Resume 450-Pilot", use_container_width=True, type="primary"):
            start_runner_subprocess(max_workers_sel)
            st.success("450-evaluation pilot runner launched in background process.")
            time.sleep(1)
            st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Protocol Scope")
    st.markdown(r"""
    - **Dataset:** 10 BSD68 Images (`test001`-`test010`)
    - **Noise Levels:** $\sigma \in \{15, 25, 50\}$
    - **Trials:** 1 through 5
    - **Methods:** 3 Pipelines
      1. Noise-Adaptive ASL-SR-DPT
      2. OMP ($M=38, N=64$)
      3. LASSO-ADMM ($\lambda=0.01$)
    - **Total Workload:** **450 Full Evaluations**
    """)

    st.markdown("---")
    st.markdown("### ⚙️ System Invariants")
    st.caption(f"**Repo Root:** `{REPO_ROOT}`")
    st.caption(f"**Available CPUs:** `{avail_cpus}`")
    st.caption(f"**Code Hash:** `{app.CODE_HASH[:12]}...`")


# ==============================================================================
# Header & Main Title
# ==============================================================================
st.markdown('<div class="main-title">ASL-SR-DPT 450-Evaluation Pilot Benchmark Control Center</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Rigorous evaluation of the Noise-Adaptive ASL-SR-DPT candidate against canonical OMP and LASSO-ADMM baselines across 10 BSD68 pilot images.</div>', unsafe_allow_html=True)


# ==============================================================================
# Real-Time Telemetry & Progress Dashboard (Fragment Auto-Refresh: 2s)
# ==============================================================================
@st.fragment(run_every="2s")
def render_live_telemetry_fragment():
    manifest = get_manifest()
    total_evals = manifest.get("expected_evaluations", EXPECTED_TOTAL_EVALS)
    completed = manifest.get("completed_valid_evaluations", 0)
    failed = manifest.get("failed_evaluations", 0)
    rem = manifest.get("remaining_evaluations", total_evals - completed)
    pct = (completed / total_evals) * 100.0 if total_evals > 0 else 0.0
    rate = manifest.get("evaluations_per_minute", 0.0)
    eta_s = manifest.get("estimated_remaining_seconds", 0.0)
    eta_str = f"{eta_s / 3600.0:.1f}h" if eta_s >= 3600 else f"{eta_s / 60.0:.1f}m"

    # Top KPI Metrics Cards
    kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
    with kpi_c1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="glass-metric-lbl">Completed Valid</div>
            <div class="glass-metric-val">{completed:,} <span style="font-size:1.0rem; color:#94a3b8;">/ {total_evals:,}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_c2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="glass-metric-lbl">Progress</div>
            <div class="glass-metric-val">{pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_c3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="glass-metric-lbl">Throughput</div>
            <div class="glass-metric-val">{rate:.1f} <span style="font-size:0.9rem; color:#94a3b8;">evals/min</span></div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_c4:
        st.markdown(f"""
        <div class="glass-card">
            <div class="glass-metric-lbl">Estimated Remaining</div>
            <div class="glass-metric-val">{eta_str}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_c5:
        st.markdown(f"""
        <div class="glass-card">
            <div class="glass-metric-lbl">Failed Evaluations</div>
            <div class="glass-metric-val" style="color: {'#ef4444' if failed > 0 else '#4ade80'};">{failed}</div>
        </div>
        """, unsafe_allow_html=True)

    st.progress(min(1.0, max(0.0, completed / total_evals)))

    # Real-Time Preview and Monospace Terminal
    st.markdown("### 🖼️ Real-Time Evaluation Telemetry & Live Preview")
    prev_col, term_col = st.columns([1, 1])

    with prev_col:
        meta_data = {}
        if os.path.exists(LIVE_PREVIEW_META):
            try:
                with open(LIVE_PREVIEW_META, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
            except Exception:
                pass

        if meta_data:
            img_id = meta_data.get("image_id", "N/A")
            noise_s = meta_data.get("noise_sigma", "N/A")
            tr = meta_data.get("trial", "N/A")
            meth = meta_data.get("method", "N/A")
            psnr_v = meta_data.get("psnr", 0.0)
            ssim_v = meta_data.get("ssim", 0.0)
            t_v = meta_data.get("runtime_seconds", 0.0)
            st.markdown(f"""
            <div class="live-badge-bar">
                <span class="live-badge"><b>Image:</b> {img_id}</span>
                <span class="live-badge"><b>&sigma;:</b> {int(float(noise_s))}</span>
                <span class="live-badge"><b>Trial:</b> {tr}</span>
                <span class="live-badge"><b>Method:</b> {meth}</span>
                <span class="live-badge"><b>PSNR:</b> {psnr_v:.2f} dB</span>
                <span class="live-badge"><b>SSIM:</b> {ssim_v:.4f}</span>
                <span class="live-badge"><b>Solve Time:</b> {t_v:.2f}s</span>
            </div>
            """, unsafe_allow_html=True)

        img_c1, img_c2, img_c3 = st.columns(3)
        with img_c1:
            st.caption("Clean Reference")
            if os.path.exists(LIVE_PREVIEW_CLEAN):
                st.image(LIVE_PREVIEW_CLEAN, use_container_width=True)
            else:
                st.info("Clean preview pending...")
        with img_c2:
            st.caption("Noisy Input")
            if os.path.exists(LIVE_PREVIEW_NOISY):
                st.image(LIVE_PREVIEW_NOISY, use_container_width=True)
            else:
                st.info("Noisy preview pending...")
        with img_c3:
            st.caption("Reconstructed Output")
            if os.path.exists(LIVE_PREVIEW_RECON):
                st.image(LIVE_PREVIEW_RECON, use_container_width=True)
            else:
                st.info("Reconstruction pending...")

    with term_col:
        st.caption("Live Process Terminal Log")
        log_text = get_recent_terminal_lines(max_lines=60)
        st.markdown(f'<div class="terminal-box">{log_text}</div>', unsafe_allow_html=True)


render_live_telemetry_fragment()

st.markdown("---")

# ==============================================================================
# Analytical Tabs
# ==============================================================================
tab_results, tab_visuals, tab_repro, tab_export = st.tabs([
    "📊 Performance Analytics",
    "🎨 Reconstruction Previews",
    "🛡️ Pre-Run & Reproducibility Audit",
    "📦 Export & Reports",
])

# Tab 1: Performance Analytics
with tab_results:
    st.markdown("### 📊 Descriptive Performance Summary")
    if os.path.exists(RAW_CSV_PATH) and os.path.getsize(RAW_CSV_PATH) > 0:
        df_raw = pd.read_csv(RAW_CSV_PATH)
        st.caption(f"Active raw evaluation records loaded: **{len(df_raw)} records**")

        c_filter1, c_filter2 = st.columns(2)
        with c_filter1:
            noise_filter = st.multiselect("Filter by Noise Level (sigma)", options=[15.0, 25.0, 50.0], default=[15.0, 25.0, 50.0])
        with c_filter2:
            method_filter = st.multiselect("Filter by Method", options=METHODS_ORDER, default=METHODS_ORDER)

        df_filtered = df_raw[(df_raw["noise_sigma"].isin(noise_filter)) & (df_raw["method"].isin(method_filter))]

        if len(df_filtered) > 0:
            st.markdown("#### Performance Grouped by Method")
            summary_by_method = df_filtered.groupby("method").agg(
                Evaluations=("psnr", "count"),
                Mean_PSNR=("psnr", "mean"),
                Std_PSNR=("psnr", "std"),
                Median_PSNR=("psnr", "median"),
                Mean_SSIM=("ssim", "mean"),
                Std_SSIM=("ssim", "std"),
                Median_SSIM=("ssim", "median"),
                Mean_MSE=("mse", "mean"),
                Mean_Runtime_s=("runtime_seconds", "mean"),
                Mean_Residual=("measurement_residual", "mean"),
            ).reset_index()
            st.dataframe(summary_by_method.style.format({
                "Mean_PSNR": "{:.2f}", "Std_PSNR": "{:.2f}", "Median_PSNR": "{:.2f}",
                "Mean_SSIM": "{:.4f}", "Std_SSIM": "{:.4f}", "Median_SSIM": "{:.4f}",
                "Mean_MSE": "{:.6f}", "Mean_Runtime_s": "{:.2f}", "Mean_Residual": "{:.4f}",
            }), use_container_width=True)

            st.markdown("#### Performance Grouped by Method & Noise Level")
            summary_by_noise = df_filtered.groupby(["method", "noise_sigma"]).agg(
                Evaluations=("psnr", "count"),
                Mean_PSNR=("psnr", "mean"),
                Std_PSNR=("psnr", "std"),
                Mean_SSIM=("ssim", "mean"),
                Std_SSIM=("ssim", "std"),
                Mean_Runtime_s=("runtime_seconds", "mean"),
                Mean_Residual=("measurement_residual", "mean"),
            ).reset_index()
            st.dataframe(summary_by_noise.style.format({
                "Mean_PSNR": "{:.2f}", "Std_PSNR": "{:.2f}",
                "Mean_SSIM": "{:.4f}", "Std_SSIM": "{:.4f}",
                "Mean_Runtime_s": "{:.2f}", "Mean_Residual": "{:.4f}",
            }), use_container_width=True)

            # Nonconvex Convergence Diagnostics for Noise-Adaptive ASL-SR-DPT
            asl_df = df_filtered[df_filtered["method"] == "Noise-Adaptive ASL-SR-DPT"]
            if len(asl_df) > 0 and "iteration_count" in asl_df.columns:
                st.markdown("#### Noise-Adaptive ASL-SR-DPT Optimization Diagnostics")
                diag_summary = asl_df.groupby("noise_sigma").agg(
                    Evaluations=("iteration_count", "count"),
                    Effective_Lambda=("effective_lambda", "mean"),
                    Mean_Iterations=("iteration_count", "mean"),
                    Mean_Active_Support_Ratio=("active_support_ratio", "mean"),
                    Low_Band_Retention=("low_band_retention", "mean"),
                    Mid_Band_Retention=("mid_band_retention", "mean"),
                    High_Band_Retention=("high_band_retention", "mean"),
                    Mean_Final_Sigma=("sigma_final", "mean"),
                ).reset_index()
                st.dataframe(diag_summary.style.format({
                    "Effective_Lambda": "{:.4f}", "Mean_Iterations": "{:.1f}",
                    "Mean_Active_Support_Ratio": "{:.4f}", "Low_Band_Retention": "{:.2%}",
                    "Mid_Band_Retention": "{:.2%}", "High_Band_Retention": "{:.2%}",
                    "Mean_Final_Sigma": "{:.4f}",
                }), use_container_width=True)
    else:
        st.info("No raw results available yet. Start the pilot to populate analytics.")


# Tab 2: Visual Previews
with tab_visuals:
    st.markdown("### 🎨 High-Resolution Reconstruction Previews")
    preview_compact_path = os.path.join(OUTPUT_DIR, "reconstruction_preview_compact.png")
    preview_full_path = os.path.join(OUTPUT_DIR, "reconstruction_preview.png")

    if os.path.exists(preview_compact_path):
        st.markdown(r"#### Compact Comparison Grid (`test001`, $\sigma=15$, Trial 1)")
        st.image(preview_compact_path, use_container_width=True)

    if os.path.exists(preview_full_path):
        st.markdown(r"#### Comprehensive 9-Condition Comparison Grid (`test001`, `test004`, `test008` across $\sigma \in \{15, 25, 50\}$)")
        st.image(preview_full_path, use_container_width=True)

    if not os.path.exists(preview_compact_path) and not os.path.exists(preview_full_path):
        st.info("Previews will be automatically compiled upon benchmark completion or via the analysis tool.")


# Tab 3: Reproducibility & Pre-Run Audit
with tab_repro:
    st.markdown("### 🛡️ Pre-Run Validation & Reproducibility Certification")
    repro_rep_path = os.path.join(OUTPUT_DIR, "reproducibility_report.md")
    if os.path.exists(repro_rep_path):
        with open(repro_rep_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info("Reproducibility report pending generation. Run verify_pilot_reproducibility.py to audit.")


# Tab 4: Export Center
with tab_export:
    st.markdown("### 📦 Export Center & Artifact Downloads")
    exp_c1, exp_c2, exp_c3 = st.columns(3)

    with exp_c1:
        if os.path.exists(RAW_CSV_PATH):
            with open(RAW_CSV_PATH, "rb") as f:
                st.download_button("📥 Download raw_results.csv", f, file_name="raw_results.csv", mime="text/csv", use_container_width=True)
        if os.path.exists(SUMMARY_CSV_PATH):
            with open(SUMMARY_CSV_PATH, "rb") as f:
                st.download_button("📥 Download pilot_summary.csv", f, file_name="pilot_summary.csv", mime="text/csv", use_container_width=True)

    with exp_c2:
        if os.path.exists(RUNTIME_CSV_PATH):
            with open(RUNTIME_CSV_PATH, "rb") as f:
                st.download_button("📥 Download runtime_summary.csv", f, file_name="runtime_summary.csv", mime="text/csv", use_container_width=True)
        if os.path.exists(INTEGRITY_MD_PATH):
            with open(INTEGRITY_MD_PATH, "rb") as f:
                st.download_button("📥 Download pilot_integrity_report.md", f, file_name="pilot_integrity_report.md", mime="text/markdown", use_container_width=True)

    with exp_c3:
        if os.path.exists(LIVE_LOG_PATH):
            with open(LIVE_LOG_PATH, "rb") as f:
                st.download_button("📥 Download live_terminal.log", f, file_name="live_terminal.log", mime="text/plain", use_container_width=True)
