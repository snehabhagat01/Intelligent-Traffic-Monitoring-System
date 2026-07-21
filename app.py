"""
app.py
======
Main Streamlit entry point for the Intelligent Traffic Monitoring System.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import pandas as pd
import streamlit as st

import config
from analytics import TrafficAnalytics
from detector import ModelLoadError, VehicleDetector
from report_generator import (
    generate_pdf_report,
    generate_traffic_summary_csv,
    generate_vehicle_statistics_csv,
)
from utils import (
    init_session_state,
    logger,
    push_upload_history,
    reset_analysis_state,
    save_uploaded_file,
    timestamp_now,
    validate_video_file,
)

# --------------------------------------------------------------------------- #
# PAGE CONFIG (must be the first Streamlit call)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()


# --------------------------------------------------------------------------- #
# STYLE INJECTION
# --------------------------------------------------------------------------- #
def load_css() -> None:
    """Inject the custom stylesheet into the Streamlit app."""
    try:
        if config.STYLE_CSS_PATH.exists():
            css = config.STYLE_CSS_PATH.read_text(encoding="utf-8")
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load stylesheet: %s", exc)


def image_to_base64(path: Path) -> str:
    """Encode a local image as base64 for embedding in HTML/CSS."""
    if not path or not path.exists():
        return ""
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not encode image %s: %s", path, exc)
        return ""


load_css()


# --------------------------------------------------------------------------- #
# CACHED MODEL LOADER
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_detector() -> VehicleDetector:
    """Load (and cache) the YOLOv8 vehicle detector, once per session."""
    return VehicleDetector()


# --------------------------------------------------------------------------- #
# ANALYTICS CACHING (avoid recomputation across reruns / pages)
# --------------------------------------------------------------------------- #
def _results_signature(frame_results: list) -> str:
    """
    Build a deterministic, stable cache signature for a set of detection results.

    Uses stable session identifiers (video name, output path, frame count) plus
    a cheap fingerprint of the first/last frame counts instead of the Python
    object id, which is not reliable across reruns or reloaded state.
    """
    video_name = st.session_state.get("uploaded_video_name", "")
    output_path = st.session_state.get("output_video_path", "")
    frame_count = len(frame_results)
    first_count = sum(frame_results[0].counts_by_class.values()) if frame_count else 0
    last_count = sum(frame_results[-1].counts_by_class.values()) if frame_count else 0
    return f"{video_name}|{output_path}|{frame_count}|{first_count}|{last_count}"


def get_cached_analytics() -> Optional[Tuple[TrafficAnalytics, Any, pd.DataFrame]]:
    """
    Compute (or reuse) TrafficAnalytics, summary and dataframe for the current
    detection results. Recomputes only when the underlying results change,
    based on a deterministic signature rather than object identity.
    """
    frame_results = st.session_state.get("detection_results")
    if not frame_results:
        return None

    cache_key = _results_signature(frame_results)
    if st.session_state.get("_analytics_cache_key") != cache_key:
        try:
            analytics = TrafficAnalytics(frame_results)
            summary = analytics.compute_summary()
            df = analytics.to_dataframe()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Analytics computation failed")
            st.error(f"❌ Failed to compute analytics: {exc}")
            return None

        st.session_state._analytics_cache_key = cache_key
        st.session_state._analytics_cache = (analytics, summary, df)
        st.session_state.analytics_summary = summary

    return st.session_state._analytics_cache


# --------------------------------------------------------------------------- #
# VIDEO / FORMAT HELPERS
# --------------------------------------------------------------------------- #
def get_video_metadata(path: Path) -> Dict[str, Any]:
    """Read basic metadata (resolution, fps, duration) from a video file."""
    meta: Dict[str, Any] = {"width": None, "height": None, "fps": None, "duration": None}
    cap = None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return meta
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        meta["width"] = width or None
        meta["height"] = height or None
        meta["fps"] = round(fps, 2) if fps else None
        meta["duration"] = (frame_count / fps) if fps and frame_count else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read video metadata for %s: %s", path, exc)
    finally:
        if cap is not None:
            cap.release()
    return meta


def format_duration(seconds: Optional[float]) -> str:
    """Format a duration in seconds as mm:ss, gracefully handling missing data."""
    if seconds is None:
        return "Unknown"
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def format_eta(seconds: Optional[float]) -> str:
    """Format an estimated-time-remaining value for display."""
    if seconds is None or seconds < 0:
        return "Calculating..."
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s"


# --------------------------------------------------------------------------- #
# REUSABLE UI COMPONENTS
# --------------------------------------------------------------------------- #
def metric_card(icon: str, label: str, value: str, sub: str = "", status: str = "primary") -> str:
    """Return HTML for a single glass-style metric card."""
    return f"""
    <div class="metric-card {status}">
        <span class="metric-icon">{icon}</span>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """


def render_metric_row(cards: List[dict]) -> None:
    """Render a row of metric cards given a list of kwargs dicts."""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            st.markdown(metric_card(**card), unsafe_allow_html=True)


def density_badge(level: str) -> str:
    """Return an HTML badge for a traffic density level."""
    css_class = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}.get(level, "badge-medium")
    label = config.DENSITY_LABELS.get(level, level)
    return f'<span class="badge {css_class}">{label}</span>'


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def info_card(icon: str, title: str, body: str) -> None:
    """Render a small glass-style info card with an icon, title and body text."""
    st.markdown(
        f"""
        <div class="glass-card">
            <b>{icon} {title}</b><br>
            <span style="color:var(--text-secondary); font-size:0.85rem;">{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def alert_error(message: str) -> None:
    st.error(f"❌ {message}")


def alert_warning(message: str) -> None:
    st.warning(f"⚠️ {message}")


def alert_success(message: str) -> None:
    st.success(f"✅ {message}")


def render_chart_safe(build_fig, warning_msg: str) -> None:
    """Render a Plotly chart, grouping the try/except so callers stay concise."""
    try:
        st.plotly_chart(build_fig(), use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s: %s", warning_msg, exc)
        alert_warning(warning_msg)


def safe_file_download_button(
    path_str: Optional[str],
    label: str,
    file_name: str,
    mime: str,
    missing_msg: Optional[str] = None,
    **kwargs: Any,
) -> bool:
    """
    Render a download button for a file on disk, handling the common
    "missing" / "unreadable" failure modes in one place.
    Returns True if the button was rendered successfully.
    """
    if not path_str or not Path(path_str).exists():
        if missing_msg:
            alert_warning(missing_msg)
        return False
    try:
        with open(path_str, "rb") as f:
            st.download_button(label, data=f, file_name=file_name, mime=mime, **kwargs)
        return True
    except OSError as exc:
        logger.warning("Could not open file %s: %s", path_str, exc)
        alert_warning(f"'{file_name}' could not be opened for download.")
        return False


def render_extended_statistics(df: pd.DataFrame) -> None:
    """
    Show additional derived statistics (FPS spread, density, detection rate,
    processing time) on the Analytics page — without modifying analytics.py.
    Skips any metric whose source data isn't available.
    """
    cards: List[dict] = []

    if "fps" in df.columns and not df["fps"].empty:
        cards.append({"icon": "⚡", "label": "Average FPS", "value": f"{df['fps'].mean():.1f}", "status": "primary"})
        cards.append({"icon": "🚀", "label": "Peak FPS", "value": f"{df['fps'].max():.1f}", "status": "success"})
        cards.append({"icon": "🐢", "label": "Minimum FPS", "value": f"{df['fps'].min():.1f}", "status": "warning"})

    vehicle_col = next((c for c in ("vehicle_count", "total_vehicles") if c in df.columns), None)
    if vehicle_col:
        detected_frames = int((df[vehicle_col] > 0).sum())
        detection_rate = (detected_frames / len(df) * 100) if len(df) else 0.0
        cards.append({"icon": "🎯", "label": "Detection Rate", "value": f"{detection_rate:.1f}%", "status": "danger"})

    cards.append({"icon": "🖼️", "label": "Total Frames", "value": str(len(df)), "status": "primary"})

    processing_time = st.session_state.get("processing_time")
    if processing_time is not None:
        cards.append({"icon": "⏳", "label": "Processing Time", "value": format_duration(processing_time), "status": "success"})

    if not cards:
        return

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📐 Extended Statistics", expanded=False):
        for i in range(0, len(cards), 4):
            render_metric_row(cards[i : i + 4])


def file_size_label(path_str: Optional[str]) -> str:
    """Return a human-readable file size for a path, or 'Unknown' if unavailable."""
    if not path_str:
        return "Unknown"
    try:
        size_bytes = Path(path_str).stat().st_size
    except OSError:
        return "Unknown"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


# --------------------------------------------------------------------------- #
# SIDEBAR
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    with st.sidebar:
        logo_b64 = image_to_base64(config.LOGO_PATH)
        if logo_b64:
            st.markdown(
                f'<div style="text-align:center; padding: 10px 0 4px 0;">'
                f'<img src="data:image/png;base64,{logo_b64}" width="70"/></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<h3 style="text-align:center; margin-top:0;">{config.APP_TITLE}</h3>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        selection = st.radio(
            "Navigation",
            config.NAV_ITEMS,
            index=config.NAV_ITEMS.index(st.session_state.page),
            label_visibility="collapsed",
        )
        st.session_state.page = selection

        st.markdown("---")
        st.markdown("**📁 Recent Uploads**")
        history = st.session_state.get("upload_history", [])
        if history:
            for name in reversed(history[-5:]):
                st.caption(f"🎞️ {name}")
            if len(history) > 5:
                st.caption(f"…and {len(history) - 5} more")
        else:
            st.caption("No videos analyzed yet.")

        st.markdown("---")
        st.markdown("**🧠 Model Information**")
        st.caption("YOLOv8n · COCO pretrained weights")
        st.caption(f"Classes: {', '.join(config.VEHICLE_CLASSES)}")

        st.markdown("---")
        st.markdown("**🕒 Session Information**")
        st.caption(f"Session time: {timestamp_now()}")
        if st.session_state.get("processing_in_progress"):
            st.caption("Status: 🟡 Processing in progress")
        elif st.session_state.get("processed"):
            st.caption("Status: 🟢 Analysis ready")
        else:
            st.caption("Status: ⚪ Idle")


# --------------------------------------------------------------------------- #
# PAGE: HOME
# --------------------------------------------------------------------------- #
def page_home() -> None:
    banner_b64 = image_to_base64(config.BANNER_PATH)
    banner_style = (
        f"background-image: linear-gradient(120deg, rgba(15,42,92,0.88), rgba(47,129,247,0.75)), "
        f"url('data:image/jpg;base64,{banner_b64}'); background-size: cover; background-position:center;"
        if banner_b64
        else "background-image: linear-gradient(120deg, rgba(15,42,92,0.95), rgba(47,129,247,0.85));"
    )

    st.markdown(
        f"""
        <div class="hero-banner" style="{banner_style}">
            <div class="hero-badge">🚦 AI-Powered Computer Vision</div>
            <h1>{config.APP_TITLE}</h1>
            <p class="subtitle">{config.APP_SUBTITLE}</p>
            <p class="description">
                Upload traffic surveillance footage and let a pretrained YOLOv8 model automatically
                detect vehicles, calculate traffic density, and generate rich analytics — built for
                traffic authorities, researchers, and smart-city applications.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([2, 1])

    with col_left:
        section_title("📤 Quick Start")
        st.write(
            "Head to **🚗 Vehicle Detection** in the sidebar to upload a traffic video and "
            "run AI-powered analysis, or explore the sections below."
        )
        quick_cols = st.columns(3)
        with quick_cols[0]:
            info_card("1️⃣", "Upload", "MP4, AVI or MOV traffic footage")
        with quick_cols[1]:
            info_card("2️⃣", "Analyze", "YOLOv8 detects cars, bikes, buses & trucks")
        with quick_cols[2]:
            info_card("3️⃣", "Report", "Export CSV / PDF analytics reports")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Go to Vehicle Detection", use_container_width=False):
            st.session_state.page = config.NAV_DETECTION
            st.rerun()

    with col_right:
        st.markdown(
            f"""
            <div class="tips-card">
                <b>💡 Today's Tips</b>
                <ul>
                    <li>Supported formats: {', '.join(config.SUPPORTED_VIDEO_FORMATS).upper()}</li>
                    <li>Recommended resolution: {config.RECOMMENDED_RESOLUTION}</li>
                    <li>Detected classes: {', '.join(config.VEHICLE_CLASSES)}</li>
                    <li>Shorter clips (&lt;2 min) process fastest on CPU</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    section_title("🚘 Supported Vehicle Classes")
    render_metric_row(
        [
            {"icon": "🚗", "label": "Car", "value": "Detected", "status": "primary"},
            {"icon": "🏍️", "label": "Motorcycle", "value": "Detected", "status": "warning"},
            {"icon": "🚌", "label": "Bus", "value": "Detected", "status": "success"},
            {"icon": "🚚", "label": "Truck", "value": "Detected", "status": "danger"},
        ]
    )


# --------------------------------------------------------------------------- #
# PAGE: VEHICLE DETECTION
# --------------------------------------------------------------------------- #
def page_detection() -> None:
    section_title("🚗 Vehicle Detection")
    st.write("Upload a traffic surveillance video to detect and track vehicles using YOLOv8.")

    uploaded_file = st.file_uploader(
        "Upload traffic video",
        type=config.SUPPORTED_VIDEO_FORMATS,
        help=f"Supported formats: {', '.join(config.SUPPORTED_VIDEO_FORMATS).upper()} "
        f"(max {config.MAX_UPLOAD_SIZE_MB} MB)",
    )

    if uploaded_file is not None:
        try:
            video_path = save_uploaded_file(uploaded_file)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to save uploaded file")
            alert_error(f"Could not save the uploaded file: {exc}")
            return

        if video_path is None:
            alert_error("The uploaded file could not be saved. Please try again.")
            return

        video_path = Path(video_path)

        if not validate_video_file(video_path):
            alert_error("This file doesn't look like a supported/valid video. Please upload a different file.")
            return

        # New upload -> clear any stale downstream artifacts (PDF, old results tied to a diff video)
        if st.session_state.get("uploaded_video_name") != uploaded_file.name:
            st.session_state.pdf_report_path = None

        st.session_state.uploaded_video_path = str(video_path)
        st.session_state.uploaded_video_name = uploaded_file.name

        meta = get_video_metadata(video_path)

        col_preview, col_info = st.columns([2, 1])
        with col_preview:
            st.video(str(video_path))
        with col_info:
            resolution = (
                f"{meta['width']}×{meta['height']}" if meta["width"] and meta["height"] else "Unknown"
            )
            st.markdown(
                f"""
                <div class="glass-card">
                    <b>📄 File Info</b><br>
                    <span style="color:var(--text-secondary); font-size:0.85rem;">
                    Name: {uploaded_file.name}<br>
                    Size: {uploaded_file.size / (1024 * 1024):.2f} MB<br>
                    Resolution: {resolution}<br>
                    Duration: {format_duration(meta["duration"])}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            start_clicked = st.button(
                "▶️ Start Analysis",
                use_container_width=True,
                disabled=st.session_state.get("processing_in_progress", False),
            )

        if start_clicked:
            reset_analysis_state()
            st.session_state.processing_in_progress = True
            run_detection_pipeline(video_path, uploaded_file.name)

    elif st.session_state.get("uploaded_video_path"):
        st.info("A previous video is loaded. Upload a new file to replace it, or check Analytics/Reports.")

    # Show results if a run has already completed in this session
    if st.session_state.get("processed") and st.session_state.get("detection_results") is not None:
        render_live_results_summary()


def run_detection_pipeline(video_path: Path, video_name: str) -> None:
    """Run YOLOv8 detection across the uploaded video with a live dashboard."""
    try:
        with st.spinner("Loading YOLOv8 model (pretrained on COCO)..."):
            detector = get_detector()
    except ModelLoadError as exc:
        alert_error(str(exc))
        st.session_state.processing_in_progress = False
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while loading the model")
        alert_error(f"Could not load the detection model: {exc}")
        st.session_state.processing_in_progress = False
        return

    output_path = config.OUTPUTS_DIR / f"processed_{video_path.stem}.mp4"

    section_title("📡 Live Processing Dashboard")
    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()
    dashboard_placeholder = st.empty()
    frame_placeholder = st.empty()
    class_metrics_placeholder = st.empty()

    frame_results: List[Any] = []
    start_time = time.time()

    try:
        for frame_result, annotated_frame, idx, total in detector.process_video(video_path, output_path):
            frame_results.append(frame_result)

            progress = (idx / total) if total else 0.0
            progress_bar.progress(min(progress, 1.0))

            elapsed = time.time() - start_time
            avg_fps = idx / elapsed if elapsed > 0 else 0.0
            avg_per_frame = elapsed / idx if idx else 0.0
            eta_seconds = avg_per_frame * max(total - idx, 0) if idx else None
            vehicles_this_frame = sum(frame_result.counts_by_class.values())

            # --- Live dashboard metrics (native st.metric, professional & compact) ---
            with dashboard_placeholder.container():
                row1 = st.columns(4)
                row1[0].metric("Progress", f"{progress * 100:.1f}%")
                row1[1].metric("Current Frame", f"{idx}", help=f"of {total} total frames")
                row1[2].metric("Total Frames", f"{total}")
                row1[3].metric("Vehicles (this frame)", f"{vehicles_this_frame}")

                row2 = st.columns(4)
                row2[0].metric("Current FPS", f"{frame_result.fps:.1f}")
                row2[1].metric("Average FPS", f"{avg_fps:.1f}")
                row2[2].metric("Elapsed Time", format_duration(elapsed))
                row2[3].metric("Est. Remaining", format_eta(eta_seconds))

            status_placeholder.caption(f"Status: 🟡 Processing frame {idx}/{total}…")

            # Show every 5th frame in the live preview to keep the UI responsive
            if idx % 5 == 0 or idx == total:
                frame_rgb = annotated_frame[:, :, ::-1]  # BGR -> RGB
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                with class_metrics_placeholder.container():
                    counts = frame_result.counts_by_class
                    render_metric_row(
                        [
                            {"icon": "🚗", "label": "Cars", "value": str(counts.get("Car", 0)), "status": "primary"},
                            {"icon": "🏍️", "label": "Motorcycles", "value": str(counts.get("Motorcycle", 0)), "status": "warning"},
                            {"icon": "🚌", "label": "Buses", "value": str(counts.get("Bus", 0)), "status": "success"},
                            {"icon": "🚚", "label": "Trucks", "value": str(counts.get("Truck", 0)), "status": "danger"},
                        ]
                    )
    except IOError as exc:
        alert_error(str(exc))
        st.session_state.processing_in_progress = False
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Detection pipeline failed")
        alert_error(f"An unexpected error occurred during processing: {exc}")
        st.session_state.processing_in_progress = False
        return

    if not frame_results:
        alert_warning("No frames could be processed from this video.")
        st.session_state.processing_in_progress = False
        return

    total_elapsed = time.time() - start_time
    status_placeholder.caption("Status: 🟢 Processing complete!")
    finalize_analysis_results(frame_results, output_path, video_name, total_elapsed)

    alert_success("Analysis complete! View full analytics in the **Traffic Analytics** tab.")


def finalize_analysis_results(
    frame_results: List[Any], output_path: Path, video_name: str, elapsed_seconds: float
) -> None:
    """
    Single source of truth for marking an analysis run complete.
    Keeps every related session-state key synchronized in one place:
    detection_results, output_video_path, processed, processing_in_progress,
    upload_history and processing_time.
    """
    st.session_state.detection_results = frame_results
    st.session_state.output_video_path = str(output_path)
    st.session_state.processed = True
    st.session_state.processing_in_progress = False
    st.session_state.processing_time = elapsed_seconds
    push_upload_history(video_name)

    # Pre-compute analytics once so downstream pages (Analytics/Reports) reuse it.
    get_cached_analytics()


def render_live_results_summary() -> None:
    """Render a quick summary + processed video preview/download right after analysis."""
    cached = get_cached_analytics()
    if cached is None:
        alert_warning("No analysis results available to summarize.")
        return
    _analytics, summary, _df = cached

    section_title("📊 Quick Summary")
    render_metric_row(
        [
            {
                "icon": "🚘", "label": "Total Vehicles", "value": str(summary.total_vehicles),
                "sub": f"across {summary.total_frames_analyzed} frames", "status": "primary",
            },
            {
                "icon": "⭐", "label": "Most Detected", "value": summary.most_detected_vehicle,
                "status": "success",
            },
            {
                "icon": "📈", "label": "Avg / Frame", "value": str(summary.avg_vehicles_per_frame),
                "status": "warning",
            },
            {
                "icon": "🔥", "label": "Peak Count", "value": str(summary.peak_vehicle_count),
                "status": "danger",
            },
        ]
    )
    render_metric_row(
        [
            {
                "icon": "⏱️", "label": "Peak Timestamp", "value": f"{summary.peak_timestamp_sec:.1f}s",
                "sub": f"frame {summary.peak_frame}", "status": "primary",
            },
            {
                "icon": "🚦", "label": "Traffic Density", "value": summary.density_level,
                "status": "warning",
            },
            {
                "icon": "🧭", "label": "Congestion Level", "value": config.DENSITY_LABELS.get(
                    summary.density_level, summary.density_level
                ),
                "status": "danger",
            },
        ]
    )

    output_path_str = st.session_state.get("output_video_path")
    if output_path_str and Path(output_path_str).exists():
        st.markdown("<br>", unsafe_allow_html=True)
        section_title("🎬 Processed Video")
        col_video, col_download = st.columns([2, 1])
        with col_video:
            st.video(output_path_str)
        with col_download:
            safe_file_download_button(
                output_path_str,
                "⬇️ Download Processed Video",
                Path(output_path_str).name,
                "video/mp4",
                use_container_width=True,
            )
    elif output_path_str:
        alert_warning("The processed video file is missing from disk.")


# --------------------------------------------------------------------------- #
# PAGE: TRAFFIC ANALYTICS
# --------------------------------------------------------------------------- #
def page_analytics() -> None:
    section_title("📊 Traffic Analytics")

    if not st.session_state.get("processed"):
        st.info("⚠️ No analysis results yet. Please run **Vehicle Detection** first.")
        return

    cached = get_cached_analytics()
    if cached is None:
        alert_warning("Analysis results could not be loaded. Please re-run detection.")
        return
    analytics, summary, df = cached

    # --- Statistics ---
    section_title("📈 Detection Statistics")
    render_metric_row(
        [
            {"icon": "🚗", "label": "Total Cars", "value": str(summary.counts_by_class.get("Car", 0)), "status": "primary"},
            {"icon": "🏍️", "label": "Total Motorcycles", "value": str(summary.counts_by_class.get("Motorcycle", 0)), "status": "warning"},
            {"icon": "🚌", "label": "Total Buses", "value": str(summary.counts_by_class.get("Bus", 0)), "status": "success"},
            {"icon": "🚚", "label": "Total Trucks", "value": str(summary.counts_by_class.get("Truck", 0)), "status": "danger"},
        ]
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        render_chart_safe(
            lambda: analytics.vehicle_distribution_pie(summary), "Could not render the vehicle distribution chart."
        )
    with col_b:
        render_chart_safe(
            lambda: analytics.vehicle_count_bar(summary), "Could not render the vehicle count chart."
        )

    render_chart_safe(analytics.vehicle_trend_over_time, "Could not render the trend chart.")

    col_c, col_d = st.columns([1.4, 1])
    with col_c:
        render_chart_safe(analytics.traffic_density_timeline, "Could not render the density timeline.")
    with col_d:
        render_chart_safe(
            lambda: analytics.congestion_gauge(summary), "Could not render the congestion gauge."
        )
        st.markdown(
            f"<div style='text-align:center;'>Congestion Level: {density_badge(summary.density_level)}</div>",
            unsafe_allow_html=True,
        )

    # --- Optional extended statistics (derived without touching analytics.py) ---
    render_extended_statistics(df)

    section_title("📝 Summary Panel")
    st.markdown(
        f"""
        <div class="glass-card">
        <ul>
            <li><b>Most common vehicle:</b> {summary.most_detected_vehicle}</li>
            <li><b>Peak traffic moment:</b> {summary.peak_timestamp_sec:.1f}s (frame {summary.peak_frame}), {summary.peak_vehicle_count} vehicles</li>
            <li><b>Total detections:</b> {summary.total_vehicles}</li>
            <li><b>Average vehicles/frame:</b> {summary.avg_vehicles_per_frame}</li>
            <li><b>Traffic density:</b> {summary.density_level}</li>
            <li><b>Congestion level:</b> {density_badge(summary.density_level)}</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔍 View Raw Per-Frame Data"):
        st.dataframe(df, use_container_width=True)


# --------------------------------------------------------------------------- #
# PAGE: REPORTS
# --------------------------------------------------------------------------- #
def page_reports() -> None:
    section_title("📄 Reports & Downloads")

    if not st.session_state.get("processed"):
        st.info("⚠️ No analysis results yet. Please run **Vehicle Detection** first.")
        return

    cached = get_cached_analytics()
    if cached is None:
        alert_warning("Analysis results could not be loaded. Please re-run detection.")
        return
    analytics, summary, df = cached
    video_name = st.session_state.get("uploaded_video_name", "video")
    output_path_str = st.session_state.get("output_video_path")
    pdf_path_str = st.session_state.get("pdf_report_path")

    # --- Report readiness status cards ---
    render_metric_row(
        [
            {"icon": "📑", "label": "CSV Reports", "value": "Ready", "status": "success"},
            {
                "icon": "🧾", "label": "PDF Report",
                "value": "Ready" if pdf_path_str and Path(pdf_path_str).exists() else "Not generated",
                "status": "success" if pdf_path_str and Path(pdf_path_str).exists() else "warning",
            },
            {
                "icon": "🎬", "label": "Processed Video",
                "value": "Ready" if output_path_str and Path(output_path_str).exists() else "Unavailable",
                "status": "success" if output_path_str and Path(output_path_str).exists() else "danger",
            },
        ]
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        info_card("📑", "Vehicle Statistics CSV", "Per-frame detection data")
        try:
            gen_start = time.time()
            csv_bytes = generate_vehicle_statistics_csv(df)
            gen_time = time.time() - gen_start
            st.download_button(
                "⬇️ Download CSV", data=csv_bytes, file_name="vehicle_statistics.csv", mime="text/csv",
                use_container_width=True,
            )
            st.caption(f"✅ {len(csv_bytes) / 1024:.1f} KB · generated in {gen_time:.2f}s")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Vehicle statistics CSV generation failed")
            alert_error(f"Could not generate the vehicle statistics CSV: {exc}")

    with col2:
        info_card("📊", "Traffic Summary CSV", "Aggregated summary metrics")
        try:
            gen_start = time.time()
            summary_csv_bytes = generate_traffic_summary_csv(summary)
            gen_time = time.time() - gen_start
            st.download_button(
                "⬇️ Download CSV", data=summary_csv_bytes, file_name="traffic_summary.csv", mime="text/csv",
                use_container_width=True,
            )
            st.caption(f"✅ {len(summary_csv_bytes) / 1024:.1f} KB · generated in {gen_time:.2f}s")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Traffic summary CSV generation failed")
            alert_error(f"Could not generate the traffic summary CSV: {exc}")

    with col3:
        info_card("🧾", "Full PDF Report", "Formatted report with charts")
        if st.button("🛠️ Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                gen_start = time.time()
                chart_images = {}
                try:
                    # Requires the 'kaleido' package to export Plotly figures as images.
                    chart_images["Vehicle Distribution"] = analytics.vehicle_distribution_pie(summary).to_image(format="png")
                    chart_images["Vehicle Count by Class"] = analytics.vehicle_count_bar(summary).to_image(format="png")
                    chart_images["Vehicle Trend Over Time"] = analytics.vehicle_trend_over_time().to_image(format="png")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Chart image export unavailable: %s", exc)
                    alert_warning(
                        "Charts could not be embedded (install the `kaleido` package for chart images). "
                        "Generating a text-only report instead."
                    )

                try:
                    pdf_path = generate_pdf_report(summary, video_name, chart_images=chart_images or None)
                    st.session_state["pdf_report_path"] = str(pdf_path)
                    pdf_path_str = str(pdf_path)
                    gen_time = time.time() - gen_start
                    alert_success(
                        f"PDF report generated · {file_size_label(pdf_path_str)} · {gen_time:.2f}s"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("PDF report generation failed")
                    alert_error(f"Could not generate the PDF report: {exc}")
                    st.session_state["pdf_report_path"] = None
                    pdf_path_str = None

        safe_file_download_button(
            pdf_path_str, "⬇️ Download PDF", "traffic_report.pdf", "application/pdf",
            use_container_width=True,
        )

    if output_path_str and Path(output_path_str).exists():
        section_title("🎬 Processed Video")
        st.caption(f"File size: {file_size_label(output_path_str)}")
        safe_file_download_button(
            output_path_str, "⬇️ Download Processed Video", Path(output_path_str).name, "video/mp4",
        )


# --------------------------------------------------------------------------- #
# PAGE: ABOUT
# --------------------------------------------------------------------------- #
def page_about() -> None:
    section_title("ℹ️ About This Project")

    st.markdown(
        """
        <div class="glass-card">
        <p>The <b>Intelligent Traffic Monitoring System</b> is an AI-powered computer vision
        application that analyzes traffic surveillance footage to automatically detect vehicles,
        compute traffic density, and generate actionable analytics — built to help traffic
        authorities and researchers monitor urban congestion more effectively.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("🛠️ Technologies Used")
    tech_cols = st.columns(4)
    techs = [
        ("🧠", "YOLOv8", "Pretrained object detection (Ultralytics)"),
        ("📷", "OpenCV", "Video I/O & frame processing"),
        ("🗂️", "COCO Dataset", "Source of pretrained weights"),
        ("🖥️", "Streamlit", "Interactive web dashboard"),
    ]
    for col, (icon, name, desc) in zip(tech_cols, techs):
        with col:
            st.markdown(
                f'<div class="glass-card" style="text-align:center;">'
                f'<div style="font-size:1.8rem;">{icon}</div><b>{name}</b><br>'
                f'<span style="color:var(--text-secondary); font-size:0.8rem;">{desc}</span></div>',
                unsafe_allow_html=True,
            )

    section_title("✨ Key Features")
    feature_cols = st.columns(3)
    features = [
        ("🎯", "Real-Time Detection", "Frame-by-frame vehicle detection with live progress"),
        ("📊", "Rich Analytics", "Interactive charts, gauges and density timelines"),
        ("📄", "Exportable Reports", "One-click CSV and PDF report generation"),
    ]
    for col, (icon, name, desc) in zip(feature_cols, features):
        with col:
            info_card(icon, name, desc)

    section_title("🚀 Future Improvements")
    st.markdown(
        """
        <div class="glass-card">
        <ul>
            <li><b>DeepSORT / ByteTrack</b> — multi-object tracking for unique vehicle IDs</li>
            <li><b>Vehicle Speed Detection</b> — estimate real-world speed from tracked trajectories</li>
            <li><b>Accident Detection</b> — flag anomalous collision events automatically</li>
            <li><b>Heatmaps</b> — visualize congestion hotspots over time</li>
            <li><b>Lane Counting</b> — per-lane vehicle counts and flow direction</li>
            <li><b>Emergency Vehicle Detection</b> — prioritize ambulances/fire trucks</li>
            <li><b>Live CCTV Feed Integration</b> — real-time RTSP/IP camera streaming</li>
            <li><b>Cloud Deployment</b> — scalable inference via cloud GPU infrastructure</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="app-footer">Built with Streamlit, YOLOv8 & OpenCV · '
        'Intelligent Traffic Monitoring System</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# ROUTER
# --------------------------------------------------------------------------- #
def main() -> None:
    render_sidebar()

    page = st.session_state.page
    try:
        if page == config.NAV_HOME:
            page_home()
        elif page == config.NAV_DETECTION:
            page_detection()
        elif page == config.NAV_ANALYTICS:
            page_analytics()
        elif page == config.NAV_REPORTS:
            page_reports()
        elif page == config.NAV_ABOUT:
            page_about()
        else:
            alert_warning("Unknown page selected. Redirecting to Home.")
            st.session_state.page = config.NAV_HOME
            st.rerun()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error while rendering page '%s'", page)
        alert_error(f"Something went wrong while rendering this page: {exc}")
        st.caption("Please try again, or navigate to a different page from the sidebar.")


if __name__ == "__main__":
    main()
