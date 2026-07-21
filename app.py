"""
app.py
======
Main Streamlit entry point for the Intelligent Traffic Monitoring System.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import base64
from pathlib import Path

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
    if config.STYLE_CSS_PATH.exists():
        css = config.STYLE_CSS_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def image_to_base64(path: Path) -> str:
    """Encode a local image as base64 for embedding in HTML/CSS."""
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode()


load_css()


# --------------------------------------------------------------------------- #
# CACHED MODEL LOADER
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_detector() -> VehicleDetector:
    """Load (and cache) the YOLOv8 vehicle detector, once per session."""
    return VehicleDetector()


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


def render_metric_row(cards: list[dict]) -> None:
    """Render a row of metric cards given a list of kwargs dicts."""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            st.markdown(metric_card(**card), unsafe_allow_html=True)


def density_badge(level: str) -> str:
    """Return an HTML badge for a traffic density level."""
    css_class = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}[level]
    label = config.DENSITY_LABELS[level]
    return f'<span class="badge {css_class}">{label}</span>'


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


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
            for name in history:
                st.caption(f"🎞️ {name}")
        else:
            st.caption("No videos analyzed yet.")

        st.markdown("---")
        st.caption(f"Model: YOLOv8n · COCO pretrained")
        st.caption(f"Session time: {timestamp_now()}")


# --------------------------------------------------------------------------- #
# PAGE: HOME
# --------------------------------------------------------------------------- #
def page_home() -> None:
    banner_b64 = image_to_base64(config.BANNER_PATH)
    banner_style = (
        f"background-image: linear-gradient(120deg, rgba(15,42,92,0.88), rgba(47,129,247,0.75)), "
        f"url('data:image/jpg;base64,{banner_b64}'); background-size: cover; background-position:center;"
        if banner_b64
        else ""
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

    col_left, col_right = st.columns([2, 1])

    with col_left:
        section_title("📤 Quick Start")
        st.write(
            "Head to **🚗 Vehicle Detection** in the sidebar to upload a traffic video and "
            "run AI-powered analysis, or explore the sections below."
        )
        quick_cols = st.columns(3)
        with quick_cols[0]:
            st.markdown(
                '<div class="glass-card">'
                '<b>1️⃣ Upload</b><br><span style="color:var(--text-secondary); font-size:0.85rem;">'
                'MP4, AVI or MOV traffic footage</span></div>',
                unsafe_allow_html=True,
            )
        with quick_cols[1]:
            st.markdown(
                '<div class="glass-card">'
                '<b>2️⃣ Analyze</b><br><span style="color:var(--text-secondary); font-size:0.85rem;">'
                'YOLOv8 detects cars, bikes, buses & trucks</span></div>',
                unsafe_allow_html=True,
            )
        with quick_cols[2]:
            st.markdown(
                '<div class="glass-card">'
                '<b>3️⃣ Report</b><br><span style="color:var(--text-secondary); font-size:0.85rem;">'
                'Export CSV / PDF analytics reports</span></div>',
                unsafe_allow_html=True,
            )

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
        video_path = save_uploaded_file(uploaded_file)
        if video_path is None:
            return

        if not validate_video_file(video_path):
            return

        st.session_state.uploaded_video_path = str(video_path)
        st.session_state.uploaded_video_name = uploaded_file.name

        col_preview, col_info = st.columns([2, 1])
        with col_preview:
            st.video(str(video_path))
        with col_info:
            st.markdown(
                f"""
                <div class="glass-card">
                    <b>📄 File Info</b><br>
                    <span style="color:var(--text-secondary); font-size:0.85rem;">
                    Name: {uploaded_file.name}<br>
                    Size: {uploaded_file.size / (1024*1024):.2f} MB
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            start_clicked = st.button("▶️ Start Analysis", use_container_width=True)

        if start_clicked:
            reset_analysis_state()
            st.session_state.processing_in_progress = True
            run_detection_pipeline(Path(video_path), uploaded_file.name)

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
        st.error(f"❌ {exc}")
        st.session_state.processing_in_progress = False
        return

    output_path = config.OUTPUTS_DIR / f"processed_{video_path.stem}.mp4"

    section_title("📡 Live Processing Dashboard")
    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()
    frame_placeholder = st.empty()
    metrics_placeholder = st.empty()

    frame_results = []

    try:
        for frame_result, annotated_frame, idx, total in detector.process_video(video_path, output_path):
            frame_results.append(frame_result)

            progress = idx / total if total else 0.0
            progress_bar.progress(min(progress, 1.0))
            status_placeholder.markdown(
                f"**Status:** Processing frame {idx}/{total} "
                f"&nbsp;|&nbsp; **FPS:** {frame_result.fps:.1f}"
            )

            # Show every 5th frame in the live preview to keep the UI responsive
            if idx % 5 == 0 or idx == total:
                frame_rgb = annotated_frame[:, :, ::-1]  # BGR -> RGB
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                with metrics_placeholder.container():
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
        st.error(f"❌ {exc}")
        st.session_state.processing_in_progress = False
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Detection pipeline failed")
        st.error(f"❌ An unexpected error occurred during processing: {exc}")
        st.session_state.processing_in_progress = False
        return

    if not frame_results:
        st.warning("⚠️ No frames could be processed from this video.")
        st.session_state.processing_in_progress = False
        return

    status_placeholder.markdown("**Status:** ✅ Processing complete!")
    st.session_state.detection_results = frame_results
    st.session_state.output_video_path = str(output_path)
    st.session_state.processed = True
    st.session_state.processing_in_progress = False
    push_upload_history(video_name)

    st.success("✅ Analysis complete! View full analytics in the **Traffic Analytics** tab.")
    render_live_results_summary()


def render_live_results_summary() -> None:
    """Render a quick summary + processed video download right after analysis."""
    frame_results = st.session_state.detection_results
    analytics = TrafficAnalytics(frame_results)
    summary = analytics.compute_summary()
    st.session_state.analytics_summary = summary

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
                "sub": f"at {summary.peak_timestamp_sec:.1f}s", "status": "danger",
            },
        ]
    )

    st.markdown(
        f"**Traffic Density:** {density_badge(summary.density_level)}",
        unsafe_allow_html=True,
    )

    if st.session_state.get("output_video_path"):
        with open(st.session_state.output_video_path, "rb") as f:
            st.download_button(
                "⬇️ Download Processed Video",
                data=f,
                file_name=Path(st.session_state.output_video_path).name,
                mime="video/mp4",
            )


# --------------------------------------------------------------------------- #
# PAGE: TRAFFIC ANALYTICS
# --------------------------------------------------------------------------- #
def page_analytics() -> None:
    section_title("📊 Traffic Analytics")

    if not st.session_state.get("processed"):
        st.info("⚠️ No analysis results yet. Please run **Vehicle Detection** first.")
        return

    frame_results = st.session_state.detection_results
    analytics = TrafficAnalytics(frame_results)
    summary = analytics.compute_summary()
    df = analytics.to_dataframe()

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

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(analytics.vehicle_distribution_pie(summary), use_container_width=True)
    with col_b:
        st.plotly_chart(analytics.vehicle_count_bar(summary), use_container_width=True)

    st.plotly_chart(analytics.vehicle_trend_over_time(), use_container_width=True)

    col_c, col_d = st.columns([1.4, 1])
    with col_c:
        st.plotly_chart(analytics.traffic_density_timeline(), use_container_width=True)
    with col_d:
        st.plotly_chart(analytics.congestion_gauge(summary), use_container_width=True)
        st.markdown(
            f"<div style='text-align:center;'>Congestion Level: {density_badge(summary.density_level)}</div>",
            unsafe_allow_html=True,
        )

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

    frame_results = st.session_state.detection_results
    analytics = TrafficAnalytics(frame_results)
    summary = analytics.compute_summary()
    df = analytics.to_dataframe()
    video_name = st.session_state.get("uploaded_video_name", "video")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="glass-card"><b>📑 Vehicle Statistics CSV</b><br>'
                    '<span style="color:var(--text-secondary); font-size:0.85rem;">Per-frame detection data</span></div>',
                    unsafe_allow_html=True)
        csv_bytes = generate_vehicle_statistics_csv(df)
        st.download_button(
            "⬇️ Download CSV", data=csv_bytes, file_name="vehicle_statistics.csv", mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.markdown('<div class="glass-card"><b>📊 Traffic Summary CSV</b><br>'
                    '<span style="color:var(--text-secondary); font-size:0.85rem;">Aggregated summary metrics</span></div>',
                    unsafe_allow_html=True)
        summary_csv_bytes = generate_traffic_summary_csv(summary)
        st.download_button(
            "⬇️ Download CSV", data=summary_csv_bytes, file_name="traffic_summary.csv", mime="text/csv",
            use_container_width=True,
        )

    with col3:
        st.markdown('<div class="glass-card"><b>🧾 Full PDF Report</b><br>'
                    '<span style="color:var(--text-secondary); font-size:0.85rem;">Formatted report with charts</span></div>',
                    unsafe_allow_html=True)
        if st.button("🛠️ Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                chart_images = {}
                try:
                    # Requires the 'kaleido' package to export Plotly figures as images.
                    chart_images["Vehicle Distribution"] = analytics.vehicle_distribution_pie(summary).to_image(format="png")
                    chart_images["Vehicle Count by Class"] = analytics.vehicle_count_bar(summary).to_image(format="png")
                    chart_images["Vehicle Trend Over Time"] = analytics.vehicle_trend_over_time().to_image(format="png")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Chart image export unavailable: %s", exc)
                    st.warning(
                        "⚠️ Charts could not be embedded (install the `kaleido` package for chart images). "
                        "Generating a text-only report instead."
                    )

                pdf_path = generate_pdf_report(summary, video_name, chart_images=chart_images or None)
                st.session_state["pdf_report_path"] = str(pdf_path)

        if st.session_state.get("pdf_report_path"):
            with open(st.session_state["pdf_report_path"], "rb") as f:
                st.download_button(
                    "⬇️ Download PDF", data=f, file_name="traffic_report.pdf", mime="application/pdf",
                    use_container_width=True,
                )

    if st.session_state.get("output_video_path"):
        section_title("🎬 Processed Video")
        with open(st.session_state.output_video_path, "rb") as f:
            st.download_button(
                "⬇️ Download Processed Video", data=f,
                file_name=Path(st.session_state.output_video_path).name, mime="video/mp4",
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


if __name__ == "__main__":
    main()
