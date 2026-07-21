"""
utils.py
========
General-purpose helper functions shared across the application:
file validation, formatting helpers, session-state helpers, and
lightweight logging wrappers.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st

import config

# --------------------------------------------------------------------------- #
# LOGGING
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("traffic-monitor")


# --------------------------------------------------------------------------- #
# FILE HANDLING
# --------------------------------------------------------------------------- #
def is_supported_video(filename: str) -> bool:
    """Return True if the given filename has a supported video extension."""
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext in config.SUPPORTED_VIDEO_FORMATS


def save_uploaded_file(uploaded_file) -> Path | None:
    """
    Persist a Streamlit UploadedFile object to a temporary file on disk and
    return its path. Returns None (and shows an error) if the file is
    missing, empty, or of an unsupported type.
    """
    if uploaded_file is None:
        st.error("⚠️ No file was uploaded. Please select a video first.")
        return None

    if not is_supported_video(uploaded_file.name):
        st.error(
            f"⚠️ Unsupported file type `.{Path(uploaded_file.name).suffix}`. "
            f"Supported formats: {', '.join(config.SUPPORTED_VIDEO_FORMATS).upper()}"
        )
        return None

    if uploaded_file.size == 0:
        st.error("⚠️ The uploaded file is empty. Please choose a valid video.")
        return None

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_SIZE_MB:
        st.error(
            f"⚠️ File is too large ({size_mb:.1f} MB). "
            f"Maximum allowed size is {config.MAX_UPLOAD_SIZE_MB} MB."
        )
        return None

    try:
        suffix = Path(uploaded_file.name).suffix
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_file.write(uploaded_file.read())
        tmp_file.close()
        logger.info("Saved uploaded file to %s", tmp_file.name)
        return Path(tmp_file.name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to save uploaded file")
        st.error(f"⚠️ Could not save the uploaded video: {exc}")
        return None


def validate_video_file(video_path: Path) -> bool:
    """
    Quick sanity check that a video file can actually be opened and contains
    at least one readable frame. Returns True if valid, otherwise shows an
    error message and returns False.
    """
    import cv2  # local import keeps module import time low

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        st.error("⚠️ The video file appears to be corrupted or unreadable.")
        cap.release()
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, _ = cap.read()
    cap.release()

    if not ret or frame_count <= 0:
        st.error("⚠️ The video contains no readable frames.")
        return False

    return True


# --------------------------------------------------------------------------- #
# FORMATTING HELPERS
# --------------------------------------------------------------------------- #
def format_number(value: int | float) -> str:
    """Format a number with thousands separators."""
    try:
        return f"{value:,.0f}"
    except (ValueError, TypeError):
        return str(value)


def format_percentage(value: float) -> str:
    """Format a float as a percentage string, e.g. 42.3%."""
    return f"{value:.1f}%"


def format_duration(seconds: float) -> str:
    """Format seconds as mm:ss."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def timestamp_now() -> str:
    """Return a human-readable current timestamp."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def filename_safe_timestamp() -> str:
    """Return a filesystem-safe timestamp for naming output files."""
    return time.strftime("%Y%m%d_%H%M%S")


# --------------------------------------------------------------------------- #
# STREAMLIT SESSION-STATE HELPERS
# --------------------------------------------------------------------------- #
def init_session_state() -> None:
    """Initialise every session-state key the app depends on, once."""
    defaults: dict[str, Any] = {
        "page": config.NAV_HOME,
        "processed": False,
        "processing_in_progress": False,
        "detection_results": None,
        "analytics_summary": None,
        "output_video_path": None,
        "uploaded_video_path": None,
        "uploaded_video_name": None,
        "upload_history": [],
        "theme": "dark",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis_state() -> None:
    """Clear results from a previous analysis run before starting a new one."""
    st.session_state.processed = False
    st.session_state.processing_in_progress = False
    st.session_state.detection_results = None
    st.session_state.analytics_summary = None
    st.session_state.output_video_path = None


def push_upload_history(name: str) -> None:
    """Track recently uploaded/analyzed video names (most recent first)."""
    history: list[str] = st.session_state.get("upload_history", [])
    if name in history:
        history.remove(name)
    history.insert(0, name)
    st.session_state.upload_history = history[:5]


# --------------------------------------------------------------------------- #
# MISC
# --------------------------------------------------------------------------- #
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers, returning `default` instead of raising on /0."""
    return numerator / denominator if denominator else default
