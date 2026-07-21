"""
config.py
=========
Centralized configuration for the Intelligent Traffic Monitoring System.

This module stores every constant, path, threshold, and style value used
across the application so that behaviour can be tuned from a single place
without touching business logic elsewhere.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# BASE PATHS
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent
ASSETS_DIR: Path = BASE_DIR / "assets"
MODELS_DIR: Path = BASE_DIR / "models"
VIDEOS_DIR: Path = BASE_DIR / "videos"
OUTPUTS_DIR: Path = VIDEOS_DIR / "outputs"
REPORTS_DIR: Path = BASE_DIR / "reports"

STYLE_CSS_PATH: Path = ASSETS_DIR / "style.css"
LOGO_PATH: Path = ASSETS_DIR / "logo.png"
BANNER_PATH: Path = ASSETS_DIR / "banner.jpg"

# Ensure runtime directories always exist (models/videos/reports are created
# on first run if a fresh clone does not already contain them).
for _dir in (MODELS_DIR, VIDEOS_DIR, OUTPUTS_DIR, REPORTS_DIR, ASSETS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# MODEL CONFIGURATION
# --------------------------------------------------------------------------- #
# Pretrained YOLOv8 nano checkpoint (COCO weights). Ultralytics will
# automatically download this file the first time it is requested if it is
# not already present in MODELS_DIR.
YOLO_MODEL_NAME: str = "yolov8n.pt"
YOLO_MODEL_PATH: Path = MODELS_DIR / YOLO_MODEL_NAME

# Confidence / IoU thresholds used during inference.
CONFIDENCE_THRESHOLD: float = 0.35
IOU_THRESHOLD: float = 0.45

# COCO class IDs that correspond to road vehicles.
# (Full COCO index: 0=person ... 2=car, 3=motorcycle, 5=bus, 7=truck)
VEHICLE_CLASS_MAP: dict[int, str] = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}
VEHICLE_CLASSES: list[str] = list(VEHICLE_CLASS_MAP.values())

# Bounding-box colors (BGR, for OpenCV) per vehicle class.
BOX_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "Car": (66, 135, 245),        # blue
    "Motorcycle": (245, 176, 66), # orange
    "Bus": (66, 245, 156),        # green
    "Truck": (203, 66, 245),      # purple
}

# --------------------------------------------------------------------------- #
# TRAFFIC DENSITY RULES
# --------------------------------------------------------------------------- #
DENSITY_LOW_MAX: int = 10          # 0 - 10 vehicles  -> Low
DENSITY_MEDIUM_MAX: int = 25       # 11 - 25 vehicles -> Medium
# anything above DENSITY_MEDIUM_MAX -> High

DENSITY_LABELS: dict[str, str] = {
    "Low": "🟢 Low",
    "Medium": "🟠 Medium",
    "High": "🔴 High",
}

DENSITY_COLORS: dict[str, str] = {
    "Low": "#2ecc71",
    "Medium": "#f39c12",
    "High": "#e74c3c",
}

# --------------------------------------------------------------------------- #
# UPLOAD / VIDEO SETTINGS
# --------------------------------------------------------------------------- #
SUPPORTED_VIDEO_FORMATS: list[str] = ["mp4", "avi", "mov"]
MAX_UPLOAD_SIZE_MB: int = 500
RECOMMENDED_RESOLUTION: str = "1280x720 (HD) or lower, for best processing speed"

# Skip factor: process every Nth frame during preview / analysis to keep the
# UI responsive on CPU-only environments while still sampling the video well.
FRAME_SKIP: int = 1

# --------------------------------------------------------------------------- #
# THEME / UI SETTINGS
# --------------------------------------------------------------------------- #
APP_TITLE: str = "Intelligent Traffic Monitoring System"
APP_SUBTITLE: str = "AI-Powered Vehicle Detection & Traffic Analytics Platform"
APP_ICON: str = "🚦"

THEME_COLORS: dict[str, str] = {
    "background": "#0b1220",
    "surface": "#111a2b",
    "surface_light": "#16213a",
    "primary": "#2f81f7",
    "primary_light": "#5aa4ff",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "text_primary": "#e6edf7",
    "text_secondary": "#8ea0c0",
    "border": "#22304a",
}

# --------------------------------------------------------------------------- #
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------- #
NAV_HOME = "🏠 Home"
NAV_DETECTION = "🚗 Vehicle Detection"
NAV_ANALYTICS = "📊 Traffic Analytics"
NAV_REPORTS = "📄 Reports"
NAV_ABOUT = "ℹ️ About Project"

NAV_ITEMS: list[str] = [NAV_HOME, NAV_DETECTION, NAV_ANALYTICS, NAV_REPORTS, NAV_ABOUT]
