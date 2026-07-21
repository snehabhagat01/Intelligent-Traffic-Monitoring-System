# 🚦 Intelligent Traffic Monitoring System

An AI-powered computer vision dashboard that analyzes traffic surveillance footage to automatically **detect vehicles, compute traffic density, and generate interactive analytics** — built with YOLOv8, OpenCV, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Project Description

Urban traffic congestion is difficult to monitor manually. This project uses a **pretrained YOLOv8 object detection model** to analyze uploaded traffic videos frame-by-frame, detect vehicles (cars, motorcycles, buses, trucks), and turn raw detections into meaningful traffic analytics — density classification, congestion levels, trend charts, and exportable CSV/PDF reports.

> **Note:** This project performs **inference only**. The YOLOv8n model ships pretrained on the **COCO dataset** (80 object classes) by Ultralytics — no custom training is performed. We simply filter COCO's output down to the four vehicle classes we care about (`car`, `motorcycle`, `bus`, `truck`).

---

## ✨ Features

- 🎯 **Real-time-style vehicle detection** using YOLOv8 (car, motorcycle, bus, truck)
- 📡 **Live processing dashboard** — frame-by-frame preview, FPS, progress bar, live counts
- 📊 **Interactive Plotly analytics** — pie charts, bar charts, trend lines, density timeline, congestion gauge
- 🚦 **Traffic density & congestion classification** (Low / Medium / High) with color-coded badges
- 📄 **Exportable reports** — `vehicle_statistics.csv`, `traffic_summary.csv`, and a formatted `traffic_report.pdf`
- 🎬 **Downloadable processed video** with bounding boxes and HUD overlay
- 🎨 **Modern dark-navy dashboard UI** — glassmorphism cards, gradient hero banner, custom CSS theme
- 🧭 **Multi-page navigation** — Home, Vehicle Detection, Traffic Analytics, Reports, About
- 🛡️ **Robust error handling** for unsupported files, corrupted videos, and model load failures

---

## 🖼️ Screenshots

> _Add screenshots after running the app locally:_

| Home Dashboard | Live Detection | Analytics |
|---|---|---|
| `screenshots/home.png` | `screenshots/detection.png` | `screenshots/analytics.png` |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| UI / Dashboard | Streamlit, custom CSS |
| Object Detection | YOLOv8 (Ultralytics), pretrained on COCO |
| Computer Vision | OpenCV |
| Data Processing | NumPy, Pandas |
| Visualization | Plotly, Matplotlib |
| Reporting | ReportLab (PDF), CSV export |
| Imaging | Pillow |

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Traffic-Monitoring-System.git
cd Traffic-Monitoring-System
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> The YOLOv8n weights (`yolov8n.pt`) are downloaded automatically by Ultralytics on first run and cached in the `models/` directory. An internet connection is required the first time you launch the app.

---

## ▶️ How to Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal (typically `http://localhost:8501`).

**Usage flow:**
1. Go to **🚗 Vehicle Detection** in the sidebar.
2. Upload a traffic video (`.mp4`, `.avi`, or `.mov`).
3. Click **▶️ Start Analysis** to run YOLOv8 detection with a live dashboard.
4. View full charts and statistics under **📊 Traffic Analytics**.
5. Export CSV/PDF reports and the processed video under **📄 Reports**.

---

## 📁 Folder Structure

```
Traffic-Monitoring-System/
│
├── app.py                  # Main Streamlit application (UI + routing)
├── detector.py              # YOLOv8 vehicle detection wrapper
├── analytics.py             # Traffic statistics, density rules, Plotly charts
├── report_generator.py      # CSV and PDF report generation (ReportLab)
├── utils.py                  # File validation, formatting, session-state helpers
├── config.py                  # Centralized constants, paths, thresholds, theme
│
├── assets/
│   ├── logo.png              # App logo
│   ├── banner.jpg             # Hero banner background
│   └── style.css               # Custom dashboard theme (dark navy + blue accents)
│
├── models/
│   └── yolov8n.pt              # Pretrained YOLOv8 weights (auto-downloaded)
│
├── videos/
│   ├── sample.mp4                # (optional) sample traffic video for testing
│   └── outputs/                   # Processed/annotated videos are saved here
│
├── reports/                        # Generated CSV/PDF reports are saved here
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 Model Details

- **Model:** YOLOv8n (nano) from [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Weights:** Pretrained on the **COCO dataset** (80 classes) — no fine-tuning performed
- **Classes used:** `car` (2), `motorcycle` (3), `bus` (5), `truck` (7) — filtered from COCO's full class list
- **Inference only:** the application performs prediction using `model.predict(...)`; training scripts are intentionally not included

---

## 🚧 Traffic Density Rules

| Vehicles per Frame | Density Level | Indicator |
|---|---|---|
| 0 – 10 | Low | 🟢 Green |
| 11 – 25 | Medium | 🟠 Orange |
| 25+ | High | 🔴 Red |

---

## 🔭 Future Scope

The following advanced features are **planned but not implemented** in this version:

- **DeepSORT / ByteTrack** — multi-object tracking for persistent vehicle IDs
- **Vehicle Speed Detection** — estimate real-world speed from tracked trajectories
- **Accident Detection** — flag anomalous collision events
- **Heatmaps** — visualize congestion hotspots over time
- **Lane Counting** — per-lane vehicle counts and flow direction
- **Emergency Vehicle Detection** — prioritize ambulances/fire trucks in traffic
- **Live CCTV Feed Integration** — real-time RTSP/IP camera streaming
- **Cloud Deployment** — scalable GPU-backed inference in the cloud

---

## ⚠️ Error Handling

The app gracefully handles:
- Unsupported file types and empty/missing uploads
- Corrupted or unreadable video files
- YOLO model load failures (missing dependency, download errors)
- Empty videos with no readable frames

---

## 📄 License

This project is released under the [MIT License](LICENSE). Free to use for learning, portfolio, and research purposes.

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the pretrained detection model
- [COCO Dataset](https://cocodataset.org/) for the underlying training data used by the pretrained weights
- [Streamlit](https://streamlit.io/) for the rapid dashboard framework
