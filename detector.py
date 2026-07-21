"""
detector.py
===========
Wraps the pretrained YOLOv8 (Ultralytics) model to perform vehicle
detection on video frames.

IMPORTANT: This module performs INFERENCE ONLY. The YOLOv8n checkpoint used
here is pretrained on the COCO dataset by Ultralytics — no training happens
in this project. We simply filter COCO's 80 classes down to the four
vehicle classes we care about (car, motorcycle, bus, truck).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

import config
from utils import logger


@dataclass
class Detection:
    """A single detected object within one frame."""
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class FrameResult:
    """Aggregated detection output for a single processed frame."""
    frame_number: int
    timestamp_sec: float
    detections: list[Detection] = field(default_factory=list)
    fps: float = 0.0

    @property
    def vehicle_count(self) -> int:
        return len(self.detections)

    @property
    def counts_by_class(self) -> dict[str, int]:
        counts = {cls: 0 for cls in config.VEHICLE_CLASSES}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts


class ModelLoadError(RuntimeError):
    """Raised when the YOLO model fails to load."""


class VehicleDetector:
    """
    Thin, reusable wrapper around a pretrained Ultralytics YOLOv8 model that
    filters detections down to road-vehicle classes and draws styled
    bounding boxes onto frames.
    """

    def __init__(
        self,
        model_path: Path = config.YOLO_MODEL_PATH,
        confidence: float = config.CONFIDENCE_THRESHOLD,
        iou: float = config.IOU_THRESHOLD,
    ) -> None:
        self.confidence = confidence
        self.iou = iou
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: Path):
        """
        Load the pretrained YOLOv8 model. Ultralytics automatically
        downloads the checkpoint (e.g. yolov8n.pt) on first use if it is
        not already present on disk, then caches it locally.
        """
        try:
            from ultralytics import YOLO  # imported lazily so the rest of
            # the app can still start even if ultralytics is briefly
            # unavailable (e.g. first-run dependency install).
        except ImportError as exc:
            raise ModelLoadError(
                "The 'ultralytics' package is not installed. "
                "Run `pip install -r requirements.txt` and try again."
            ) from exc

        try:
            # Passing the model *name* lets ultralytics auto-download it to
            # model_path's parent directory if it isn't already cached.
            weights = str(model_path) if model_path.exists() else config.YOLO_MODEL_NAME
            model = YOLO(weights)
            logger.info("YOLOv8 model loaded successfully (%s)", weights)
            return model
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"Failed to load YOLOv8 model: {exc}") from exc

    def detect_frame(self, frame: np.ndarray, frame_number: int, timestamp_sec: float) -> FrameResult:
        """Run inference on a single BGR frame and return a FrameResult."""
        start = time.perf_counter()

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            classes=list(config.VEHICLE_CLASS_MAP.keys()),
            verbose=False,
        )

        detections: list[Detection] = []
        if results:
            result = results[0]
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = config.VEHICLE_CLASS_MAP.get(cls_id)
                if class_name is None:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(Detection(class_name, conf, (x1, y1, x2, y2)))

        elapsed = time.perf_counter() - start
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        return FrameResult(
            frame_number=frame_number,
            timestamp_sec=timestamp_sec,
            detections=detections,
            fps=fps,
        )

    @staticmethod
    def draw_detections(frame: np.ndarray, frame_result: FrameResult) -> np.ndarray:
        """Draw bounding boxes, class labels, and confidence scores on a frame."""
        annotated = frame.copy()

        for det in frame_result.detections:
            x1, y1, x2, y2 = det.box
            color = config.BOX_COLORS_BGR.get(det.class_name, (255, 255, 255))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            label = f"{det.class_name} {det.confidence * 100:.0f}%"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - text_h - 8)),
                (x1 + text_w + 6, y1),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 3, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # HUD overlay: frame number, vehicle count, fps
        hud_text = (
            f"Frame: {frame_result.frame_number}  |  "
            f"Vehicles: {frame_result.vehicle_count}  |  "
            f"FPS: {frame_result.fps:.1f}"
        )
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (11, 18, 32), -1)
        cv2.putText(
            annotated, hud_text, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 237, 247), 1, cv2.LINE_AA,
        )
        return annotated

    def process_video(
        self, video_path: Path, output_path: Path, frame_skip: int = config.FRAME_SKIP
    ) -> Generator[tuple[FrameResult, np.ndarray, int, int], None, None]:
        """
        Generator that processes a video frame-by-frame, yielding
        (frame_result, annotated_frame, current_index, total_frames) tuples
        so the caller (Streamlit UI) can update a live dashboard while
        detection runs. The fully annotated video is written to output_path.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (width, height))

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                timestamp_sec = frame_idx / src_fps
                frame_result = self.detect_frame(frame, frame_idx, timestamp_sec)
                annotated = self.draw_detections(frame, frame_result)
                writer.write(annotated)

                yield frame_result, annotated, frame_idx, total_frames
        finally:
            cap.release()
            writer.release()
