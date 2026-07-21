"""
report_generator.py
====================
Generates downloadable reports from analysis results:
  - vehicle_statistics.csv  (per-frame detection data)
  - traffic_summary.csv     (aggregated summary)
  - traffic_report.pdf      (formatted PDF report with charts, via ReportLab)
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import config
from analytics import TrafficSummary
from utils import filename_safe_timestamp, timestamp_now


# --------------------------------------------------------------------------- #
# CSV REPORTS
# --------------------------------------------------------------------------- #
def generate_vehicle_statistics_csv(df: pd.DataFrame) -> bytes:
    """Return the per-frame detection DataFrame encoded as CSV bytes."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def generate_traffic_summary_csv(summary: TrafficSummary) -> bytes:
    """Return the aggregated traffic summary encoded as CSV bytes."""
    rows = [
        {"Metric": "Total Frames Analyzed", "Value": summary.total_frames_analyzed},
        {"Metric": "Total Vehicles Detected", "Value": summary.total_vehicles},
        *[
            {"Metric": f"Total {cls}s", "Value": count}
            for cls, count in summary.counts_by_class.items()
        ],
        {"Metric": "Average Vehicles per Frame", "Value": summary.avg_vehicles_per_frame},
        {"Metric": "Most Detected Vehicle", "Value": summary.most_detected_vehicle},
        {"Metric": "Peak Frame", "Value": summary.peak_frame},
        {"Metric": "Peak Vehicle Count", "Value": summary.peak_vehicle_count},
        {"Metric": "Peak Timestamp (s)", "Value": round(summary.peak_timestamp_sec, 2)},
        {"Metric": "Traffic Density Level", "Value": summary.density_level},
        {"Metric": "Report Generated", "Value": timestamp_now()},
    ]
    df = pd.DataFrame(rows)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


# --------------------------------------------------------------------------- #
# PDF REPORT
# --------------------------------------------------------------------------- #
def generate_pdf_report(
    summary: TrafficSummary,
    video_name: str,
    chart_images: dict[str, bytes] | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    Build a formatted PDF traffic report using ReportLab.

    chart_images: optional dict mapping a chart title -> PNG image bytes
                  (e.g. produced via `fig.to_image(format='png')`) to embed.
    """
    if output_path is None:
        output_path = config.REPORTS_DIR / f"traffic_report_{filename_safe_timestamp()}.pdf"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], textColor=colors.HexColor("#0b1220"),
        fontSize=22, spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle", parent=styles["Normal"], textColor=colors.HexColor("#555555"),
        fontSize=11, spaceAfter=20,
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"], textColor=colors.HexColor("#2f81f7"),
        spaceBefore=14, spaceAfter=8,
    )
    body_style = styles["Normal"]

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=1.7 * cm,
        rightMargin=1.7 * cm,
    )

    story = []

    # --- Header ---
    story.append(Paragraph(f"{config.APP_ICON} {config.APP_TITLE}", title_style))
    story.append(Paragraph("AI-Generated Traffic Analysis Report", subtitle_style))

    meta_table = Table(
        [
            ["Video File:", video_name],
            ["Report Generated:", timestamp_now()],
            ["Detection Model:", "YOLOv8n (pretrained on COCO dataset)"],
        ],
        colWidths=[4 * cm, 11 * cm],
    )
    meta_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#2f81f7")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # --- Summary Section ---
    story.append(Paragraph("Traffic Summary", heading_style))
    summary_rows = [["Metric", "Value"]]
    summary_rows += [
        ["Total Frames Analyzed", str(summary.total_frames_analyzed)],
        ["Total Vehicles Detected", str(summary.total_vehicles)],
        ["Average Vehicles / Frame", str(summary.avg_vehicles_per_frame)],
        ["Most Detected Vehicle", summary.most_detected_vehicle],
        ["Peak Vehicle Count", f"{summary.peak_vehicle_count} (frame {summary.peak_frame})"],
        ["Peak Timestamp", f"{summary.peak_timestamp_sec:.1f}s"],
        ["Traffic Density Level", summary.density_level],
    ]
    summary_table = Table(summary_rows, colWidths=[7 * cm, 8 * cm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b1220")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fc")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # --- Vehicle Counts by Class ---
    story.append(Paragraph("Vehicle Counts by Class", heading_style))
    class_rows = [["Vehicle Class", "Count", "Percentage"]]
    pct = summary.percentages_by_class
    for cls, count in summary.counts_by_class.items():
        class_rows.append([cls, str(count), f"{pct.get(cls, 0):.1f}%"])
    class_table = Table(class_rows, colWidths=[6 * cm, 4.5 * cm, 4.5 * cm])
    class_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f81f7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fc")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(class_table)
    story.append(Spacer(1, 14))

    # --- Charts ---
    if chart_images:
        story.append(Paragraph("Analytics Charts", heading_style))
        for title, img_bytes in chart_images.items():
            story.append(Paragraph(title, body_style))
            story.append(Spacer(1, 4))
            try:
                img_buffer = io.BytesIO(img_bytes)
                story.append(Image(img_buffer, width=15 * cm, height=8 * cm))
            except Exception:  # noqa: BLE001
                story.append(Paragraph("(chart image unavailable)", body_style))
            story.append(Spacer(1, 10))

    # --- Footer note ---
    story.append(Spacer(1, 16))
    story.append(
        Paragraph(
            "Generated automatically by the Intelligent Traffic Monitoring System "
            "using a pretrained YOLOv8 model (COCO dataset). This report is intended "
            "for traffic monitoring and analytical purposes only.",
            ParagraphStyle("Footer", parent=body_style, fontSize=8, textColor=colors.grey),
        )
    )

    doc.build(story)
    return output_path
