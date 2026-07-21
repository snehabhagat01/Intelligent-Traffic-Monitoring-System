"""
analytics.py
============
Turns raw per-frame detection results into aggregated traffic statistics,
density/congestion classification, and Plotly visualisations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config
from detector import FrameResult
from utils import safe_divide


@dataclass
class TrafficSummary:
    """High-level summary metrics computed over an entire video."""
    total_frames_analyzed: int = 0
    total_vehicles: int = 0
    counts_by_class: dict[str, int] = field(default_factory=dict)
    avg_vehicles_per_frame: float = 0.0
    most_detected_vehicle: str = "N/A"
    peak_frame: int = 0
    peak_vehicle_count: int = 0
    peak_timestamp_sec: float = 0.0
    density_level: str = "Low"
    congestion_color: str = config.DENSITY_COLORS["Low"]

    @property
    def percentages_by_class(self) -> dict[str, float]:
        return {
            cls: safe_divide(count, self.total_vehicles) * 100
            for cls, count in self.counts_by_class.items()
        }


class TrafficAnalytics:
    """
    Consumes a list of per-frame `FrameResult` objects (produced by
    `VehicleDetector`) and derives statistics, density classification,
    and ready-to-render Plotly figures.
    """

    def __init__(self, frame_results: list[FrameResult]) -> None:
        self.frame_results = frame_results
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # DATA PREPARATION
    # ------------------------------------------------------------------ #
    def to_dataframe(self) -> pd.DataFrame:
        """Flatten frame results into a tidy per-frame DataFrame (cached)."""
        if self._df is not None:
            return self._df

        rows = []
        for fr in self.frame_results:
            row = {
                "frame": fr.frame_number,
                "timestamp_sec": fr.timestamp_sec,
                "vehicle_count": fr.vehicle_count,
                "fps": fr.fps,
            }
            row.update(fr.counts_by_class)
            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(
                columns=["frame", "timestamp_sec", "vehicle_count", "fps", *config.VEHICLE_CLASSES]
            )
        self._df = df
        return df

    # ------------------------------------------------------------------ #
    # DENSITY / CONGESTION
    # ------------------------------------------------------------------ #
    @staticmethod
    def classify_density(vehicle_count: float) -> str:
        """Classify a vehicle count into Low / Medium / High density."""
        if vehicle_count <= config.DENSITY_LOW_MAX:
            return "Low"
        if vehicle_count <= config.DENSITY_MEDIUM_MAX:
            return "Medium"
        return "High"

    # ------------------------------------------------------------------ #
    # SUMMARY
    # ------------------------------------------------------------------ #
    def compute_summary(self) -> TrafficSummary:
        """Compute the full TrafficSummary for the analyzed video."""
        df = self.to_dataframe()
        summary = TrafficSummary()

        if df.empty:
            return summary

        summary.total_frames_analyzed = len(df)
        counts_by_class = {cls: int(df[cls].sum()) for cls in config.VEHICLE_CLASSES if cls in df}
        summary.counts_by_class = counts_by_class
        summary.total_vehicles = int(sum(counts_by_class.values()))
        summary.avg_vehicles_per_frame = round(safe_divide(summary.total_vehicles, len(df)), 2)

        if counts_by_class:
            summary.most_detected_vehicle = max(counts_by_class, key=counts_by_class.get)

        peak_idx = df["vehicle_count"].idxmax()
        summary.peak_frame = int(df.loc[peak_idx, "frame"])
        summary.peak_vehicle_count = int(df.loc[peak_idx, "vehicle_count"])
        summary.peak_timestamp_sec = float(df.loc[peak_idx, "timestamp_sec"])

        summary.density_level = self.classify_density(summary.avg_vehicles_per_frame)
        summary.congestion_color = config.DENSITY_COLORS[summary.density_level]

        return summary

    # ------------------------------------------------------------------ #
    # PLOTLY CHARTS
    # ------------------------------------------------------------------ #
    def vehicle_distribution_pie(self, summary: TrafficSummary) -> go.Figure:
        """Pie chart of vehicle class distribution."""
        labels = list(summary.counts_by_class.keys())
        values = list(summary.counts_by_class.values())
        colors = [
            f"rgb{tuple(int(config.BOX_COLORS_BGR[c][::-1][i]) for i in range(3))}"
            for c in labels
        ]

        fig = go.Figure(
            data=[go.Pie(labels=labels, values=values, hole=0.45, marker=dict(colors=colors))]
        )
        fig.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value} (%{percent})")
        fig.update_layout(
            title="Vehicle Distribution",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15),
            margin=dict(t=50, b=10, l=10, r=10),
        )
        return fig

    def vehicle_count_bar(self, summary: TrafficSummary) -> go.Figure:
        """Bar chart comparing total counts per vehicle class."""
        labels = list(summary.counts_by_class.keys())
        values = list(summary.counts_by_class.values())

        fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=config.THEME_COLORS["primary"])])
        fig.update_layout(
            title="Vehicle Count by Class",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Vehicle Class",
            yaxis_title="Count",
            margin=dict(t=50, b=10, l=10, r=10),
        )
        return fig

    def vehicle_trend_over_time(self) -> go.Figure:
        """Line chart of total vehicle count across the video timeline."""
        df = self.to_dataframe()
        fig = px.line(
            df, x="timestamp_sec", y="vehicle_count",
            labels={"timestamp_sec": "Time (s)", "vehicle_count": "Vehicles Detected"},
            title="Vehicle Count Trend Over Time",
        )
        fig.update_traces(line_color=config.THEME_COLORS["primary_light"], line_width=2)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=10, l=10, r=10),
        )
        return fig

    def traffic_density_timeline(self) -> go.Figure:
        """Step/area chart showing density classification changes over time."""
        df = self.to_dataframe().copy()
        if df.empty:
            return go.Figure()

        df["density"] = df["vehicle_count"].apply(self.classify_density)
        density_rank = {"Low": 1, "Medium": 2, "High": 3}
        df["density_rank"] = df["density"].map(density_rank)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["timestamp_sec"],
                y=df["density_rank"],
                mode="lines",
                line=dict(color=config.THEME_COLORS["warning"], width=2, shape="hv"),
                fill="tozeroy",
                fillcolor="rgba(243, 156, 18, 0.15)",
                customdata=df["density"],
                hovertemplate="Time: %{x:.1f}s<br>Density: %{customdata}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Traffic Density Timeline",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(tickmode="array", tickvals=[1, 2, 3], ticktext=["Low", "Medium", "High"]),
            xaxis_title="Time (s)",
            margin=dict(t=50, b=10, l=10, r=10),
        )
        return fig

    def congestion_gauge(self, summary: TrafficSummary) -> go.Figure:
        """Circular gauge indicator for the current congestion level."""
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=summary.avg_vehicles_per_frame,
                title={"text": f"Congestion Level: {summary.density_level}"},
                gauge={
                    "axis": {"range": [0, max(30, summary.avg_vehicles_per_frame + 5)]},
                    "bar": {"color": summary.congestion_color},
                    "steps": [
                        {"range": [0, config.DENSITY_LOW_MAX], "color": "rgba(46, 204, 113, 0.25)"},
                        {
                            "range": [config.DENSITY_LOW_MAX, config.DENSITY_MEDIUM_MAX],
                            "color": "rgba(243, 156, 18, 0.25)",
                        },
                        {"range": [config.DENSITY_MEDIUM_MAX, 40], "color": "rgba(231, 76, 60, 0.25)"},
                    ],
                },
            )
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=10, l=20, r=20),
            height=280,
        )
        return fig
