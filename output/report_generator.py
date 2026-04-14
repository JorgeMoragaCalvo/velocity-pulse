from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter

from config.settings import GROUND_TRUTH_SPEEDS_KMH, SPEED_SMOOTHING_POLY, SPEED_SMOOTHING_WINDOW

logger = logging.getLogger(__name__)

# Type alias: method_name → list of (time_sec, speed_kmh)
SpeedTimeSeries = dict[str, list[tuple[float, float]]]
TraceData = dict[str, list[tuple[float, float]]]  # method → [(mx, my), ...]


class PDFReportGenerator:
    """
    Generates a 4-page PDF:
      1. Speed vs. Time
      2. Acceleration vs. Time
      3. Tracking Trace (pixel + metric)
      4. Benchmark Table
    """

    def __init__(
        self,
        speed_series: SpeedTimeSeries,
        pixel_traces: TraceData,
        metric_traces: TraceData,
        ground_truth_kmh: float | None = None,
    ) -> None:
        self._speed = speed_series
        self._pixel_traces = pixel_traces
        self._metric_traces = metric_traces
        self._gt = ground_truth_kmh

    # ------------------------------------------------------------------
    def generate(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(str(output_path)) as pdf:
            self._page_speed(pdf)
            self._page_acceleration(pdf)
            self._page_traces(pdf)
            self._page_benchmark(pdf)
        logger.info("PDF report saved: %s", output_path)

    # ------------------------------------------------------------------
    def _page_speed(self, pdf: PdfPages) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        for method, series in self._speed.items():
            if not series:
                continue
            ts = [s[0] for s in series]
            vs = [s[1] for s in series]
            ax.plot(ts, vs, label=method)
        if self._gt:
            ax.axhline(self._gt, linestyle="--", color="gray", label=f"GT {self._gt} km/h")
        for gt in GROUND_TRUTH_SPEEDS_KMH:
            ax.axhline(gt, linestyle=":", color="red", alpha=0.4, label=f"Ref {gt} km/h")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (km/h)")
        ax.set_title("Speed vs Time")
        ax.legend()
        ax.grid(True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _page_acceleration(self, pdf: PdfPages) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        for method, series in self._speed.items():
            if len(series) < 3:
                continue
            ts = np.array([s[0] for s in series])
            vs = np.array([s[1] for s in series]) / 3.6  # km/h → m/s
            dt = np.diff(ts)
            dt = np.where(dt == 0, 1e-6, dt)
            acc = np.diff(vs) / dt
            w = min(SPEED_SMOOTHING_WINDOW, len(acc) if len(acc) % 2 == 1 else len(acc) - 1)
            if w >= 3:
                acc = savgol_filter(acc, w, SPEED_SMOOTHING_POLY)
            ax.plot(ts[1:], acc, label=method)
        ax.axhline(0, linestyle="--", color="gray", alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s²)")
        ax.set_title("Acceleration vs Time")
        ax.legend()
        ax.grid(True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _page_traces(self, pdf: PdfPages) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_px, ax_mt = axes

        for method, pts in self._pixel_traces.items():
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax_px.plot(xs, ys, label=method)
        ax_px.set_title("Pixel-space Trajectory")
        ax_px.set_xlabel("x (px)")
        ax_px.set_ylabel("y (px)")
        ax_px.invert_yaxis()
        ax_px.legend()
        ax_px.grid(True)

        sc = None
        for method, pts in self._metric_traces.items():
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            sc = ax_mt.scatter(xs, ys, c=range(len(xs)), cmap="cool", s=10, label=method)
        ax_mt.set_title("Ground-Plane Metric Trajectory")
        ax_mt.set_xlabel("X (m)")
        ax_mt.set_ylabel("Y (m)")
        ax_mt.legend()
        ax_mt.grid(True)
        if sc is not None:
            plt.colorbar(sc, ax=ax_mt, label="Frame (start→end)")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _page_benchmark(self, pdf: PdfPages) -> None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")

        gt_speed = self._gt or 30.0
        col_labels = ["Method", "Mean (km/h)", "Std (km/h)", "MAE (km/h)", "RMSE (km/h)"]
        rows = []

        for method, series in self._speed.items():
            if not series:
                continue
            vs = np.array([s[1] for s in series])
            mean = float(vs.mean())
            std = float(vs.std())
            mae = float(np.abs(vs - gt_speed).mean())
            rmse = float(np.sqrt(((vs - gt_speed) ** 2).mean()))
            rows.append([method, f"{mean:.2f}", f"{std:.2f}", f"{mae:.2f}", f"{rmse:.2f}"])

        if rows:
            tbl = ax.table(
                cellText=rows,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(11)
            tbl.scale(1.2, 1.8)

        ax.set_title(f"Benchmark Table  (GT ≈ {gt_speed} km/h)", pad=20)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
