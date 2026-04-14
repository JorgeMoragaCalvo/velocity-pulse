from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from benchmark.benchmark_runner import BenchmarkRunner
from detection.yolo_detector import YOLODetector
from output.osd_renderer import OSDRenderer
from output.report_generator import PDFReportGenerator
from output.video_writer import VideoWriter
from pipeline.interfaces import IPipeline
from speed.bbox_speed import BBoxBottomCenterSpeedEstimator
from speed.centroid_speed import CentroidSpeedEstimator
from speed.interfaces import SpeedEstimate
from speed.optical_flow_speed import OpticalFlowSpeedEstimator
from speed.speed_smoother import SpeedSmoother
from tracking.byte_tracker import TrackManager

logger = logging.getLogger(__name__)


def _project(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
    p = H @ np.array([px, py, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])


class Module1Pipeline(IPipeline):
    """
    Runs YOLO detection + BoT-SORT tracking + 3 speed estimation methods.
    Generates OSD video and PDF report per method.
    """

    def __init__(
        self,
        homography: np.ndarray,
        fps: float | None = None,
        method: str = "all",
        device: str = "cpu",
    ) -> None:
        self._H = homography
        self._H_inv = np.linalg.inv(homography)
        self._fps_override = fps
        self._method = method
        self._device = device

    # ------------------------------------------------------------------
    def run(
        self,
        input_path: Path,
        output_dir: Path,
        ground_truth_kmh: float | None = None,
    ) -> None:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        fps = self._fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Processing %s  fps=%.1f  %dx%d  frames=%d", input_path.name, fps, width, height, total)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Components
        detector = YOLODetector(device=self._device)
        tracker = TrackManager()
        estimators = {
            "centroid": CentroidSpeedEstimator(),
            "optflow": OpticalFlowSpeedEstimator(),
            "bbox": BBoxBottomCenterSpeedEstimator(),
        }
        if self._method != "all":
            estimators = {k: v for k, v in estimators.items() if k == self._method}

        smoothers = {m: SpeedSmoother() for m in estimators}
        renderer = OSDRenderer(H_inv=self._H_inv)
        benchmark = BenchmarkRunner(ground_truth_kmh=ground_truth_kmh or 30.0)

        # Per-method accumulators for report
        speed_series: dict[str, dict[str, list]] = {m: defaultdict(list) for m in estimators}
        metric_traces: dict[str, list] = defaultdict(list)
        pixel_traces: dict[str, list] = defaultdict(list)

        # Video writers — one per method
        writers: dict[str, VideoWriter] = {}
        for m in estimators:
            stem = input_path.stem
            out_path = output_dir / f"{stem}_{m}.mp4"
            writers[m] = VideoWriter(out_path, fps, width, height)

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = detector.detect(frame)
                tracker.update(detections, frame)

                # Seed optical flow
                of_est = estimators.get("optflow")
                if of_est and isinstance(of_est, OpticalFlowSpeedEstimator):
                    for det in detections:
                        of_est.seed(
                            det.track_id,
                            gray,
                            det.mask,
                            det.bbox.as_xyxy(),
                        )

                for method_name, estimator in estimators.items():
                    current_speeds: dict[int, SpeedEstimate] = {}
                    for det in detections:
                        history = tracker.get_history(det.track_id)
                        if history is None:
                            continue
                        raw = estimator.estimate(det, history, self._H, fps)
                        if raw is None:
                            continue
                        smoothed = smoothers[method_name].smooth(raw)
                        current_speeds[det.track_id] = smoothed
                        benchmark.record(smoothed)

                        # Accumulate time series
                        t = frame_idx / fps
                        speed_series[method_name][det.track_id].append((t, smoothed.speed_kmh))

                        # Trace
                        cx, cy = det.bbox.centroid
                        mx, my = _project(self._H, cx, cy)
                        metric_traces[method_name].append((mx, my))
                        pixel_traces[method_name].append((cx, cy))

                    # Render OSD frame for this method
                    rendered = renderer.render(frame, detections, current_speeds, {})
                    writers[method_name].write(rendered)

                frame_idx += 1
                if frame_idx % 100 == 0:
                    logger.info("  Frame %d / %d", frame_idx, total)

        finally:
            cap.release()
            for w in writers.values():
                w.release()

        benchmark.print_table()

        # Flatten per-track series to single list for report
        flat_speed: dict[str, list] = {}
        for method_name in estimators:
            merged: list[tuple[float, float]] = []
            for track_series in speed_series[method_name].values():
                merged.extend(track_series)
            merged.sort(key=lambda x: x[0])
            flat_speed[method_name] = merged

        # PDF report
        gt = ground_truth_kmh or 30.0
        report = PDFReportGenerator(
            speed_series=flat_speed,
            pixel_traces={m: pixel_traces[m] for m in estimators},
            metric_traces={m: metric_traces[m] for m in estimators},
            ground_truth_kmh=gt,
        )
        stem = input_path.stem
        report_path = output_dir / f"{stem}_module1_report.pdf"
        report.generate(report_path)
        logger.info("Module 1 pipeline complete. Report: %s", report_path)
