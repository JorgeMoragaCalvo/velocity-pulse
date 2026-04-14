from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from benchmark.benchmark_runner import BenchmarkRunner
from config.settings import DEPTH_INFERENCE_INTERVAL
from depth.midas_depth import MiDaSDepthEstimator
from detection.yolo_detector import YOLODetector
from fusion.meta_estimator import WeightedFusionMetaEstimator
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


class Module2Pipeline(IPipeline):
    """
    Module 2: all Module 1 methods + MiDaS depth + WeightedFusionMetaEstimator.
    Outputs fusion video + PDF report.
    """

    def __init__(
        self,
        homography: np.ndarray,
        fps: float | None = None,
        device: str = "cpu",
    ) -> None:
        self._H = homography
        self._H_inv = np.linalg.inv(homography)
        self._fps_override = fps
        self._device = device

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
        logger.info("Module2 processing %s  fps=%.1f  %dx%d", input_path.name, fps, width, height)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Components
        detector = YOLODetector(device=self._device)
        tracker = TrackManager()
        depth_est = MiDaSDepthEstimator(device=self._device)

        module1_estimators = {
            "centroid": CentroidSpeedEstimator(),
            "optflow": OpticalFlowSpeedEstimator(),
            "bbox": BBoxBottomCenterSpeedEstimator(),
        }
        smoothers = {m: SpeedSmoother() for m in module1_estimators}
        meta = WeightedFusionMetaEstimator(depth_estimator=depth_est)
        meta.set_fps(fps)

        renderer = OSDRenderer(H_inv=self._H_inv)
        benchmark = BenchmarkRunner(ground_truth_kmh=ground_truth_kmh or 30.0)

        out_path = output_dir / f"{input_path.stem}_fusion.mp4"
        writer = VideoWriter(out_path, fps, width, height)

        speed_series: dict[str, list] = defaultdict(list)
        metric_traces: dict[str, list] = defaultdict(list)
        pixel_traces: dict[str, list] = defaultdict(list)

        frame_idx = 0
        depth_map = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = detector.detect(frame)
                tracker.update(detections, frame)

                # Run depth estimation every N frames
                if frame_idx % DEPTH_INFERENCE_INTERVAL == 0:
                    depth_map = depth_est.estimate(frame, frame_idx)

                # Seed optical flow
                of_est = module1_estimators.get("optflow")
                if isinstance(of_est, OpticalFlowSpeedEstimator):
                    for det in detections:
                        of_est.seed(det.track_id, gray, det.mask, det.bbox.as_xyxy())

                current_fused: dict[int, SpeedEstimate] = {}

                for det in detections:
                    history = tracker.get_history(det.track_id)
                    if history is None:
                        continue

                    # Collect all Module 1 estimates
                    m1_estimates = []
                    for m_name, estimator in module1_estimators.items():
                        raw = estimator.estimate(det, history, self._H, fps)
                        if raw is None:
                            continue
                        smoothed = smoothers[m_name].smooth(raw)
                        m1_estimates.append(smoothed)

                    if not m1_estimates:
                        continue

                    fused = meta.fuse(m1_estimates, depth_map, det)
                    if fused is None:
                        continue

                    # Convert FusedEstimate → SpeedEstimate for OSD/report
                    speed_est = SpeedEstimate(
                        frame_idx=fused.frame_idx,
                        track_id=fused.track_id,
                        speed_kmh=fused.speed_kmh,
                        speed_px_per_frame=0.0,
                        method="fusion",
                        confidence=fused.confidence,
                    )
                    current_fused[det.track_id] = speed_est
                    benchmark.record(speed_est)

                    t = frame_idx / fps
                    speed_series["fusion"].append((t, fused.speed_kmh))
                    cx, cy = det.bbox.centroid
                    mx, my = _project(self._H, cx, cy)
                    metric_traces["fusion"].append((mx, my))
                    pixel_traces["fusion"].append((cx, cy))

                rendered = renderer.render(frame, detections, current_fused, {})
                writer.write(rendered)

                frame_idx += 1
                if frame_idx % 100 == 0:
                    logger.info("  Frame %d / %d", frame_idx, total)

        finally:
            cap.release()
            writer.release()

        benchmark.print_table()

        report = PDFReportGenerator(
            speed_series=dict(speed_series),
            pixel_traces=dict(pixel_traces),
            metric_traces=dict(metric_traces),
            ground_truth_kmh=ground_truth_kmh,
        )
        report_path = output_dir / f"{input_path.stem}_module2_report.pdf"
        report.generate(report_path)
        logger.info("Module 2 pipeline complete. Report: %s", report_path)
