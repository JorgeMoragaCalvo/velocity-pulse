from __future__ import annotations

import cv2
import numpy as np

from config.settings import (
    OSD_FONT_SCALE,
    OSD_MASK_ALPHA,
    OSD_THICKNESS,
    OSD_TRACE_LEN,
    SPEEDOMETER_MAX_KMH,
)
from detection.detection_result import DetectionResult
from output.interfaces import IVideoRenderer
from speed.interfaces import SpeedEstimate

# Color palette per track ID
_PALETTE = [
    (0, 255, 0),    # green
    (255, 165, 0),  # orange
    (0, 200, 255),  # cyan
    (200, 0, 255),  # purple
    (255, 255, 0),  # yellow
]


def _track_color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


class OSDRenderer(IVideoRenderer):
    """Draws BBOX, mask, speed readout, speedometer gauge, and trajectory trace."""

    def __init__(self, H_inv: np.ndarray | None = None) -> None:
        self._H_inv = H_inv  # for projecting metric trace back to image coords

    def render(
        self,
        frame: np.ndarray,
        detections: list[DetectionResult],
        speed_estimates: dict[int, SpeedEstimate],
        track_traces: dict[int, list[tuple[float, float]]],  # metric coords
    ) -> np.ndarray:
        out = frame.copy()

        for det in detections:
            color = _track_color(det.track_id)
            self._draw_mask(out, det, color)
            self._draw_bbox(out, det, color)
            self._draw_speed_label(out, det, speed_estimates.get(det.track_id))

        # Draw trajectory traces
        for tid, trace in track_traces.items():
            self._draw_trace(out, tid, trace)

        # Draw speedometer for each active track
        gauge_x = out.shape[1] - 120
        gauge_y = 80
        for tid, est in speed_estimates.items():
            self._draw_speedometer(out, est.speed_kmh, gauge_x, gauge_y)
            gauge_y += 130

        return out

    # ------------------------------------------------------------------
    def _draw_mask(self, frame: np.ndarray, det: DetectionResult, color: tuple) -> None:
        if det.mask is None:
            return
        mask = det.mask
        h, w = frame.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay = frame.copy()
        overlay[mask] = color
        cv2.addWeighted(overlay, OSD_MASK_ALPHA, frame, 1 - OSD_MASK_ALPHA, 0, frame)

    def _draw_bbox(self, frame: np.ndarray, det: DetectionResult, color: tuple) -> None:
        x1, y1, x2, y2 = [int(v) for v in det.bbox.as_xyxy()]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, OSD_THICKNESS)
        label = f"ID:{det.track_id} {det.class_name}"
        cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    OSD_FONT_SCALE * 0.8, color, 1, cv2.LINE_AA)

    def _draw_speed_label(
        self,
        frame: np.ndarray,
        det: DetectionResult,
        est: SpeedEstimate | None,
    ) -> None:
        if est is None:
            return
        x1, y1, x2, y2 = [int(v) for v in det.bbox.as_xyxy()]
        text = f"{est.speed_kmh:.1f} km/h"
        cv2.putText(frame, text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    OSD_FONT_SCALE, (255, 255, 255), OSD_THICKNESS, cv2.LINE_AA)
        method_tag = f"[{est.method}]"
        cv2.putText(frame, method_tag, (x1, y2 + 38), cv2.FONT_HERSHEY_SIMPLEX,
                    OSD_FONT_SCALE * 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_trace(
        self,
        frame: np.ndarray,
        track_id: int,
        trace: list[tuple[float, float]],  # metric ground-plane coords
    ) -> None:
        if self._H_inv is None or len(trace) < 2:
            return
        color = _track_color(track_id)
        pts = trace[-OSD_TRACE_LEN:]
        pixel_pts: list[tuple[int, int]] = []
        for mx, my in pts:
            p = self._H_inv @ np.array([mx, my, 1.0])
            px, py = int(p[0] / p[2]), int(p[1] / p[2])
            pixel_pts.append((px, py))
        for i in range(1, len(pixel_pts)):
            alpha = i / len(pixel_pts)
            c = tuple(int(v * alpha) for v in color)
            cv2.line(frame, pixel_pts[i - 1], pixel_pts[i], c, 2, cv2.LINE_AA)

    def _draw_speedometer(self, frame: np.ndarray, speed_kmh: float, cx: int, cy: int) -> None:
        radius = 45
        # Background arc
        cv2.ellipse(frame, (cx, cy), (radius, radius), -210, 0, 240, (60, 60, 60), 8)
        # Speed arc
        angle = int(min(speed_kmh / SPEEDOMETER_MAX_KMH, 1.0) * 240)
        if angle > 0:
            cv2.ellipse(frame, (cx, cy), (radius, radius), -210, 0, angle, (0, 200, 255), 6)
        # Text
        cv2.putText(frame, f"{speed_kmh:.0f}", (cx - 20, cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "km/h", (cx - 18, cy + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
