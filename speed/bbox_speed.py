from __future__ import annotations

import numpy as np

from config.settings import MIN_TRACK_FRAMES
from detection.detection_result import DetectionResult
from speed.interfaces import ISpeedEstimator, SpeedEstimate
from tracking.track_state import TrackHistory


def _project(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
    p = H @ np.array([px, py, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])


class BBoxBottomCenterSpeedEstimator(ISpeedEstimator):
    """
    Method 3: Ground-plane speed from the BBOX bottom-center point.

    The bottom-center (cx, y2) is the closest proxy for the vehicle's
    ground-contact point, making it the most geometrically correct point
    to project through a ground-plane homography.
    """

    def estimate(
        self,
        current: DetectionResult,
        history: TrackHistory,
        homography: np.ndarray,
        fps: float,
    ) -> SpeedEstimate | None:
        if len(history) < MIN_TRACK_FRAMES:
            return None

        prev = history.previous
        if prev is None:
            return None

        cx_cur, cy_cur = current.bbox.bottom_center
        cx_prv, cy_prv = prev.bbox.bottom_center

        mx_cur, my_cur = _project(homography, cx_cur, cy_cur)
        mx_prv, my_prv = _project(homography, cx_prv, cy_prv)

        dist_m = float(np.hypot(mx_cur - mx_prv, my_cur - my_prv))
        speed_kmh = dist_m * fps * 3.6
        px_dist = float(np.hypot(cx_cur - cx_prv, cy_cur - cy_prv))

        return SpeedEstimate(
            frame_idx=current.frame_idx,
            track_id=current.track_id,
            speed_kmh=speed_kmh,
            speed_px_per_frame=px_dist,
            method="bbox",
            confidence=min(current.confidence, 1.0),
        )
