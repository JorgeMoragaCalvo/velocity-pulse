from __future__ import annotations

import logging

import cv2
import numpy as np

from config.settings import (
    MIN_TRACK_FRAMES,
    OPTICAL_FLOW_MAX_CORNERS,
    OPTICAL_FLOW_MIN_DIST,
    OPTICAL_FLOW_QUALITY,
    OPTICAL_FLOW_WIN_SIZE,
)
from detection.detection_result import DetectionResult
from speed.interfaces import ISpeedEstimator, SpeedEstimate
from tracking.track_state import TrackHistory

logger = logging.getLogger(__name__)


def _project(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
    p = H @ np.array([px, py, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])


class OpticalFlowSpeedEstimator(ISpeedEstimator):
    """Method 2: Lucas-Kanade sparse optical flow within the vehicle mask/BBOX."""

    def __init__(self) -> None:
        self._prev_gray: dict[int, np.ndarray] = {}
        self._prev_points: dict[int, np.ndarray] = {}
        self._next_corners: dict[int, np.ndarray] = {}

    def estimate(
        self,
        current: DetectionResult,
        history: TrackHistory,
        homography: np.ndarray,
        fps: float,
    ) -> SpeedEstimate | None:
        if len(history) < MIN_TRACK_FRAMES:
            return None

        prev_det = history.previous
        if prev_det is None:
            return None

        # We need the previous frame stored; handled by pipeline calling _seed()
        tid = current.track_id
        if tid not in self._prev_gray or tid not in self._prev_points:
            return None

        prev_gray = self._prev_gray[tid]
        prev_pts = self._prev_points[tid]
        if prev_pts is None or len(prev_pts) == 0:
            return None

        # Build current gray from the frame stored by pipeline
        if not hasattr(self, "_current_gray_cache"):
            return None
        curr_gray = self._current_gray_cache.get(tid)
        if curr_gray is None:
            return None

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_pts,
            None,
            winSize=OPTICAL_FLOW_WIN_SIZE,
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_pts is None or status is None:
            return None

        good_prev = prev_pts[status.flatten() == 1]
        good_next = next_pts[status.flatten() == 1]
        if len(good_prev) < 3:
            return None

        # Reject outlier flows via median
        flows = good_next - good_prev
        median_flow = np.median(flows, axis=0)
        dists = np.linalg.norm(flows - median_flow, axis=1)
        threshold = max(np.median(dists) * 2.5, 1.0)
        inliers = dists < threshold
        if inliers.sum() < 2:
            return None

        mean_flow = flows[inliers].mean(axis=0)
        fx, fy = float(mean_flow[0]), float(mean_flow[1])

        # Project flow vector to metric using homography
        bbox = current.bbox
        cx, cy = bbox.centroid
        mx0, my0 = _project(homography, cx, cy)
        mx1, my1 = _project(homography, cx + fx, cy + fy)
        dist_m = float(np.hypot(mx1 - mx0, my1 - my0))
        speed_kmh = dist_m * fps * 3.6

        px_dist = float(np.hypot(fx, fy))
        conf = min(inliers.sum() / max(len(prev_pts), 1), 1.0)

        return SpeedEstimate(
            frame_idx=current.frame_idx,
            track_id=tid,
            speed_kmh=speed_kmh,
            speed_px_per_frame=px_dist,
            method="optflow",
            confidence=conf,
        )

    # ------------------------------------------------------------------
    def seed(self, track_id: int, gray_frame: np.ndarray, mask: np.ndarray | None, bbox_xyxy: tuple) -> None:
        """Store the current frame's gray image and seed feature points."""
        if not hasattr(self, "_current_gray_cache"):
            self._current_gray_cache: dict[int, np.ndarray] = {}

        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        h_img, w_img = gray_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        roi_mask = np.zeros(gray_frame.shape[:2], dtype=np.uint8)
        if mask is not None:
            # Resize mask to frame size if needed
            m = mask.astype(np.uint8) * 255
            if m.shape != gray_frame.shape[:2]:
                m = cv2.resize(m, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            roi_mask = m
        else:
            roi_mask[y1:y2, x1:x2] = 255

        corners = cv2.goodFeaturesToTrack(
            gray_frame,
            maxCorners=OPTICAL_FLOW_MAX_CORNERS,
            qualityLevel=OPTICAL_FLOW_QUALITY,
            minDistance=OPTICAL_FLOW_MIN_DIST,
            mask=roi_mask,
        )

        # Promote previous current → previous
        if track_id in self._current_gray_cache:
            self._prev_gray[track_id] = self._current_gray_cache[track_id]
            if track_id in self._next_corners:
                self._prev_points[track_id] = self._next_corners[track_id]

        self._current_gray_cache[track_id] = gray_frame.copy()
        if corners is not None:
            self._next_corners[track_id] = corners
