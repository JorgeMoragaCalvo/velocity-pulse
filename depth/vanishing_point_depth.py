from __future__ import annotations

import logging

import cv2
import numpy as np

from depth.interfaces import DepthMap, IDepthEstimator

logger = logging.getLogger(__name__)


class VanishingPointDepthEstimator(IDepthEstimator):
    """
    Geometric depth proxy using the vanishing point of road/lane lines.
    Computes a relative depth map as (v - vy) / (H - vy) where:
      v  = row coordinate of a pixel
      vy = row of the vanishing point
      H  = image height
    Objects near the vanishing point get high depth; ground near camera gets low.
    Does NOT require a neural network.
    """

    def __init__(self, hough_threshold: int = 80, min_line_len: int = 50, max_gap: int = 10) -> None:
        self._hough_threshold = hough_threshold
        self._min_line_len = min_line_len
        self._max_gap = max_gap
        self._cached_vy: float | None = None

    def estimate(self, frame: np.ndarray, frame_idx: int = 0) -> DepthMap:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

        h, w = frame.shape[:2]
        vy = self._find_vanishing_row(edges, h, w)
        self._cached_vy = vy

        rows = np.arange(h, dtype=np.float32)
        denom = max(h - vy, 1.0)
        depth_row = np.clip((rows - vy) / denom, 0, 1)
        depth_map = np.tile(depth_row[:, None], (1, w))

        return DepthMap(
            frame_idx=frame_idx,
            relative_depth=depth_map.astype(np.float32),
            metric_scale=None,
        )

    def _find_vanishing_row(self, edges: np.ndarray, h: int, w: int) -> float:
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self._hough_threshold,
            minLineLength=self._min_line_len,
            maxLineGap=self._max_gap,
        )
        if lines is None:
            return h * 0.4  # fallback: 40% from top

        # Collect intersections of non-horizontal lines
        intersect_ys: list[float] = []
        line_params = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < 5:
                continue  # near-vertical, skip
            slope = dy / dx
            intercept = y1 - slope * x1
            line_params.append((slope, intercept))

        for i in range(len(line_params)):
            for j in range(i + 1, len(line_params)):
                m1, b1 = line_params[i]
                m2, b2 = line_params[j]
                if abs(m1 - m2) < 0.01:
                    continue
                xi = (b2 - b1) / (m1 - m2)
                yi = m1 * xi + b1
                if 0 < yi < h * 0.7:
                    intersect_ys.append(yi)

        if not intersect_ys:
            return h * 0.4

        return float(np.median(intersect_ys))
