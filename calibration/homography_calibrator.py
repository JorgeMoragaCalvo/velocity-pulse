from __future__ import annotations

import logging

import cv2
import numpy as np

from calibration.interfaces import ICalibratorStrategy

logger = logging.getLogger(__name__)


class HomographyCalibrator(ICalibratorStrategy):
    """Computes H via cv2.findHomography (RANSAC) from ≥4 point pairs."""

    def __init__(self, ransac_threshold: float = 5.0) -> None:
        self._ransac_threshold = ransac_threshold

    def calibrate(
        self,
        image_points: list[tuple[float, float]],
        metric_points: list[tuple[float, float]],
    ) -> np.ndarray:
        if len(image_points) < 4:
            raise ValueError("Need at least 4 point pairs for homography.")

        src = np.array(image_points, dtype=np.float32)
        dst = np.array(metric_points, dtype=np.float32)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self._ransac_threshold)
        if H is None:
            raise RuntimeError("findHomography failed — check point quality.")

        inliers = int(mask.sum()) if mask is not None else len(src)
        logger.info(
            "Homography computed: %d/%d inliers, reprojection error=%.4f m",
            inliers,
            len(src),
            self._reprojection_error(H, src, dst),
        )
        return H

    @staticmethod
    def _reprojection_error(
        H: np.ndarray,
        src: np.ndarray,
        dst: np.ndarray,
    ) -> float:
        src_h = np.hstack([src, np.ones((len(src), 1))])
        proj = (H @ src_h.T).T
        proj /= proj[:, 2:3]
        err = np.linalg.norm(proj[:, :2] - dst, axis=1)
        return float(err.mean())
