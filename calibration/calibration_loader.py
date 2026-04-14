from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np

from config.settings import CALIBRATION_FILE

logger = logging.getLogger(__name__)


class CalibrationData:
    def __init__(
        self,
        homography: np.ndarray,
        image_points: list[tuple[float, float]],
        metric_points: list[tuple[float, float]],
        reprojection_error: float,
        fps: float,
        frame_width: int,
        frame_height: int,
    ) -> None:
        self.homography = homography
        self.image_points = image_points
        self.metric_points = metric_points
        self.reprojection_error = reprojection_error
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height

    @property
    def H(self) -> np.ndarray:
        return self.homography

    @property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.homography)


class CalibrationLoader:
    """Save and load calibration data from JSON."""

    def __init__(self, path: Path = CALIBRATION_FILE) -> None:
        self._path = path

    def save(
        self,
        homography: np.ndarray,
        image_points: list[tuple[float, float]],
        metric_points: list[tuple[float, float]],
        reprojection_error: float,
        fps: float,
        frame_width: int,
        frame_height: int,
    ) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "homography_matrix": homography.tolist(),
            "image_points": image_points,
            "metric_points": metric_points,
            "reprojection_error_m": reprojection_error,
            "fps": fps,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "calibration_date": str(date.today()),
        }
        with open(self._path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Calibration saved to %s", self._path)

    def load(self) -> CalibrationData:
        if not self._path.exists():
            raise FileNotFoundError(
                f"No calibration file at {self._path}. "
                "Run: python main.py --calibrate --image <path>"
            )
        with open(self._path) as f:
            data = json.load(f)

        return CalibrationData(
            homography=np.array(data["homography_matrix"], dtype=np.float64),
            image_points=data["image_points"],
            metric_points=data["metric_points"],
            reprojection_error=data["reprojection_error_m"],
            fps=data.get("fps", 30.0),
            frame_width=data.get("frame_width", 0),
            frame_height=data.get("frame_height", 0),
        )
