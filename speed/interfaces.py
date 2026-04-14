from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from detection.detection_result import DetectionResult
from tracking.track_state import TrackHistory


@dataclass
class SpeedEstimate:
    frame_idx: int
    track_id: int
    speed_kmh: float
    speed_px_per_frame: float
    method: str
    confidence: float  # 0-1


class ISpeedEstimator(ABC):
    @abstractmethod
    def estimate(
        self,
        current: DetectionResult,
        history: TrackHistory,
        homography: np.ndarray,
        fps: float,
    ) -> SpeedEstimate | None:
        """Return speed estimate or None if not enough history."""
        ...
