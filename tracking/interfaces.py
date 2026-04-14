from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from detection.detection_result import DetectionResult


class ITracker(ABC):
    @abstractmethod
    def update(
        self,
        detections: list[DetectionResult],
        frame: np.ndarray,
    ) -> list[DetectionResult]:
        """Update tracker state; return detections with stable track IDs."""
        ...
