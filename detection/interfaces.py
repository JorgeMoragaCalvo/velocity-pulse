from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from detection.detection_result import DetectionResult


class IDetector(ABC):
    """Detects and optionally segments vehicles in a single frame."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Return a list of detections for the given BGR frame."""
        ...
