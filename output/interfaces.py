from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from detection.detection_result import DetectionResult
from speed.interfaces import SpeedEstimate


class IVideoRenderer(ABC):
    @abstractmethod
    def render(
        self,
        frame: np.ndarray,
        detections: list[DetectionResult],
        speed_estimates: dict[int, SpeedEstimate],
        track_traces: dict[int, list[tuple[float, float]]],
    ) -> np.ndarray:
        ...


class IReportGenerator(ABC):
    @abstractmethod
    def generate(self, output_path: Path) -> None:
        ...
