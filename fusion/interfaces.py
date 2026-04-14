from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from depth.interfaces import DepthMap
from detection.detection_result import DetectionResult
from speed.interfaces import SpeedEstimate


@dataclass
class FusedEstimate:
    frame_idx: int
    track_id: int
    speed_kmh: float
    contributing_methods: dict[str, float] = field(default_factory=dict)  # method → weight
    depth_correction_factor: float = 1.0
    method: str = "fusion"
    confidence: float = 1.0


class IMetaEstimator(ABC):
    @abstractmethod
    def fuse(
        self,
        estimates: list[SpeedEstimate],
        depth_map: DepthMap | None,
        detection: DetectionResult,
    ) -> FusedEstimate | None:
        ...
