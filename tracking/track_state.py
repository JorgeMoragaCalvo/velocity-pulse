from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from detection.detection_result import DetectionResult


class TrackStatus(str, Enum):
    TRACKED = "tracked"
    INTERPOLATED = "interpolated"
    LOST = "lost"


@dataclass
class TrackState:
    track_id: int
    status: TrackStatus = TrackStatus.TRACKED
    lost_frames: int = 0


@dataclass
class TrackHistory:
    track_id: int
    detections: deque[DetectionResult] = field(default_factory=lambda: deque(maxlen=120))

    def add(self, det: DetectionResult) -> None:
        self.detections.append(det)

    def __len__(self) -> int:
        return len(self.detections)

    @property
    def latest(self) -> DetectionResult | None:
        return self.detections[-1] if self.detections else None

    @property
    def previous(self) -> DetectionResult | None:
        return self.detections[-2] if len(self.detections) >= 2 else None
