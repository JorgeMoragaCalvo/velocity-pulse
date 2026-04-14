from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def bottom_center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, self.y2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass
class DetectionResult:
    frame_idx: int
    track_id: int
    bbox: BoundingBox
    confidence: float
    class_name: str
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    # mask: HxW boolean array, None when segmentation not available
