from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DepthMap:
    frame_idx: int
    relative_depth: np.ndarray   # HxW float32, larger = further (MiDaS convention)
    metric_scale: Optional[float] = None  # metres per unit when calibrated


class IDepthEstimator(ABC):
    @abstractmethod
    def estimate(self, frame: np.ndarray, frame_idx: int = 0) -> DepthMap:
        ...
