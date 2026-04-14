from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ICalibratorStrategy(ABC):
    """Computes a homography matrix from point correspondences."""

    @abstractmethod
    def calibrate(
        self,
        image_points: list[tuple[float, float]],
        metric_points: list[tuple[float, float]],
    ) -> np.ndarray:
        """Return a 3x3 homography matrix H mapping pixel→metric ground plane."""
        ...
