from __future__ import annotations

from collections import defaultdict, deque

from scipy.signal import savgol_filter

from config.settings import SPEED_SMOOTHING_POLY, SPEED_SMOOTHING_WINDOW
from speed.interfaces import SpeedEstimate


class SpeedSmoother:
    """
    Applies Savitzky-Golay smoothing to a rolling window of speed estimates
    per track. Returns a smoothed km/h value without altering other fields.
    """

    def __init__(
        self,
        window: int = SPEED_SMOOTHING_WINDOW,
        poly: int = SPEED_SMOOTHING_POLY,
    ) -> None:
        self._window = window if window % 2 == 1 else window + 1
        self._poly = poly
        self._buffers: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._window * 3)
        )

    def smooth(self, estimate: SpeedEstimate) -> SpeedEstimate:
        buf = self._buffers[estimate.track_id]
        buf.append(estimate.speed_kmh)

        if len(buf) < self._window:
            return estimate  # not enough data yet

        smoothed = savgol_filter(
            list(buf),
            window_length=min(self._window, len(buf) if len(buf) % 2 == 1 else len(buf) - 1),
            polyorder=self._poly,
        )
        # Return a new SpeedEstimate with the last (current) smoothed value
        return SpeedEstimate(
            frame_idx=estimate.frame_idx,
            track_id=estimate.track_id,
            speed_kmh=max(0.0, float(smoothed[-1])),
            speed_px_per_frame=estimate.speed_px_per_frame,
            method=estimate.method,
            confidence=estimate.confidence,
        )
