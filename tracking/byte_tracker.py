from __future__ import annotations

import logging

import numpy as np

from config.settings import MAX_LOST_FRAMES
from detection.detection_result import DetectionResult
from tracking.interfaces import ITracker
from tracking.track_state import TrackHistory, TrackState, TrackStatus

logger = logging.getLogger(__name__)


class TrackManager(ITracker):
    """
    Thin manager that wraps ultralytics built-in BoT-SORT tracking
    (tracking already happens inside YOLODetector.detect()).

    This class maintains per-track history and status metadata that
    the speed estimators need.
    """

    def __init__(self, max_lost: int = MAX_LOST_FRAMES) -> None:
        self._max_lost = max_lost
        self._histories: dict[int, TrackHistory] = {}
        self._states: dict[int, TrackState] = {}

    # ------------------------------------------------------------------
    def update(
        self,
        detections: list[DetectionResult],
        frame: np.ndarray,
    ) -> list[DetectionResult]:
        """Record new detections and update track states. Returns same detections."""
        active_ids = {d.track_id for d in detections}

        # Update lost counters for absent tracks
        for tid, state in list(self._states.items()):
            if tid not in active_ids:
                state.lost_frames += 1
                state.status = TrackStatus.LOST
                if state.lost_frames > self._max_lost:
                    del self._states[tid]
                    del self._histories[tid]

        # Process active detections
        for det in detections:
            tid = det.track_id
            if tid not in self._states:
                self._states[tid] = TrackState(track_id=tid)
                self._histories[tid] = TrackHistory(track_id=tid)
            self._states[tid].status = TrackStatus.TRACKED
            self._states[tid].lost_frames = 0
            self._histories[tid].add(det)

        return detections

    # ------------------------------------------------------------------
    def get_history(self, track_id: int) -> TrackHistory | None:
        return self._histories.get(track_id)

    def get_state(self, track_id: int) -> TrackState | None:
        return self._states.get(track_id)

    def active_track_ids(self) -> list[int]:
        return list(self._states.keys())
