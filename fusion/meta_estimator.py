from __future__ import annotations

import logging
from collections import defaultdict, deque

import numpy as np

from config.settings import FUSION_MAX_DEPTH_WEIGHT, FUSION_WARMUP_FRAMES
from depth.interfaces import DepthMap
from detection.detection_result import DetectionResult
from fusion.interfaces import FusedEstimate, IMetaEstimator
from speed.interfaces import SpeedEstimate
from speed.speed_smoother import SpeedSmoother

logger = logging.getLogger(__name__)


class WeightedFusionMetaEstimator(IMetaEstimator):
    """
    Meta-estimator that combines Module 1 speed methods with depth information.

    Stage 1 — Inverse-variance weighting:
        w_i = 1 / (var_i + ε)
        speed_base = Σ(w_i * speed_i) / Σ(w_i)

    Stage 2 — Depth Z-correction for radial (toward/away camera) motion:
        v_radial from delta depth between frames
        speed_fused = weighted blend of lateral and full 3D magnitude

    Stage 3 — Kalman-style smoothing via SpeedSmoother.
    """

    def __init__(self, depth_estimator=None) -> None:
        # Per-track rolling variance buffers (last WARMUP_FRAMES samples)
        self._speed_buffers: dict[int, dict[str, deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=FUSION_WARMUP_FRAMES))
        )
        self._prev_depth: dict[int, float] = {}  # track_id → previous depth value
        self._depth_estimator = depth_estimator  # optional MiDaSDepthEstimator
        self._smoother = SpeedSmoother()

    # ------------------------------------------------------------------
    def fuse(
        self,
        estimates: list[SpeedEstimate],
        depth_map: DepthMap | None,
        detection: DetectionResult,
    ) -> FusedEstimate | None:
        if not estimates:
            return None

        tid = detection.track_id

        # Record speed in variance buffers
        for est in estimates:
            self._speed_buffers[tid][est.method].append(est.speed_kmh)

        # Compute per-method variance and weights
        weights: dict[str, float] = {}
        eps = 1e-6
        for est in estimates:
            buf = list(self._speed_buffers[tid][est.method])
            if len(buf) < 2:
                var = 1.0  # high uncertainty during warmup
            else:
                var = float(np.var(buf))
            weights[est.method] = 1.0 / (var + eps)

        total_w = sum(weights.values())
        norm_weights = {m: w / total_w for m, w in weights.items()}

        speed_base = sum(norm_weights[est.method] * est.speed_kmh for est in estimates)

        # ---- Depth Z-correction ----
        depth_correction = 1.0
        speed_fused = speed_base

        if depth_map is not None and self._depth_estimator is not None:
            bbox = detection.bbox
            curr_depth_raw = self._depth_estimator.get_vehicle_depth(
                depth_map,
                detection.mask,
                bbox.as_xyxy(),
            )
            if curr_depth_raw > 0 and tid in self._prev_depth:
                prev_depth_raw = self._prev_depth[tid]
                scale = depth_map.metric_scale or 1.0
                dz_m = (curr_depth_raw - prev_depth_raw) * scale
                # Estimate radial velocity: assume 30 fps default if not stored
                # (fps is passed by pipeline via set_fps)
                fps = getattr(self, "_fps", 30.0)
                v_radial = abs(dz_m) * fps * 3.6  # km/h
                depth_weight = float(np.clip(v_radial / (speed_base + eps), 0, FUSION_MAX_DEPTH_WEIGHT))
                speed_3d = float(np.sqrt(speed_base ** 2 + v_radial ** 2))
                speed_fused = (1 - depth_weight) * speed_base + depth_weight * speed_3d
                depth_correction = speed_fused / (speed_base + eps)

            if curr_depth_raw > 0:
                self._prev_depth[tid] = curr_depth_raw

        # ---- Smooth ----
        raw_estimate = SpeedEstimate(
            frame_idx=detection.frame_idx,
            track_id=tid,
            speed_kmh=speed_fused,
            speed_px_per_frame=0.0,
            method="fusion",
            confidence=float(np.mean([e.confidence for e in estimates])),
        )
        smoothed = self._smoother.smooth(raw_estimate)

        return FusedEstimate(
            frame_idx=detection.frame_idx,
            track_id=tid,
            speed_kmh=smoothed.speed_kmh,
            contributing_methods=norm_weights,
            depth_correction_factor=depth_correction,
            confidence=smoothed.confidence,
        )

    def set_fps(self, fps: float) -> None:
        self._fps = fps
