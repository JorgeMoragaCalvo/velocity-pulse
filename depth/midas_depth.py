from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch

from config.settings import MIDAS_MODEL
from depth.interfaces import DepthMap, IDepthEstimator

logger = logging.getLogger(__name__)


class MiDaSDepthEstimator(IDepthEstimator):
    """
    Monocular depth estimation using MiDaS via torch.hub.
    Produces relative inverse depth (disparity); scale can be calibrated
    using a known-distance reference object.
    """

    def __init__(
        self,
        model_type: str = MIDAS_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info("Loading MiDaS model: %s on %s", model_type, self._device)

        self._model = torch.hub.load(
            "intel-isl/MiDaS",
            model_type,
            pretrained=True,
            trust_repo=True,
        )
        self._model.to(self._device).eval()

        transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True,
        )
        if model_type in ("DPT_Large", "DPT_Hybrid"):
            self._transform = transforms.dpt_transform
        else:
            self._transform = transforms.small_transform

        self._metric_scale: Optional[float] = None
        logger.info("MiDaS ready.")

    # ------------------------------------------------------------------
    def calibrate_scale(self, reference_frame: np.ndarray, known_distance_m: float, mask: np.ndarray) -> None:
        """
        Set the metric scale using a reference object at a known distance.
        mask: HxW boolean array indicating pixels of the reference object.
        """
        depth_map = self.estimate(reference_frame)
        depth_vals = depth_map.relative_depth[mask]
        if len(depth_vals) == 0:
            logger.warning("Empty mask for depth scale calibration.")
            return
        median_depth = float(np.median(depth_vals))
        if median_depth > 0:
            self._metric_scale = known_distance_m / median_depth
            logger.info("MiDaS metric scale set: %.4f m/unit", self._metric_scale)

    def estimate(self, frame: np.ndarray, frame_idx: int = 0) -> DepthMap:
        """Run MiDaS inference on a BGR frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(rgb).to(self._device)

        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)

        return DepthMap(
            frame_idx=frame_idx,
            relative_depth=depth,
            metric_scale=self._metric_scale,
        )

    @staticmethod
    def get_vehicle_depth(depth_map: DepthMap, mask: np.ndarray | None, bbox_xyxy: tuple) -> float:
        """Return the median depth value within the vehicle region."""
        d = depth_map.relative_depth
        if mask is not None:
            h, w = d.shape
            m = mask.astype(bool)
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            vals = d[m]
        else:
            x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
            vals = d[y1:y2, x1:x2].flatten()

        if len(vals) == 0:
            return 0.0
        return float(np.median(vals))
