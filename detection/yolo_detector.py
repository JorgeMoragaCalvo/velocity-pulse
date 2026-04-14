from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from config.settings import (
    YOLO_MODEL,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_VEHICLE_CLASSES,
    TRACKER_CONFIG,
)
from detection.detection_result import BoundingBox, DetectionResult
from detection.interfaces import IDetector

logger = logging.getLogger(__name__)


class YOLODetector(IDetector):
    """YOLOv8-seg detector with built-in BoT-SORT tracking via ultralytics."""

    def __init__(
        self,
        model_path: str = YOLO_MODEL,
        conf: float = YOLO_CONF_THRESHOLD,
        iou: float = YOLO_IOU_THRESHOLD,
        vehicle_classes: tuple[int, ...] = YOLO_VEHICLE_CLASSES,
        tracker: str = TRACKER_CONFIG,
        device: Optional[str] = None,
    ) -> None:
        from ultralytics import YOLO  # lazy import so module loads without GPU

        self._model = YOLO(model_path)
        self._conf = conf
        self._iou = iou
        self._vehicle_classes = set(vehicle_classes)
        self._tracker = tracker
        self._device = device or "cpu"
        self._frame_idx: int = 0
        logger.info("YOLODetector loaded model=%s device=%s", model_path, self._device)

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Run YOLO tracking on one BGR frame and return DetectionResults."""
        results = self._model.track(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            classes=list(self._vehicle_classes),
            tracker=self._tracker,
            persist=True,
            device=self._device,
            verbose=False,
        )

        detections: list[DetectionResult] = []
        if not results or results[0].boxes is None:
            self._frame_idx += 1
            return detections

        res = results[0]
        boxes = res.boxes
        masks_data = res.masks  # may be None for detection-only models

        for i, box in enumerate(boxes):
            track_id_tensor = box.id
            if track_id_tensor is None:
                continue  # untracked detection

            track_id = int(track_id_tensor.item())
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = res.names.get(cls_id, str(cls_id))

            mask: Optional[np.ndarray] = None
            if masks_data is not None and i < len(masks_data):
                mask_tensor = masks_data[i].data
                if mask_tensor is not None:
                    mask = mask_tensor.squeeze().cpu().numpy().astype(bool)

            detections.append(
                DetectionResult(
                    frame_idx=self._frame_idx,
                    track_id=track_id,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=conf,
                    class_name=class_name,
                    mask=mask,
                )
            )

        self._frame_idx += 1
        return detections
