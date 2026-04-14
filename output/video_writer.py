from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from config.settings import VIDEO_CODEC

logger = logging.getLogger(__name__)


class VideoWriter:
    """Context-manager wrapper around cv2.VideoWriter."""

    def __init__(self, output_path: str | Path, fps: float, width: int, height: int) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        self._writer = cv2.VideoWriter(str(self._path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for {self._path}")
        logger.info("VideoWriter opened: %s  fps=%.1f  %dx%d", self._path, fps, width, height)

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()
        logger.info("VideoWriter released: %s", self._path)

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
