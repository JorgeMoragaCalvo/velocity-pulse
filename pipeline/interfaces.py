from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class IPipeline(ABC):
    @abstractmethod
    def run(self, input_path: Path, output_dir: Path, ground_truth_kmh: float | None = None) -> None:
        ...
