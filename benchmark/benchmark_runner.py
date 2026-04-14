from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from speed.interfaces import SpeedEstimate

logger = logging.getLogger(__name__)


@dataclass
class MethodStats:
    method: str
    mean_kmh: float
    std_kmh: float
    mae_kmh: float
    rmse_kmh: float
    n_samples: int


class BenchmarkRunner:
    """Collects speed estimates from multiple methods and computes statistics."""

    def __init__(self, ground_truth_kmh: float) -> None:
        self._gt = ground_truth_kmh
        self._data: dict[str, list[float]] = {}

    def record(self, estimate: SpeedEstimate) -> None:
        self._data.setdefault(estimate.method, []).append(estimate.speed_kmh)

    def compute(self) -> list[MethodStats]:
        results: list[MethodStats] = []
        for method, speeds in self._data.items():
            vs = np.array(speeds)
            results.append(
                MethodStats(
                    method=method,
                    mean_kmh=float(vs.mean()),
                    std_kmh=float(vs.std()),
                    mae_kmh=float(np.abs(vs - self._gt).mean()),
                    rmse_kmh=float(np.sqrt(((vs - self._gt) ** 2).mean())),
                    n_samples=len(vs),
                )
            )
        results.sort(key=lambda s: s.rmse_kmh)
        return results

    def print_table(self) -> None:
        stats = self.compute()
        print("\n" + "=" * 72)
        print(f"{'BENCHMARK TABLE':^72}")
        print(f"{'Ground Truth: ' + str(self._gt) + ' km/h':^72}")
        print("=" * 72)
        header = f"{'Method':<18} {'Mean':>10} {'Std':>10} {'MAE':>10} {'RMSE':>10} {'N':>6}"
        print(header)
        print("-" * 72)
        for s in stats:
            print(
                f"{s.method:<18} {s.mean_kmh:>10.2f} {s.std_kmh:>10.2f} "
                f"{s.mae_kmh:>10.2f} {s.rmse_kmh:>10.2f} {s.n_samples:>6}"
            )
        print("=" * 72 + "\n")
