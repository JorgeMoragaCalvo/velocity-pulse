from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TEST_VEL_DIR = PROJECT_ROOT / "TEST_VEL"

CALIBRATION_FILE = CONFIG_DIR / "calibration.json"

# ---------------------------------------------------------------------------
# Detection / tracking
# ---------------------------------------------------------------------------
YOLO_MODEL: str = os.environ.get("YOLO_MODEL", "yolov8n-seg.pt")
YOLO_CONF_THRESHOLD: float = 0.40
YOLO_IOU_THRESHOLD: float = 0.45
YOLO_VEHICLE_CLASSES: tuple[int, ...] = (2, 5, 7)  # car, bus, truck (COCO)
TRACKER_CONFIG: str = "botsort.yaml"
MAX_LOST_FRAMES: int = 30          # frames before track is dropped
TRACK_HISTORY_LEN: int = 120       # max positions kept per track

# ---------------------------------------------------------------------------
# Speed estimation
# ---------------------------------------------------------------------------
FPS_DEFAULT: float = 30.0
KALMAN_PROCESS_NOISE: float = 1e-4
KALMAN_MEAS_NOISE: float = 1e-2
SPEED_SMOOTHING_WINDOW: int = 15   # Savitzky-Golay window (odd)
SPEED_SMOOTHING_POLY: int = 2
MIN_TRACK_FRAMES: int = 5          # frames before speed is reported
OPTICAL_FLOW_MAX_CORNERS: int = 50
OPTICAL_FLOW_QUALITY: float = 0.01
OPTICAL_FLOW_MIN_DIST: float = 7.0
OPTICAL_FLOW_WIN_SIZE: tuple[int, int] = (21, 21)

# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------
MIDAS_MODEL: str = os.environ.get("MIDAS_MODEL", "MiDaS_small")
DEPTH_INFERENCE_INTERVAL: int = 3  # run depth every N frames

# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
FUSION_WARMUP_FRAMES: int = 30     # frames before variance weights stabilize
FUSION_MAX_DEPTH_WEIGHT: float = 0.40

# ---------------------------------------------------------------------------
# Output / visualisation
# ---------------------------------------------------------------------------
OSD_FONT_SCALE: float = 0.65
OSD_THICKNESS: int = 2
OSD_MASK_ALPHA: float = 0.35
OSD_TRACE_LEN: int = 60            # trajectory points to draw
SPEEDOMETER_MAX_KMH: float = 80.0
VIDEO_CODEC: str = "mp4v"

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
GROUND_TRUTH_SPEEDS_KMH: tuple[float, ...] = (20.0, 40.0)
