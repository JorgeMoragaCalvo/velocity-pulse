# Velocity Pulse

**Vehicle speed estimation for industrial and mining environments.**

Velocity Pulse uses fixed cameras with traffic cones as metric reference points to measure vehicle speed from video. It produces on-screen display (OSD) videos and multi-page PDF reports with speed, acceleration, and benchmark metrics.

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![Docker](https://img.shields.io/badge/docker-supported-blue)

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Calibration](#calibration)
6. [Running the Pipeline](#running-the-pipeline)
7. [CLI Reference](#cli-reference)
8. [Output Files](#output-files)
9. [Configuration](#configuration)
10. [Project Structure](#project-structure)
11. [Architecture](#architecture)

---

## Overview

Velocity Pulse processes video files recorded by fixed cameras to estimate vehicle speeds. The system does not require radar or GPS — it computes real-world distances by projecting pixel coordinates through a homography matrix calibrated from ground-plane reference points (traffic cone bases).

**Key capabilities:**
- Three independent speed estimation methods (centroid, optical flow, bounding box)
- Optional monocular depth fusion via MiDaS for improved accuracy
- Automatic vehicle detection and multi-object tracking (YOLOv8 + BoT-SORT)
- OSD video output with speed overlay and trajectory trace
- 4-page PDF report: speed vs. time, acceleration vs. time, tracking trace, benchmark table

---

## How It Works

```
Video frames
  └─► YoloDetector          — YOLOv8n-seg detects vehicles, returns bbox + mask + track_id
        └─► TrackManager    — rolling deque of detections per track
              └─► Speed estimators
                    ├── CentroidSpeed    — centroid projected through homography H
                    ├── OpticalFlowSpeed — sparse LK flow on mask corners, RANSAC filtered
                    └── BBoxSpeed        — bottom-center (cx, y2) projected through H
              └─► [fusion] MiDaS        — relative disparity → radial Z-velocity
              └─► [fusion] WeightedFusionMetaEstimator
                    — inverse-variance weighting + depth Z-correction + Kalman smoothing
              └─► OSD renderer + VideoWriter
              └─► ReportGenerator       — 4-page matplotlib PDF
```

**Speed formula:** `speed_kmh = distance_m × fps × 3.6`

Distance is computed in the ground-plane metric space after applying the homography H to pixel coordinates.

---

## Requirements

- Python 3.9 or higher
- pip dependencies (see `requirements.txt`)
- GPU optional — CPU inference is supported out of the box; swap the PyTorch wheel for CUDA if needed

**Core dependencies:**

| Package | Version | Purpose |
|---|---|---|
| opencv-python-headless | 4.9.0.80 | Video I/O, homography, optical flow |
| torch / torchvision | 2.2.2+cpu | Neural network inference |
| ultralytics | 8.1.47 | YOLOv8n-seg + BoT-SORT tracker |
| timm | 0.9.16 | MiDaS backbone |
| numpy / scipy | 2.4.4 / 1.13.0 | Numerics, Savitzky-Golay smoothing |
| matplotlib / Pillow | 3.10.8 / 12.2.0 | PDF report generation |

---

## Installation

### Local (pip)

```bash
git clone <repo-url>
cd Velocity
pip install -r requirements.txt
```

GPU users: replace the `torch==2.2.2+cpu` line in `requirements.txt` with the appropriate CUDA wheel from [pytorch.org](https://pytorch.org).

Model weights are downloaded automatically on first run:
- `yolov8n-seg.pt` — YOLOv8 nano segmentation model
- MiDaS small — monocular depth model

### Docker

```bash
docker compose build
```

Model weights are pre-downloaded at build time so the container works offline.

---

## Calibration

Calibration computes a homography matrix H that maps pixel coordinates to real-world ground-plane coordinates (metres). It must be run once before processing any video.

### Recommended — use pre-captured reference data

The repository ships with a reference image (`images/captura_21pts.jpg`) and pre-captured pixel coordinates (`data/mediaamp_data.json`). You only need to supply the real-world metric coordinates by reading them from `images/dimensions_image.png`.

```bash
python main.py --calibrate \
  --image images/captura_21pts.jpg \
  --points-json data/mediaamp_data.json
```

An OpenCV window opens showing the 21 numbered points. Press any key, then type the `x,y` metric coordinate for each point when prompted.

### Alternative — fully interactive

```bash
python main.py --calibrate --image <any_image> --n-points 4
```

Click cone base points in the window, press Q, then type metric coordinates for each point.

Both paths save the result to `config/calibration.json` (homography matrix H, point pairs, reprojection error, fps, resolution).

---

## Running the Pipeline

Place video files (`.mp4`, `.avi`, `.mov`, `.mkv`, `.m4v`) in the `TEST_VEL/` directory, then run:

```bash
# Single method on a directory
python main.py --method centroid --input TEST_VEL/
python main.py --method optflow  --input TEST_VEL/
python main.py --method bbox     --input TEST_VEL/

# All three Module 1 methods + benchmark table
python main.py --method all --input TEST_VEL/ --ground-truth 20

# Full fusion (Module 1 + MiDaS depth + meta-estimator)
python main.py --method fusion --input TEST_VEL/ --ground-truth 40

# Process a single file
python main.py --method fusion --input TEST_VEL/video.mp4

# GPU inference
python main.py --method fusion --input TEST_VEL/ --device cuda

# Override FPS and output directory
python main.py --method all --input TEST_VEL/ --fps 25 --output results/
```

### Docker

```bash
# Default: fusion method on /app/TEST_VEL
docker compose run velocity-pulse

# Override method
docker compose run velocity-pulse --method all --ground-truth 20
docker compose run velocity-pulse --method centroid --input /app/TEST_VEL
```

Volumes mounted: `./TEST_VEL → /app/TEST_VEL`, `./outputs → /app/outputs`, `./config → /app/config`.

---

## CLI Reference

### Calibration mode

| Argument | Description |
|---|---|
| `--calibrate` | Enable calibration mode |
| `--image PATH` | Path to calibration reference image (required) |
| `--n-points INT` | Number of points to click (default: 4) |
| `--points-json PATH` | JSON with pre-captured pixel coords — skips manual clicking |

### Processing mode

| Argument | Description |
|---|---|
| `--method` | `centroid` / `optflow` / `bbox` / `all` / `fusion` (required) |
| `--input PATH` | Video file or directory (default: `TEST_VEL/`) |
| `--output PATH` | Output directory (default: `outputs/`) |
| `--fps FLOAT` | Override FPS auto-detection |
| `--ground-truth FLOAT` | Known speed in km/h for benchmark metrics |
| `--device` | `cpu` / `cuda` / `mps` (default: `cpu`) |

---

## Output Files

For each input video `foo.mp4`:

| Method | Files produced |
|---|---|
| `centroid` | `foo_centroid.mp4` |
| `optflow` | `foo_optflow.mp4` |
| `bbox` | `foo_bbox.mp4` |
| `all` | `foo_centroid.mp4`, `foo_optflow.mp4`, `foo_bbox.mp4`, `foo_module1_report.pdf` |
| `fusion` | `foo_fusion.mp4`, `foo_module2_report.pdf` |

**PDF report pages:**
1. Speed vs. time
2. Acceleration vs. time
3. Tracking trace (pixel space + ground-plane)
4. Benchmark table — Mean, Std, MAE, RMSE vs. ground truth

---

## Configuration

All tunable parameters are in `config/settings.py`.

| Setting | Default | Effect |
|---|---|---|
| `YOLO_CONF_THRESHOLD` | 0.40 | Detection confidence gate |
| `YOLO_VEHICLE_CLASSES` | car, bus, truck | COCO class IDs to keep |
| `MAX_LOST_FRAMES` | 30 | Frames before a track is dropped |
| `SPEED_SMOOTHING_WINDOW` | 15 | Savitzky-Golay window size (must be odd) |
| `DEPTH_INFERENCE_INTERVAL` | 3 | Run MiDaS every N frames |
| `FUSION_WARMUP_FRAMES` | 30 | Frames before fusion weights stabilise |
| `FUSION_MAX_DEPTH_WEIGHT` | 0.40 | Max contribution from depth correction |
| `SPEEDOMETER_MAX_KMH` | 80.0 | OSD speedometer upper bound |

---

## Project Structure

```
Velocity/
├── main.py                  # CLI entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── calibration/             # Homography calibration
├── config/                  # settings.py + calibration.json (generated)
├── detection/               # YOLOv8n-seg wrapper
├── tracking/                # BoT-SORT track manager
├── speed/                   # CentroidSpeed, OpticalFlowSpeed, BBoxSpeed
├── depth/                   # MiDaS depth estimator + fallback
├── fusion/                  # Inverse-variance meta-estimator
├── output/                  # OSD renderer, VideoWriter, ReportGenerator
├── pipeline/                # Module1Pipeline, Module2Pipeline
├── benchmark/               # RMSE / MAE metrics
│
├── images/                  # Calibration reference images
│   ├── captura_21pts.jpg    # Reference image with 21 marked points
│   └── dimensions_image.png # Metric distances for calibration
│
├── data/                    # mediaamp_data.json (pre-captured pixel coords)
├── TEST_VEL/                # Place input videos here
└── outputs/                 # Generated OSD videos and PDF reports
```

---

## Architecture

### Pipelines

| Class | Module | Methods |
|---|---|---|
| `Module1Pipeline` | `pipeline/module1_pipeline.py` | centroid, optflow, bbox, all |
| `Module2Pipeline` | `pipeline/module2_pipeline.py` | fusion (extends Module 1) |

Both accept `video_path`, `output_dir`, `ground_truth_kmh` and produce OSD `.mp4` + PDF reports.

### Fusion (Module 2)

1. **Inverse-variance weighting** across the three Module 1 methods (30-frame rolling variance warmup)
2. **Depth Z-correction** — radial velocity from MiDaS disparity blends with lateral speed
3. **Kalman smoothing** applied to the fused result

### Interfaces

Each subsystem defines an ABC interface in `interfaces.py` with one or more concrete implementations, making components independently testable and swappable.
