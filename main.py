"""
Velocity Pulse — CLI entry point
=================================
Usage examples:
  python main.py --method fusion --input TEST_VEL/
  python main.py --method centroid --input TEST_VEL/video.mp4
  python main.py --method all --input TEST_VEL/
  python main.py --calibrate --image data/calibration_reference.jpg
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("velocity_pulse")

METHODS_M1 = {"centroid", "optflow", "bbox", "all"}
METHODS_M2 = {"fusion"}
ALL_METHODS = METHODS_M1 | METHODS_M2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def _collect_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)


def _run_calibration(args: argparse.Namespace) -> None:
    from calibration.calibration_loader import CalibrationLoader
    from calibration.calibration_ui import pick_image_points, prompt_metric_points
    from calibration.homography_calibrator import HomographyCalibrator

    image_path = Path(args.image)
    n_points = args.n_points

    points_json = Path(args.points_json) if args.points_json else None

    print(f"\nCalibration mode: {image_path}")
    image_points = pick_image_points(image_path, n_points=n_points, points_json_path=points_json)
    metric_points = prompt_metric_points(len(image_points))

    calibrator = HomographyCalibrator()
    H = calibrator.calibrate(image_points, metric_points)

    loader = CalibrationLoader()
    loader.save(
        homography=H,
        image_points=image_points,
        metric_points=metric_points,
        reprojection_error=0.0,  # computed inside calibrator log
        fps=args.fps,
        frame_width=0,
        frame_height=0,
    )
    print(f"\nCalibration saved to {loader._path}")
    print("Homography matrix:")
    print(H)


def _run_pipeline(args: argparse.Namespace) -> None:
    from calibration.calibration_loader import CalibrationLoader
    from config.settings import OUTPUTS_DIR

    calib = CalibrationLoader().load()
    H = calib.H
    fps = args.fps or calib.fps or 30.0
    device = args.device

    input_path = Path(args.input)
    videos = _collect_videos(input_path)
    if not videos:
        logger.error("No video files found in: %s", input_path)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else OUTPUTS_DIR
    gt = args.ground_truth

    method = args.method.lower()

    for video in videos:
        logger.info("=== Processing: %s  method=%s ===", video.name, method)
        if method in METHODS_M2 or method == "fusion":
            from pipeline.module2_pipeline import Module2Pipeline
            pipe = Module2Pipeline(homography=H, fps=fps, device=device)
            pipe.run(video, output_dir, ground_truth_kmh=gt)
        else:
            from pipeline.module1_pipeline import Module1Pipeline
            pipe = Module1Pipeline(homography=H, fps=fps, method=method, device=device)
            pipe.run(video, output_dir, ground_truth_kmh=gt)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Velocity Pulse — Vehicle speed estimation from fixed cameras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--calibrate",
        action="store_true",
        help="Run interactive calibration to compute homography.",
    )
    mode.add_argument(
        "--method",
        choices=["centroid", "optflow", "bbox", "all", "fusion"],
        help="Speed estimation method to use.",
    )

    # Calibration args
    parser.add_argument("--image", type=str, help="Path to calibration reference image.")
    parser.add_argument("--n-points", type=int, default=4, dest="n_points",
                        help="Number of calibration points to pick (default: 4).")
    parser.add_argument("--points-json", type=str, default=None, dest="points_json",
                        help="JSON file with pre-captured pixel coords (key 'puntos'). "
                             "Skips manual clicking; only metric input is needed.")

    # Processing args
    parser.add_argument("--input", type=str, default="TEST_VEL",
                        help="Path to a video file or folder (default: TEST_VEL/).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: outputs/).")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override FPS (auto-detected from video if not set).")
    parser.add_argument("--ground-truth", type=float, default=None, dest="ground_truth",
                        help="Known speed in km/h for benchmark (e.g. 20 or 40).")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Inference device (default: cpu).")

    args = parser.parse_args()

    if args.calibrate:
        if not args.image:
            parser.error("--calibrate requires --image <path>")
        _run_calibration(args)
    else:
        _run_pipeline(args)


if __name__ == "__main__":
    main()
