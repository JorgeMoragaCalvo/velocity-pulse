from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2


logger = logging.getLogger(__name__)

_clicked_points: list[tuple[float, float]] = []


def _mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicked_points.append((float(x), float(y)))
        logger.debug("Point clicked: (%d, %d)", x, y)


def pick_image_points(
    image_path: str | Path,
    n_points: int = 4,
    points_json_path: str | Path | None = None,
) -> list[tuple[float, float]]:
    """Open an OpenCV window; the user left-clicks n_points calibration points.

    If *points_json_path* is given the pixel coordinates are loaded from that
    JSON file (key ``"puntos"``) and the interactive click loop is skipped.
    The image is still shown so the user can visually confirm the points.
    """
    global _clicked_points
    _clicked_points = []

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    # ── Pre-loaded points path ──────────────────────────────────────────────
    if points_json_path is not None:
        with open(points_json_path) as f:
            data = json.load(f)
        loaded = [(float(p[0]), float(p[1])) for p in data["puntos"]]
        display = img.copy()
        for i, (px, py) in enumerate(loaded):
            cv2.circle(display, (int(px), int(py)), 8, (0, 255, 0), -1)
            cv2.putText(
                display,
                str(i + 1),
                (int(px) + 10, int(py) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        win = f"Loaded {len(loaded)} points from JSON — press any key to continue"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, display)
        print(f"\nLoaded {len(loaded)} pixel points from {points_json_path}. Press any key to continue.\n")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return loaded
    # ───────────────────────────────────────────────────────────────────────

    win = "Calibration — click cone base points (press Q when done)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _mouse_callback)

    print(f"\nClick {n_points} cone base points in the image. Press Q when done.\n")

    while True:
        display = img.copy()
        for i, (px, py) in enumerate(_clicked_points):
            cv2.circle(display, (int(px), int(py)), 8, (0, 255, 0), -1)
            cv2.putText(
                display,
                str(i + 1),
                (int(px) + 10, int(py) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        if len(_clicked_points) >= n_points:
            print(f"All {n_points} points selected. Press Q to confirm.")

    cv2.destroyAllWindows()

    if len(_clicked_points) < n_points:
        raise RuntimeError(f"Need at least {n_points} points, got {len(_clicked_points)}.")

    return list(_clicked_points)


def prompt_metric_points(n_points: int) -> list[tuple[float, float]]:
    """Ask the user to type real-world metric coordinates for each clicked point."""
    print(f"\nEnter real-world (X, Y) coordinates in metres for each of the {n_points} points.")
    print("Format: x,y  (e.g. 0,0)\n")
    metric: list[tuple[float, float]] = []
    for i in range(n_points):
        while True:
            raw = input(f"  Point {i + 1}: ").strip()
            try:
                x_str, y_str = raw.split(",")
                metric.append((float(x_str), float(y_str)))
                break
            except (ValueError, TypeError):
                print("  Invalid format — use x,y (e.g. 0,0)")
    return metric
