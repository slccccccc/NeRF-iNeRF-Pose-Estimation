"""Calibrate a pinhole camera from ordered chessboard observations.

The calibration routine intentionally uses chessboard geometry rather than
selecting the strongest image corners. A detected corner must have a known
board coordinate before it can be used for calibration or pose estimation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - reported when a command is executed
    cv2 = None


def build_board_points(pattern_size: tuple[int, int], square_size: float) -> np.ndarray:
    """Return chessboard points in the board coordinate system."""
    cols, rows = pattern_size
    points = np.zeros((cols * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return points * float(square_size)


def detect_ordered_corners(
    image: np.ndarray, pattern_size: tuple[int, int]
) -> np.ndarray | None:
    """Detect and sub-pixel-refine ordered inner chessboard corners."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cols, rows = pattern_size
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray, (cols, rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE
        )
    else:
        found, corners = cv2.findChessboardCorners(
            gray,
            (cols, rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK,
        )
    if not found or corners is None:
        return None
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        40,
        1e-3,
    )
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)


def reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion
    )
    residual = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def calibrate(
    image_paths: list[Path],
    pattern_size: tuple[int, int],
    square_size: float,
    reject_outliers: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    board = build_board_points(pattern_size, square_size)
    object_sets: list[np.ndarray] = []
    image_sets: list[np.ndarray] = []
    used_paths: list[str] = []
    image_size: tuple[int, int] | None = None

    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        corners = detect_ordered_corners(image, pattern_size)
        if corners is None:
            continue
        current_size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = current_size
        if current_size != image_size:
            raise ValueError("All calibration images must have the same size.")
        object_sets.append(board.copy())
        image_sets.append(corners)
        used_paths.append(str(path))

    if image_size is None or len(object_sets) < 3:
        raise RuntimeError(
            "At least three images with detected chessboard corners are required."
        )

    def run_calibration(indices: list[int]):
        return cv2.calibrateCamera(
            [object_sets[i] for i in indices],
            [image_sets[i] for i in indices],
            image_size,
            None,
            None,
        )

    indices = list(range(len(object_sets)))
    rms, camera_matrix, distortion, rvecs, tvecs = run_calibration(indices)
    errors = [
        reprojection_error(
            object_sets[i], image_sets[i], rvecs[pos], tvecs[pos], camera_matrix, distortion
        )
        for pos, i in enumerate(indices)
    ]

    rejected: list[str] = []
    if reject_outliers and len(indices) >= 5:
        median = float(np.median(errors))
        mad = float(np.median(np.abs(np.asarray(errors) - median)))
        threshold = max(2.0, median + 3.0 * max(mad, 1e-6))
        keep = [i for i, error in zip(indices, errors) if error <= threshold]
        if len(keep) >= 3 and len(keep) < len(indices):
            rejected = [used_paths[i] for i in indices if i not in keep]
            indices = keep
            rms, camera_matrix, distortion, rvecs, tvecs = run_calibration(indices)
            errors = [
                reprojection_error(
                    object_sets[i], image_sets[i], rvecs[pos], tvecs[pos], camera_matrix, distortion
                )
                for pos, i in enumerate(indices)
            ]

    report = {
        "image_size": list(image_size),
        "pattern_size": list(pattern_size),
        "square_size": square_size,
        "views_detected": len(object_sets),
        "views_used": len(indices),
        "rejected_views": rejected,
        "calibration_rms": float(rms),
        "view_reprojection_rmse": dict(
            zip([used_paths[i] for i in indices], [float(x) for x in errors])
        ),
    }
    return camera_matrix, distortion, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Calibration image glob.")
    parser.add_argument("--pattern-cols", type=int, required=True, help="Inner corners per row.")
    parser.add_argument("--pattern-rows", type=int, required=True, help="Inner corners per column.")
    parser.add_argument("--square-size", type=float, required=True, help="Physical square size.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path.")
    parser.add_argument("--report", type=Path, help="Optional JSON report path.")
    parser.add_argument("--keep-outliers", action="store_true", help="Disable robust view rejection.")
    args = parser.parse_args()
    if cv2 is None:
        raise SystemExit("OpenCV is required. Install it with: python -m pip install opencv-python")

    paths = sorted(Path().glob(str(args.images))) if not args.images.is_absolute() else sorted(args.images.parent.glob(args.images.name))
    camera_matrix, distortion, report = calibrate(
        paths,
        (args.pattern_cols, args.pattern_rows),
        args.square_size,
        reject_outliers=not args.keep_outliers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        camera_matrix=camera_matrix,
        distortion=distortion,
    )
    report_path = args.report or args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
