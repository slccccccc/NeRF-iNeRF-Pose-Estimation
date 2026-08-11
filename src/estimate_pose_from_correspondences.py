"""Estimate an object-to-camera pose from matched 2D-3D points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - reported when a command is executed
    cv2 = None


def load_points(path: Path, dimensions: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        points = np.load(path)
    else:
        points = np.loadtxt(path, delimiter=",")
    points = np.asarray(points, dtype=np.float64).reshape(-1, dimensions)
    if len(points) < 4:
        raise ValueError("At least four correspondences are required.")
    return points


def estimate_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    reprojection_threshold: float,
    confidence: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if len(object_points) != len(image_points):
        raise ValueError("2D and 3D point counts must match.")
    if len(object_points) < 6:
        flags = cv2.SOLVEPNP_IPPE if np.ptp(object_points[:, 2]) < 1e-8 else cv2.SOLVEPNP_AP3P
    else:
        flags = cv2.SOLVEPNP_EPNP
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=flags,
        reprojectionError=reprojection_threshold,
        confidence=confidence,
        iterationsCount=iterations,
    )
    if not success or inliers is None:
        raise RuntimeError("RANSAC could not find a valid pose.")
    inlier_ids = inliers.reshape(-1)
    if len(inlier_ids) >= 6:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[inlier_ids], image_points[inlier_ids], camera_matrix, distortion, rvec, tvec
        )
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    residual = projected.reshape(-1, 2) - image_points
    errors = np.sqrt(np.sum(residual * residual, axis=1))
    rmse = float(np.sqrt(np.mean(errors[inlier_ids] ** 2)))
    return rvec, tvec, inlier_ids, rmse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-points", type=Path, required=True, help="CSV or NPY file with N x 3 points.")
    parser.add_argument("--image-points", type=Path, required=True, help="CSV or NPY file with N x 2 points.")
    parser.add_argument("--calibration", type=Path, required=True, help="Camera calibration NPZ from calibrate_camera.py.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--reprojection-threshold", type=float, default=3.0)
    parser.add_argument("--confidence", type=float, default=0.999)
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()
    if cv2 is None:
        raise SystemExit("OpenCV is required. Install it with: python -m pip install opencv-python")

    calibration = np.load(args.calibration)
    object_points = load_points(args.object_points, 3)
    image_points = load_points(args.image_points, 2)
    rvec, tvec, inliers, rmse = estimate_pose(
        object_points,
        image_points,
        calibration["camera_matrix"],
        calibration["distortion"],
        args.reprojection_threshold,
        args.confidence,
        args.iterations,
    )
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec.reshape(3)
    result = {
        "convention": "T_camera_object maps object coordinates into camera coordinates",
        "inlier_indices": inliers.tolist(),
        "inlier_ratio": float(len(inliers) / len(object_points)),
        "inlier_reprojection_rmse_pixels": rmse,
        "rvec": rvec.reshape(3).tolist(),
        "tvec": tvec.reshape(3).tolist(),
        "T_camera_object": transform.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
