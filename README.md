# NeRF-iNeRF-Pose-Estimation

PyTorch implementations and experiments for neural radiance field
reconstruction, inverse NeRF camera-pose optimization, and PoseCNN-based pose
initialization. The release also includes a robust classical-camera geometry
baseline for calibration and known 2D-3D correspondences.

## Public-release improvements

- Data and checkpoint paths are supplied through command-line arguments.
- The launchers no longer force CUDA as PyTorch's default tensor type.
- Chessboard calibration uses ordered detections rather than the four strongest
  image corners.
- Pose estimation uses `solvePnPRansac`, LM refinement, and reports inlier
  ratio and reprojection RMSE.
- Video extraction stops before writing an invalid frame and uses stable names.

## Classical pose geometry

The original corner script was not a reliable calibration method: selecting
four points with `goodFeaturesToTrack` does not establish their physical order
or guarantee that they belong to one rigid planar target. The public baseline
requires an ordered chessboard pattern for calibration:

```bash
python src/calibrate_camera.py \
  --images "examples/calibration/*.png" \
  --pattern-cols 9 \
  --pattern-rows 6 \
  --square-size 0.024 \
  --output outputs/camera_calibration.npz
```

For known 2D-3D correspondences:

```bash
python src/estimate_pose_from_correspondences.py \
  --object-points examples/object_points.csv \
  --image-points examples/image_points.csv \
  --calibration outputs/camera_calibration.npz \
  --output outputs/pose.json
```

The reported transform maps object coordinates into camera coordinates. Check
the inlier ratio and pixel reprojection RMSE before using the pose downstream.

## NeRF and iNeRF

Supply a public NeRF synthetic or LLFF dataset and an output directory through
the configuration files or command line:

```bash
python scripts/run_nerf.py --config configs/nerf/lego.txt
python scripts/run_inerf.py --config configs/inerf/lego.txt
```

PoseCNN requires the separately distributed PROPS-Pose dataset, which is not
included:

```bash
python scripts/run_posecnn.py --train --data-dir path/to/PROPS-Pose-Dataset
python scripts/run_posecnn.py --eval --data-dir path/to/PROPS-Pose-Dataset
```

## Attribution

The NeRF, iNeRF, and PoseCNN components are adaptations of the projects below:

- [yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)
- [salykovaa/inerf](https://github.com/salykovaa/inerf)
- [DeepRob PoseCNN project](https://deeprob.org/projects/project4/)

Their original notices and licenses must be preserved when the code is
redistributed. The added camera-calibration, robust PnP, and video-extraction
utilities are provided under the repository license.
