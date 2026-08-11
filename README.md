# NeRF and Fast Pose Estimation Tools

This public release contains a Blender-style synthetic-data NeRF runner, a
clean-room differentiable camera-pose refiner, and independent
camera-calibration and robust PnP utilities. It is deliberately limited to
components with clear redistribution terms.

## Public-release improvements

- Data and checkpoint paths are supplied through command-line arguments.
- The launcher does not force CUDA as PyTorch's default tensor type.
- Blender JSON loading is explicit and self-contained.
- Chessboard calibration uses ordered detections rather than the four
  strongest image corners.
- Pose estimation uses `solvePnPRansac`, LM refinement, and reports inlier ratio
  and reprojection RMSE.
- Video extraction stops before writing an invalid frame and uses stable names.
- Pose refinement uses an SE(3) exponential map, coarse-to-fine image scales,
  random ray sampling, robust Charbonnier loss, gradient clipping, and early
  stopping with a JSON optimization trace.

## Camera calibration and pose estimation

Calibration from ordered chessboard observations:

```bash
python src/calibrate_camera.py \
  --images "examples/calibration/*.png" \
  --pattern-cols 9 \
  --pattern-rows 6 \
  --square-size 0.024 \
  --output outputs/camera_calibration.npz
```

Pose estimation from known 2D-3D correspondences:

```bash
python src/estimate_pose_from_correspondences.py \
  --object-points examples/object_points.csv \
  --image-points examples/image_points.csv \
  --calibration outputs/camera_calibration.npz \
  --output outputs/pose.json
```

The reported transform maps object coordinates into camera coordinates. Check
the inlier ratio and pixel reprojection RMSE before using the pose downstream.

## NeRF

The NeRF runner accepts a Blender-style synthetic dataset containing
`transforms_train.json`, `transforms_val.json`, and `transforms_test.json`:

```bash
python scripts/run_nerf.py \
  --datadir path/to/blender_dataset \
  --expname lego \
  --basedir outputs/nerf
```

## Fast pose refinement

After training a Blender NeRF, refine an initial camera pose against one
observed RGB image. The initial pose is a `.npy` or JSON 4x4 camera-to-world
matrix and the focal length is in pixels:

```bash
python scripts/run_fast_pose.py \
  --datadir path/to/blender_dataset \
  --basedir outputs/nerf \
  --expname lego \
  --ft_path outputs/nerf/lego/100000.tar \
  --observed-image path/to/observation.png \
  --initial-pose path/to/initial_pose.npy \
  --focal 555.0 \
  --pose-output outputs/fast_pose/pose.json
```

The optimizer changes only the camera pose; NeRF parameters remain frozen.
The output includes the refined pose, final twist, per-scale losses, and
stopping diagnostics. A low photometric loss alone does not prove a unique
camera pose, so inspect the trace and evaluate on held-out views.

The public release does not include LLFF, iNeRF, PoseCNN, implicit-depth, or
PROPS-Pose code or datasets. Those components were removed because their
redistribution terms were not sufficiently clear for a repository-wide MIT
release.

## Attribution and licensing

The NeRF core follows the MIT-licensed implementation at
[yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch), and its
copyright and license terms remain applicable to adapted core files. The
camera-calibration, robust PnP, video-extraction, and Blender-only loader were
added or rewritten for this release.

See [`LICENSE`](LICENSE) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the complete scope.
