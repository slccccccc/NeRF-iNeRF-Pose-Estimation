# Third-Party Notices

## Retained adapted component

The NeRF core is adapted from:

- <https://github.com/yenchenlin/nerf-pytorch>

The upstream project is MIT licensed. Its copyright and license terms remain
applicable to adapted portions of the NeRF core. This repository does not
bundle the upstream repository as a dependency.

The differentiable pose-refinement module under `src/fast_pose/` is an
independent implementation. The MIT-licensed `silvery107/fast-inerf` project
was used as related reference work; its copyright notice is not a substitute
for the licenses of the upstream iNeRF and PoseCNN components it references.

## Removed components

The public release does not contain LLFF data loading, iNeRF, PoseCNN,
implicit-depth augmentation, or PROPS-Pose dataset code. These components were
removed because either GPL-3.0 obligations or an unconfirmed upstream license
made a simple repository-wide MIT release inappropriate.

## Runtime dependencies

The Blender-only NeRF and camera-geometry tools use packages distributed under
their own licenses, including PyTorch (BSD 3-Clause), NumPy (BSD 3-Clause),
OpenCV (Apache-2.0), imageio (BSD-2-Clause), configargparse (MIT), tqdm
(MPL-2.0), and Matplotlib (Matplotlib license). These packages are installed
by the user and are not bundled into this repository. Consult the installed
distribution for the exact license text and version-specific notices.

## Data and media

No NeRF, LLFF, PROPS-Pose, or other external dataset is redistributed. The
`examples/` directory contains input-format instructions only. Users must
download datasets from their original sources and comply with their terms.
