"""Refine one camera pose against a frozen Blender NeRF checkpoint."""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import torch

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from nerf import create_nerf
from src.fast_pose.optimizer import RefinementConfig, refine_pose
from utils.blender_loader import config_parser


def _load_pose(path: str | Path) -> torch.Tensor:
    pose = np.load(path) if str(path).lower().endswith(".npy") else np.asarray(__import__("json").loads(Path(path).read_text(encoding="utf-8")))
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape == (3, 4):
        pose = np.vstack([pose, [0, 0, 0, 1]])
    if pose.shape != (4, 4):
        raise ValueError("initial pose must be a 4x4 matrix or a 3x4 matrix")
    return torch.from_numpy(pose)


def main() -> None:
    parser = config_parser()
    parser.add_argument("--observed-image", required=True)
    parser.add_argument("--initial-pose", required=True)
    parser.add_argument("--focal", type=float, required=True)
    parser.add_argument("--pose-output", default="outputs/fast_pose/pose.json")
    parser.add_argument("--pose-scales", default="4,2,1")
    parser.add_argument("--pose-steps", default="120,160,240")
    parser.add_argument("--pose-samples", type=int, default=2048)
    parser.add_argument("--pose-lr", type=float, default=0.01)
    args = parser.parse_args()
    render_train, _, _, _, _ = create_nerf(args)
    config = RefinementConfig(
        scales=tuple(int(value) for value in args.pose_scales.split(",")),
        steps_per_scale=tuple(int(value) for value in args.pose_steps.split(",")),
        samples_per_step=args.pose_samples,
        learning_rate=args.pose_lr,
    )
    diagnostics = refine_pose(render_train, args.observed_image, _load_pose(args.initial_pose), args.focal, args.pose_output, config)
    print(f"Saved refined pose to {args.pose_output}; best loss={diagnostics['best_loss']:.6f}")


if __name__ == "__main__":
    main()
