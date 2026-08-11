"""Load Blender-style NeRF datasets without external LLFF code."""

from __future__ import annotations

import json
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


def config_parser():
    """Build the command-line parser for the Blender-only NeRF runner."""
    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, default=None)
    parser.add_argument("--expname", type=str, default="blender_experiment")
    parser.add_argument("--basedir", type=str, default="./logs/")
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--netdepth", type=int, default=8)
    parser.add_argument("--netwidth", type=int, default=256)
    parser.add_argument("--netdepth_fine", type=int, default=8)
    parser.add_argument("--netwidth_fine", type=int, default=256)
    parser.add_argument("--N_rand", type=int, default=1024)
    parser.add_argument("--lrate", type=float, default=5e-4)
    parser.add_argument("--lrate_decay", type=int, default=250)
    parser.add_argument("--chunk", type=int, default=32768)
    parser.add_argument("--netchunk", type=int, default=65536)
    parser.add_argument("--no_batching", action="store_true")
    parser.add_argument("--no_reload", action="store_true")
    parser.add_argument("--ft_path", type=str, default=None)
    parser.add_argument("--N_samples", type=int, default=64)
    parser.add_argument("--N_importance", type=int, default=0)
    parser.add_argument("--perturb", type=float, default=1.0)
    parser.add_argument("--use_viewdirs", action="store_true")
    parser.add_argument("--i_embed", type=int, default=0)
    parser.add_argument("--multires", type=int, default=10)
    parser.add_argument("--multires_views", type=int, default=4)
    parser.add_argument("--raw_noise_std", type=float, default=0.0)
    parser.add_argument("--render_only", action="store_true")
    parser.add_argument("--render_test", action="store_true")
    parser.add_argument("--render_factor", type=int, default=0)
    parser.add_argument("--precrop_iters", type=int, default=0)
    parser.add_argument("--precrop_frac", type=float, default=0.5)
    parser.add_argument("--testskip", type=int, default=1)
    parser.add_argument("--white_bkgd", action="store_true")
    parser.add_argument("--half_res", action="store_true")
    parser.add_argument("--dataset_type", choices=["blender"], default="blender")
    parser.add_argument("--no_ndc", action="store_true", default=True)
    parser.add_argument("--lindisp", action="store_true")
    parser.add_argument("--i_print", type=int, default=100)
    parser.add_argument("--i_img", type=int, default=500)
    parser.add_argument("--i_weights", type=int, default=10000)
    parser.add_argument("--i_testset", type=int, default=50000)
    parser.add_argument("--i_video", type=int, default=50000)
    return parser


def pose_spherical(theta: float, phi: float, radius: float) -> torch.Tensor:
    """Create a camera pose on a spherical rendering path."""
    theta_rad, phi_rad = np.deg2rad(theta), np.deg2rad(phi)
    trans = torch.eye(4)
    trans[2, 3] = radius
    rot_phi = torch.tensor(
        [[1, 0, 0, 0], [0, np.cos(phi_rad), -np.sin(phi_rad), 0], [0, np.sin(phi_rad), np.cos(phi_rad), 0], [0, 0, 0, 1]],
        dtype=torch.float32,
    )
    rot_theta = torch.tensor(
        [[np.cos(theta_rad), 0, -np.sin(theta_rad), 0], [0, 1, 0, 0], [np.sin(theta_rad), 0, np.cos(theta_rad), 0], [0, 0, 0, 1]],
        dtype=torch.float32,
    )
    camera_axes = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32)
    return camera_axes @ rot_theta @ rot_phi @ trans


def _read_split(root: str, split: str, skip: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    with open(os.path.join(root, f"transforms_{split}.json"), "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    images, poses = [], []
    frames = metadata["frames"] if split == "train" or skip == 0 else metadata["frames"][::skip]
    for frame in frames:
        image_path = os.path.join(root, frame["file_path"] + ".png")
        image = imageio.imread(image_path).astype(np.float32) / 255.0
        images.append(image)
        poses.append(np.asarray(frame["transform_matrix"], dtype=np.float32))
    return images, poses


def load_blender_data(root: str, half_res: bool = False, testskip: int = 1):
    """Load Blender JSON metadata and return images, poses, and split indices."""
    all_images, all_poses, split_indices, count = [], [], [], 0
    metadata = None
    for split in ("train", "val", "test"):
        images, poses = _read_split(root, split, testskip)
        all_images.extend(images)
        all_poses.extend(poses)
        split_indices.append(np.arange(count, count + len(images)))
        count += len(images)
        with open(os.path.join(root, f"transforms_{split}.json"), "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    images = np.asarray(all_images, dtype=np.float32)
    poses = np.asarray(all_poses, dtype=np.float32)
    height, width = images[0].shape[:2]
    focal = 0.5 * width / np.tan(0.5 * float(metadata["camera_angle_x"]))
    if half_res:
        height //= 2
        width //= 2
        focal /= 2.0
        images = np.asarray([cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) for image in images], dtype=np.float32)
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180.0, 180.0, 40, endpoint=False)])
    return images, poses, render_poses, [height, width, focal], split_indices
