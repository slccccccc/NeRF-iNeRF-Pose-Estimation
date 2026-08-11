"""Coarse-to-fine differentiable camera-pose refinement."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from nerf import device as nerf_device
from utils.render_utils import get_rays, render

from .se3 import se3_exp


@dataclass
class RefinementConfig:
    scales: tuple[int, ...] = (4, 2, 1)
    steps_per_scale: tuple[int, ...] = (120, 160, 240)
    samples_per_step: int = 2048
    learning_rate: float = 0.01
    near: float = 2.0
    far: float = 6.0
    chunk: int = 32768
    robust_epsilon: float = 0.01
    patience: int = 35
    min_delta: float = 1e-5
    seed: int = 42


def _load_image(path: str | Path, device: torch.device) -> torch.Tensor:
    import imageio.v2 as imageio

    image = imageio.imread(path).astype(np.float32) / 255.0
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    image = image[..., :3]
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def _sample_indices(height: int, width: int, count: int, generator: torch.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    total = height * width
    count = min(total, count)
    linear = torch.randperm(total, generator=generator, device=device)[:count]
    return torch.div(linear, width, rounding_mode="floor"), linear.remainder(width)


def _charbonnier(prediction: torch.Tensor, target: torch.Tensor, epsilon: float) -> torch.Tensor:
    residual = prediction - target
    return torch.sqrt(residual.square() + epsilon * epsilon).mean()


def _render_pixels(pose: torch.Tensor, target: torch.Tensor, render_kwargs: dict, focal: float, config: RefinementConfig, rows: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    _, _, height, width = target.shape
    rays_o, rays_d = get_rays(height, width, focal, pose[:3, :4])
    selected_rays = (rays_o[rows, columns], rays_d[rows, columns])
    rgb, _, _, _ = render(height, width, focal, chunk=config.chunk, rays=selected_rays, c2w=None, ndc=False, near=config.near, far=config.far, **render_kwargs)
    return rgb


class PoseRefiner:
    """Optimize a camera pose while keeping the NeRF parameters frozen."""

    def __init__(self, render_kwargs: dict, device: torch.device | None = None):
        self.render_kwargs = {key: value for key, value in render_kwargs.items() if key not in {"perturb", "raw_noise_std", "ndc", "near", "far"}}
        self.render_kwargs["perturb"] = 0.0
        self.render_kwargs["raw_noise_std"] = 0.0
        self.device = device or nerf_device

    def refine(self, image: torch.Tensor, initial_pose: torch.Tensor, focal: float, config: RefinementConfig | None = None) -> tuple[torch.Tensor, dict]:
        config = config or RefinementConfig()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device, dtype=torch.float32)
        initial_pose = initial_pose.to(self.device, dtype=torch.float32)
        if initial_pose.shape != (4, 4):
            raise ValueError("initial_pose must have shape (4, 4)")
        if len(config.scales) != len(config.steps_per_scale):
            raise ValueError("scales and steps_per_scale must have equal length")
        torch.manual_seed(config.seed)
        generator = torch.Generator(device=self.device).manual_seed(config.seed)
        twist = torch.zeros(6, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([twist], lr=config.learning_rate)
        history: list[dict] = []
        best_loss = float("inf")
        stale = 0
        for scale, steps in zip(config.scales, config.steps_per_scale):
            target = functional.interpolate(image, scale_factor=1.0 / scale, mode="area")
            height, width = target.shape[-2:]
            stage_best = float("inf")
            for step in range(steps):
                rows, columns = _sample_indices(height, width, config.samples_per_step, generator, self.device)
                pose = se3_exp(twist) @ initial_pose
                prediction = _render_pixels(pose, target, self.render_kwargs, focal / scale, config, rows, columns)
                target_pixels = target[0, :, rows, columns].transpose(0, 1)
                loss = _charbonnier(prediction, target_pixels, config.robust_epsilon)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([twist], 1.0)
                optimizer.step()
                loss_value = float(loss.detach().cpu())
                stage_best = min(stage_best, loss_value)
                if loss_value + config.min_delta < best_loss:
                    best_loss = loss_value
                    stale = 0
                else:
                    stale += 1
                if step % 10 == 0 or step == steps - 1:
                    history.append({"scale": scale, "step": step, "loss": loss_value, "rotation_norm": float(twist[:3].detach().norm().cpu()), "translation_norm": float(twist[3:].detach().norm().cpu())})
                if stale >= config.patience:
                    break
            history.append({"scale": scale, "stage_best_loss": stage_best, "steps_completed": step + 1})
            stale = 0
        final_pose = (se3_exp(twist.detach()) @ initial_pose).detach()
        diagnostics = {"config": asdict(config), "best_loss": best_loss, "final_twist": twist.detach().cpu().tolist(), "history": history}
        return final_pose, diagnostics


def refine_pose(render_kwargs: dict, image_path: str | Path, initial_pose: torch.Tensor, focal: float, output_path: str | Path, config: RefinementConfig | None = None) -> dict:
    """Load an observed image, refine its pose, and write JSON diagnostics."""
    device = nerf_device
    refiner = PoseRefiner(render_kwargs, device=device)
    image = _load_image(image_path, device)
    final_pose, diagnostics = refiner.refine(image, initial_pose, focal, config)
    diagnostics["initial_pose"] = initial_pose.cpu().tolist()
    diagnostics["final_pose"] = final_pose.cpu().tolist()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return diagnostics
