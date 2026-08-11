"""Small, differentiable SE(3) utilities used by the pose optimizer."""

from __future__ import annotations

import torch


def _hat(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        (
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def se3_exp(twist: torch.Tensor) -> torch.Tensor:
    """Map a 6D twist ``[rotation, translation]`` to a 4x4 transform.

    The implementation uses Taylor expansions near zero, so gradients remain
    finite when optimization starts from a nearly correct pose.
    """
    if twist.shape[-1] != 6:
        raise ValueError("twist must have six values per transform")
    rotation = twist[..., :3]
    translation = twist[..., 3:]
    theta2 = (rotation * rotation).sum(dim=-1, keepdim=True)
    theta = theta2.sqrt()
    k = _hat(rotation)
    k2 = k @ k
    eps = torch.finfo(twist.dtype).eps * 100
    safe_theta = theta.clamp_min(eps)
    a_regular = torch.sin(safe_theta) / safe_theta
    b_regular = (1.0 - torch.cos(safe_theta)) / theta2.clamp_min(eps)
    c_regular = (safe_theta - torch.sin(safe_theta)) / (theta2 * safe_theta).clamp_min(eps)
    a = torch.where(theta2 < 1e-8, 1.0 - theta2 / 6.0, a_regular)
    b = torch.where(theta2 < 1e-8, 0.5 - theta2 / 24.0, b_regular)
    c = torch.where(theta2 < 1e-8, 1.0 / 6.0 - theta2 / 120.0, c_regular)
    eye = torch.eye(3, dtype=twist.dtype, device=twist.device).expand(k.shape)
    rotation_matrix = eye + a[..., None] * k + b[..., None] * k2
    v_matrix = eye + b[..., None] * k + c[..., None] * k2
    transformed_translation = (v_matrix @ translation[..., None]).squeeze(-1)
    result = torch.eye(4, dtype=twist.dtype, device=twist.device).expand(twist.shape[:-1] + (4, 4)).clone()
    result[..., :3, :3] = rotation_matrix
    result[..., :3, 3] = transformed_translation
    return result
