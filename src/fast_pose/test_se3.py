import torch

from .se3 import se3_exp


def test_zero_twist_is_identity():
    matrix = se3_exp(torch.zeros(6))
    assert torch.allclose(matrix, torch.eye(4), atol=1e-6)


def test_twist_has_finite_gradients():
    twist = torch.tensor([1e-5, -2e-5, 3e-5, 0.1, -0.2, 0.3], requires_grad=True)
    loss = se3_exp(twist).square().sum()
    loss.backward()
    assert torch.isfinite(twist.grad).all()
