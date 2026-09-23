import pytest
import torch

from microc_foundation.training import FocalLoss


def test_focal_loss_is_finite_and_supports_backpropagation():
    logits = torch.tensor([[2.0, 0.0], [0.5, 1.0]], requires_grad=True)
    targets = torch.tensor([0, 1])
    loss = FocalLoss(alpha=torch.tensor([1.0, 2.0]), gamma=2.0)(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_focal_loss_rejects_negative_gamma():
    with pytest.raises(ValueError, match="non-negative"):
        FocalLoss(gamma=-1.0)
