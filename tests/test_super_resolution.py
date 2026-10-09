import numpy as np
import pytest
import torch

from microc_foundation.super_resolution import (
    ContactSuperResolutionCNN,
    global_ssim,
    psnr,
)


def test_super_resolution_model_doubles_spatial_shape():
    result = ContactSuperResolutionCNN(2)(torch.zeros(3, 2, 8, 8))
    assert result.shape == (3, 2, 16, 16)


def test_image_metrics_are_perfect_for_identical_maps():
    values = np.eye(8)
    assert psnr(values, values) == float("inf")
    assert global_ssim(values, values) == pytest.approx(1.0)


def test_image_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        psnr(np.zeros((2, 2)), np.zeros((3, 3)))
