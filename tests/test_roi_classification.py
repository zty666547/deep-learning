import numpy as np
import pytest
import torch

from microc_foundation.roi_classification import (
    RegionShapeCNN,
    background_gradient_ratio,
    density_preserving_shuffle,
    input_gradient_map,
    paired_logits,
    region_grid,
    roi_shape_input,
    train_roi_classifier,
)


def test_class_blind_roi_geometry_and_clipping():
    grid, mask, extent = region_grid(0, 20480, 9000, 11000)
    assert extent == (8500, 11500) and mask.shape == (64, 64)
    assert np.array_equal(mask, mask.T) and grid.shape == (64, 64, 2)
    _, _, clipped = region_grid(0, 20480, 100, 18000)
    assert clipped == (0, 20480)
    with pytest.raises(ValueError):
        region_grid(0, 20480, -10, 11000)


def test_shape_transform_removes_pure_distance_and_global_log_level():
    torch.set_num_threads(1)
    row, col = np.indices((256, 256))
    distance = np.abs(row-col)
    maps = np.stack((5/(1+distance), 7/(1+distance)))
    grid, _, _ = region_grid(0, 20480, 9000, 11000)
    np.testing.assert_allclose(roi_shape_input(maps, grid), 0, atol=1e-6)
    rng = np.random.default_rng(2)
    texture = rng.random((2, 256, 256)); texture = (texture+texture.transpose(0, 2, 1))/2
    np.testing.assert_allclose(roi_shape_input(maps+texture, grid),
                               roi_shape_input(maps+texture+3, grid), atol=1e-5)


def test_same_centre_different_scope_does_not_force_identical_inputs():
    rng = np.random.default_rng(2)
    maps = rng.random((2, 256, 256)); maps = (maps+maps.transpose(0, 2, 1))/2
    a, _, _ = region_grid(0, 20480, 9000, 11000)
    b, _, _ = region_grid(0, 20480, 5000, 15000)
    assert not np.allclose(roi_shape_input(maps, a), roi_shape_input(maps, b))


def test_shuffle_preserves_every_offset_histogram_and_protected_pixels():
    rng = np.random.default_rng(1)
    maps = rng.normal(size=(2, 64, 64)).astype("float32")
    maps = (maps+maps.transpose(0, 2, 1))/2
    _, mask, _ = region_grid(0, 20480, 9000, 11000)
    shuffled = density_preserving_shuffle(maps, ~mask, seed=5)
    np.testing.assert_array_equal(shuffled[:, mask], maps[:, mask])
    assert np.array_equal(shuffled, shuffled.transpose(0, 2, 1))
    for offset in range(64):
        np.testing.assert_array_equal(np.sort(np.diagonal(shuffled, offset, 1, 2)),
                                       np.sort(np.diagonal(maps, offset, 1, 2)))


def test_background_gradient_penalty_is_differentiable_and_unmasked():
    torch.set_num_threads(1)
    model = RegionShapeCNN()
    x = torch.randn(3, 1, 64, 64, requires_grad=True)
    masks = torch.zeros(3, 64, 64, dtype=torch.bool); masks[:, 20:44, 20:44] = True
    ratio = background_gradient_ratio(model(x), x, torch.arange(3), masks)
    assert 0 <= ratio.item() <= 1
    ratio.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    saliency = input_gradient_map(model.eval(), np.ones((2, 64, 64), "float32"), 0)
    assert saliency.shape == (64, 64) and (saliency >= 0).all()
    assert saliency[~masks[0].numpy()].sum() > 0  # No display mask manufactures focus.


def test_training_determinism_and_inference_batch_independence():
    torch.set_num_threads(1)
    maps = np.random.default_rng(3).normal(size=(6, 2, 64, 64)).astype("float32")
    labels = np.array([0, 1, 2, 0, 1, 2])
    masks = np.zeros((6, 64, 64), bool); masks[:, 10:54, 10:54] = True
    a, random, history, meta = train_roi_classifier(maps, labels, masks, maps, labels,
        seed=3, gradient_penalty=.1, epochs=1)
    b, _, same, _ = train_roi_classifier(maps, labels, masks, maps, labels,
        seed=3, gradient_penalty=.1, epochs=1)
    assert history == same and meta["best_epoch"] == 1
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    assert any(not torch.equal(x, y) for x, y in zip(a.parameters(), random.parameters()))
    logits = paired_logits(a, maps)
    assert logits.shape == (6, 2, 3)
    np.testing.assert_allclose(logits[:2], paired_logits(a, maps[:2]), atol=1e-6)
