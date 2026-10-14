import numpy as np
import pytest
import torch

from microc_foundation.contrastive_texture import (
    TextureEncoder,
    encode_replicates,
    paired_contrastive_loss,
    texture_input,
    train_pair_encoder,
)


def test_texture_is_distance_envelope_invariant_and_symmetric():
    rng = np.random.default_rng(2)
    a = rng.uniform(0, .2, (128, 128))
    a = (a + a.T) / 2
    rows, columns = np.indices(a.shape)
    first = texture_input(np.expm1(a + 4/(abs(rows-columns)+1)))
    second = texture_input(np.expm1(a + 8/(abs(rows-columns)+1)))
    np.testing.assert_allclose(first, second, atol=1e-6)
    assert first.shape == (64, 64) and np.allclose(first, first.T)
    assert np.array_equal(texture_input(np.ones((128, 128))), np.zeros((64, 64)))


def test_ntxent_masks_self_pairs_and_preserves_pair_identity():
    a = torch.eye(3, requires_grad=True)
    identical = paired_contrastive_loss(a, a)
    shuffled = paired_contrastive_loss(a, torch.roll(a, 1, dims=0))
    assert identical < shuffled
    identical.backward()
    assert torch.isfinite(a.grad).all()
    collapsed = paired_contrastive_loss(torch.ones(3, 2), torch.ones(3, 2))
    assert float(collapsed) == pytest.approx(np.log(5))


def test_encoder_spatial_dimensions_and_no_query_statistics():
    torch.manual_seed(3)
    model = TextureEncoder().eval()
    x = np.random.default_rng(3).normal(size=(2, 2, 64, 64)).astype("float32")
    features = encode_replicates(model, x)
    assert features.shape == (2, 2, 128)
    extra = np.concatenate((x, np.full((1, 2, 64, 64), 10, dtype="float32")))
    np.testing.assert_allclose(features, encode_replicates(model, extra)[:2], atol=1e-6)


def test_train_only_api_deterministic_and_preserves_random_control():
    x = np.random.default_rng(4).normal(size=(4, 2, 64, 64)).astype("float32")
    torch.set_num_threads(1)
    trained, random, history = train_pair_encoder(x, seed=4, epochs=1)
    again, random_again, same = train_pair_encoder(x, seed=4, epochs=1)
    assert history == same and history[0]["training_windows"] == 4
    assert all(torch.equal(a, b) for a, b in zip(trained.parameters(), again.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(random.parameters(), random_again.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(trained.parameters(), random.parameters()))


def test_rejects_invalid_maps_and_single_pair_loss():
    with pytest.raises(ValueError):
        texture_input(np.ones((64, 64)))
    with pytest.raises(ValueError):
        paired_contrastive_loss(torch.ones(1, 2), torch.ones(1, 2))
    with pytest.raises(ValueError):
        train_pair_encoder(np.ones((1, 2, 64, 64)), seed=1)
