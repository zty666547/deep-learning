"""Math controls and tiny synthetic checkpoints, not biological evidence."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from microc_foundation.model import MicroCCNN
from microc_foundation.saliency_audit import (
    absolute_input_gradient,
    attribution_mass,
    audit_checkpoint_saliency,
    compare_attributions,
    fractional_top_weights,
)


def test_fixed_target_channel_mean_and_symmetry():
    class ToyModel(torch.nn.Module):
        def forward(self, values):
            sums = values.sum(dim=(1, 2, 3))
            return torch.stack((sums, 2 * sums), dim=1)

    inputs = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2)
    maps, logits = absolute_input_gradient(ToyModel(), inputs, torch.tensor([0, 1]))
    expected = inputs.mean(dim=1).numpy() * np.array([1, 2])[:, None, None]
    expected = (expected + expected.transpose(0, 2, 1)) / 2
    assert np.allclose(maps, expected)
    assert (logits.argmax(axis=1) == 1).all()  # sample 0 still targets class 0.
    assert inputs.grad is None


def test_mass_matches_area_for_uniform_map_and_zero_abstains():
    kwargs = {"window_start": 0, "window_end": 40, "annotation_start": 10,
              "annotation_end": 30, "minimum_offset": 1}
    result = attribution_mass(np.ones((4, 4)), **kwargs)
    assert result["near_diagonal_mass_fraction"] == 0.4
    assert result["annotation_area_fraction"] == 0.3
    assert result["annotation_mass_fraction"] == 0.3
    assert result["annotation_area_enrichment"] == 1
    assert attribution_mass(np.zeros((4, 4)), **kwargs)["annotation_mass_fraction"] is None
    with pytest.raises(ValueError, match="annotation/window"):
        attribution_mass(np.ones((4, 4)), **dict(kwargs, annotation_end=41))


def test_tie_aware_top_fraction_and_comparison_constants():
    assert fractional_top_weights(np.array([4, 3, 3, 0]), 0.5).tolist() == [1, 0.5, 0.5, 0]
    values = np.arange(100, dtype=float).reshape(10, 10)
    same = compare_attributions(values, values, minimum_offset=2)
    assert same["spearman"] == pytest.approx(1)
    assert same["top10_jaccard"] == 1
    inverse = compare_attributions(values, 99 - values, minimum_offset=2)
    assert inverse["spearman"] == pytest.approx(-1)
    assert inverse["top10_jaccard"] == 0
    assert compare_attributions(values, np.ones_like(values))["spearman"] is None
    with pytest.raises(ValueError, match="nonnegative"):
        compare_attributions(values, -values)


def test_tiny_all_sample_checkpoint_audit(tmp_path):
    torch.set_num_threads(1)
    matrices = np.random.default_rng(4).random((3, 2, 16, 16)).astype(np.float32)
    matrices = (matrices + matrices.transpose(0, 1, 3, 2)) / 2
    dataset = tmp_path / "dataset.npz"
    np.savez(dataset, matrices=matrices, split=np.array(["test"] * 3),
             labels=np.array(["CHID", "CHIN", "OPCID"]), structure_id=np.array(["z", "b", "a"]),
             chrom=np.array(["chr"] * 3), window_start=np.zeros(3, dtype=int),
             window_end=np.full(3, 160), annotation_start=np.full(3, 40),
             annotation_end=np.full(3, 80), pooled_bin_size=10)
    checkpoints = []
    for seed in (5, 6, 7):
        torch.manual_seed(seed)
        model = MicroCCNN(2, 3)
        path = tmp_path / f"model_{seed}.pt"
        torch.save({"model_state": model.state_dict(), "class_names": ["CHID", "CHIN", "OPCID"],
                    "channel_mean": [0.1, 0.2], "channel_std": [0.3, 0.4]}, path)
        checkpoints.append(str(path))
    result = audit_checkpoint_saliency(str(dataset), checkpoints, str(tmp_path / "out"),
                                      minimum_offset=2, batch_size=2)
    assert result["test_structures"] == 3
    assert result["example_structure_ids"] == ["z", "b", "a"]
    assert len(pd.read_csv(tmp_path / "out/per_sample_attribution.csv")) == 9
    assert len(pd.read_csv(tmp_path / "out/map_comparisons.csv")) == 18
    assert len(result["input_sha256"]) == 4
    assert result["comparisons"]["trained_vs_random"]["structure_n"] == 3
    assert len(pd.read_csv(tmp_path / "out/per_structure_comparisons.csv")) == 6
    with np.load(tmp_path / "out/attribution_maps.npz") as data:
        assert data["trained"].shape == (3, 3, 16, 16)
        assert np.allclose(data["trained"], data["trained"].transpose(0, 1, 3, 2))
    saved = json.loads((tmp_path / "out/summary.json").read_text())
    assert saved["comparisons"] == result["comparisons"]
