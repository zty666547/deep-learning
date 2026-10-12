"""Synthetic perturbation invariants only; not course experiment results."""

import numpy as np
import pandas as pd
import pytest
import torch

from microc_foundation.model import MicroCCNN
from microc_foundation.occlusion_audit import (
    audit_occlusion,
    distance_baseline,
    energy_ranked_selection,
    energy_strata,
    matched_random_selection,
    replace_symmetric_pairs,
    top_pair_mask,
)


def test_original_offset_median_and_symmetric_channel_replacement():
    values = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    values = (values + values.transpose(0, 2, 1)) / 2
    original = values.copy()
    baseline = distance_baseline(values)
    for channel in range(2):
        for offset in range(4):
            assert np.all(np.diag(baseline[channel], k=offset) ==
                          np.median(np.diag(values[channel], k=offset)))
    selected = np.zeros((4, 4), dtype=bool)
    selected[0, 2] = True
    result = replace_symmetric_pairs(values, baseline, selected)
    assert np.array_equal(values, original)
    assert np.array_equal(result, result.transpose(0, 2, 1))
    assert np.array_equal(result[:, 0, 2], baseline[:, 0, 2])
    unchanged = ~(selected | selected.T)
    assert np.array_equal(result[:, unchanged], values[:, unchanged])
    selected[2, 0] = True
    with pytest.raises(ValueError, match="strict-upper"):
        replace_symmetric_pairs(values, baseline, selected)


def test_top_mask_fixed_size_no_near_diagonal_and_constants_abstain():
    saliency = np.arange(64, dtype=float).reshape(8, 8)
    saliency = saliency + saliency.T
    mask = top_pair_mask(saliency, 0.2, minimum_offset=2)
    assert mask.sum() == np.ceil(21 * 0.2)
    assert not np.tril(mask, k=1).any()
    assert np.array_equal(mask, top_pair_mask(saliency, 0.2, minimum_offset=2))
    with pytest.raises(ValueError, match="constant"):
        top_pair_mask(np.zeros((8, 8)), 0.1)


def test_exact_distance_and_energy_strata_controls_include_full_pool():
    offsets = np.array([2] * 4 + [3] * 6)
    energy = np.array([1, 1, 3, 4, 0, 2, 4, 6, 8, 10], dtype=float)
    strata = energy_strata(offsets, energy)
    assert strata[0] == strata[1]  # Equal values never receive different layers.
    selected = np.array([True, False, False, True, False, False, True, False, False, True])
    for groups in (offsets, strata):
        control = matched_random_selection(selected, groups, np.random.default_rng(2026))
        assert control.sum() == selected.sum()
        for key in np.unique(groups):
            assert (control & (groups == key)).sum() == (selected & (groups == key)).sum()
        assert np.array_equal(control, matched_random_selection(
            selected, groups, np.random.default_rng(2026)
        ))
    ranked = energy_ranked_selection(selected, offsets, energy)
    assert np.flatnonzero(ranked).tolist() == [2, 3, 8, 9]
    forced = np.ones(10, dtype=bool)
    assert matched_random_selection(forced, strata, np.random.default_rng(2)).all()


def test_tiny_full_occlusion_outputs_and_structure_unit(tmp_path):
    torch.set_num_threads(1)
    matrices = np.random.default_rng(7).random((3, 2, 16, 16)).astype(np.float32)
    matrices = (matrices + matrices.transpose(0, 1, 3, 2)) / 2
    dataset = tmp_path / "dataset.npz"
    np.savez(dataset, matrices=matrices, split=np.array(["test"] * 3),
             labels=np.array(["CHID", "CHIN", "OPCID"]),
             structure_id=np.array(["a", "b", "c"]), chrom=np.array(["chr"] * 3),
             window_start=np.zeros(3, dtype=int), window_end=np.full(3, 160), pooled_bin_size=10)
    checkpoints = []
    for seed in (3, 4, 5):
        torch.manual_seed(seed)
        model = MicroCCNN(2, 3)
        path = tmp_path / f"model_{seed}.pt"
        torch.save({"model_state": model.state_dict(), "class_names": ["CHID", "CHIN", "OPCID"],
                    "channel_mean": [0.1, 0.2], "channel_std": [0.3, 0.4]}, path)
        checkpoints.append(str(path))
    output = tmp_path / "out"
    result = audit_occlusion(str(dataset), checkpoints, str(output), minimum_offset=2, draws=2)
    assert not result["skipped_maps"]
    assert result["perturbation_rows"] == 3 * 3 * 3 * 6
    assert len(result["input_sha256"]) == 4
    assert all(row["structure_n"] == 3 for row in result["aggregates"] if row["class_group"] == "all")
    rows = pd.read_csv(output / "all_perturbations.csv")
    assert np.allclose(rows.abs_logit_change, rows.signed_logit_drop.abs())
    assert rows.overlap_with_top_fraction.between(0, 1).all()
    assert len(pd.read_csv(output / "paired_model_results.csv")) == 81
    assert len(pd.read_csv(output / "per_structure_results.csv")) == 27
    with np.load(output / "top_masks.npz") as data:
        assert data["masks"].shape == (3, 3, 3, 16, 16)
        assert not np.tril(data["masks"], k=1).any()
