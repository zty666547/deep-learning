import numpy as np
import pandas as pd
import pytest
import torch

from microc_foundation.centered_detection import (
    centered_registry,
    component_binary_metrics,
    component_weights,
    detector_outputs,
    purged_centered_splits,
    train_centered_detector,
)


def examples():
    # Small engineered fixtures test software, not experimental performance.
    length = 500_000
    rows = []
    for fold in range(5):
        for offset, label in [(20000, 1), (22000, 1), (60000, 0), (81000, 0)]:
            start = fold*100000+offset
            rows.append({"start": start, "end": start+10000, "center": start+5000,
                         "fold": fold, "label": label})
    return pd.DataFrame(rows), length


def test_registry_deduplicates_centres_preserves_ids_and_background_rule():
    structures = pd.DataFrame({"structure_id": ["a", "b"],
        "chrom": ["NC_000913.3"]*2, "start": [50000, 50010],
        "end": [53000, 53010], "center": [51400, 51410],
        "structure_type": ["CHIN", "CHID"]})
    scan = pd.DataFrame({"window_id": ["w0", "w1", "w2"],
        "chrom": ["NC_000913.3"]*3, "start": [0, 40960, 81920],
        "end": [20480, 61440, 102400], "center": [10240, 51200, 92160],
        "quality_pass": [True, True, False]})
    result = centered_registry(structures, scan)
    assert len(result) == 2
    positive = result.loc[result.label == 1].iloc[0]
    assert positive.structure_ids == "a;b" and positive.types == "CHID;CHIN"
    assert positive.start % 160 == 0 and positive.end-positive.start == 20480
    assert result.loc[result.label == 0].sample_id.tolist() == ["w0"]


def test_split_actual_interval_purge_and_circular_boundary():
    registry, length = examples()
    registry.loc[registry.index[-1], ["start", "end", "center"]] = [489000, 499000, 494000]
    for fold, train, test in purged_centered_splits(registry, chrom_length=length, purge_bp=20480):
        a, b = registry.loc[train], registry.loc[test]
        assert ((a.end.to_numpy()[:, None]+20480 <= b.start.to_numpy()[None, :]) |
                (a.start.to_numpy()[:, None] >= b.end.to_numpy()[None, :]+20480)).all()
        assert not (train & test).any()
        if fold == 0:
            assert not train[-1]
        if fold == 4:
            assert (a.start >= 20480).all()
    registry.loc[0, "fold"] = 1
    with pytest.raises(ValueError, match="fixed genomic block"):
        list(purged_centered_splits(registry, chrom_length=length))


def test_component_weights_do_not_multiply_overlapping_positive_mass():
    registry, _ = examples()
    weights, groups = component_weights(registry)
    assert weights.sum() == pytest.approx(1)
    assert weights[registry.label == 1].sum() == pytest.approx(.5)
    assert groups[0] == groups[1]
    assert weights[0] == pytest.approx(weights[2])  # Two positives share one component.
    repeated = pd.concat([registry.iloc[:1], registry], ignore_index=True)
    again, _ = component_weights(repeated)
    assert again[:3].sum() == pytest.approx(weights[:2].sum())


def test_weighted_metrics_ties_and_class_imbalance():
    registry, _ = examples()
    perfect = component_binary_metrics(registry, registry.label.to_numpy(float))
    assert perfect["roc_auc"] == 1 and perfect["balanced_accuracy"] == 1
    constant = component_binary_metrics(registry, np.full(len(registry), .5))
    assert constant["roc_auc"] == .5 and constant["recall"] == 1
    assert constant["specificity"] == 0
    with pytest.raises(ValueError):
        component_binary_metrics(registry, np.full(len(registry), np.nan))


def test_training_is_deterministic_and_random_control_remains_initial():
    registry, _ = examples()
    registry = registry.iloc[:4]
    maps = np.random.default_rng(2).normal(size=(4, 64, 64)).astype("float32")
    torch.set_num_threads(1)
    trained, initial, history = train_centered_detector(maps, registry, seed=3, epochs=1)
    other, same_initial, same_history = train_centered_detector(maps, registry, seed=3, epochs=1)
    assert history == same_history
    assert all(torch.equal(x, y) for x, y in zip(trained.parameters(), other.parameters()))
    assert all(torch.equal(x, y) for x, y in zip(initial.parameters(), same_initial.parameters()))
    assert any(not torch.equal(x, y) for x, y in zip(initial.parameters(), trained.parameters()))
    scores, features = detector_outputs(trained, maps)
    assert features.shape == (4, 128) and scores.shape == (4,)
    appended = np.concatenate([maps, maps[:1]+10])
    np.testing.assert_allclose(scores, detector_outputs(trained, appended)[0][:4], atol=1e-6)


def test_invalid_training_and_missing_classes_are_rejected():
    registry, _ = examples()
    for frame in [registry.loc[registry.label == 1], registry.iloc[:0]]:
        with pytest.raises(ValueError):
            component_weights(frame)
    with pytest.raises(ValueError):
        train_centered_detector(np.full((len(registry), 64, 64), np.nan), registry, seed=1)
    with pytest.raises(ValueError):
        detector_outputs(None, np.ones((0, 64, 64)))
