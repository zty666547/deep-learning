"""Small mathematical controls; no synthetic biological conclusions."""

import numpy as np
import pandas as pd
import pytest

from microc_foundation.discovery_benchmark import (
    benchmark_features,
    binary_metrics,
    fit_logistic,
    predict_logistic,
    select_nonoverlapping_windows,
    spatial_splits,
    structure_coverage,
)


def test_auc_ties_and_constant_baseline():
    metrics = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.5, 0.5, 0.9]))
    assert metrics["roc_auc"] == 0.875
    baseline = binary_metrics(np.array([0, 1, 1]), np.ones(3))
    assert baseline["balanced_accuracy"] == 0.5
    assert baseline["recall"] == 1
    assert baseline["specificity"] == 0
    with pytest.raises(ValueError, match="both classes"):
        binary_metrics(np.ones(3), np.ones(3))


def test_logistic_train_only_scale_and_ordering():
    features = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    _, fit = fit_logistic(features, labels)
    assert fit["median"] == [0.0]
    probabilities = predict_logistic(np.array([[-100.0], [0.0], [100.0]]), fit)
    assert probabilities[0] < 0.01 and probabilities[2] > 0.99
    assert probabilities[1] == pytest.approx(0.5)
    assert fit["median"] == [0.0]  # Test inputs cannot change the fitted scaler.


def test_coverage_uses_interval_union():
    structures = pd.DataFrame({
        "structure_id": ["a", "b"], "structure_type": ["CHIN", "CHID"],
        "chrom": ["chr", "chr"], "start": [0, 100], "end": [40, 160],
    })
    windows = pd.DataFrame({"chrom": ["chr"] * 4,
                            "start": [0, 10, 100, 120], "end": [20, 30, 130, 160]})
    covered = structure_coverage(structures, windows)
    assert covered.covered_bp.tolist() == [30, 60]
    assert covered.coverage_fraction.tolist() == [0.75, 1.0]


def test_grid_and_purged_blocks():
    windows = pd.DataFrame({
        "window_id": [f"w{i}" for i in range(12)], "chrom": ["chr"] * 12,
        "start": np.arange(12) * 10, "end": np.arange(1, 13) * 10,
        "center": np.arange(12) * 10 + 5, "quality_pass": [True] * 12,
    })
    structures = pd.DataFrame({
        "structure_id": ["a", "b", "c"], "structure_type": ["CHIN"] * 3,
        "chrom": ["chr"] * 3, "start": [0, 40, 80], "end": [10, 50, 90],
    })
    selected = select_nonoverlapping_windows(
        windows, structures, chrom="chr", chrom_length=120, window_bp=10, folds=3
    )
    assert selected.label.tolist() == [1, 0, 0, 0] * 3
    fold, train, test = list(spatial_splits(selected, purge_bp=10))[1]
    assert fold == 1
    assert np.flatnonzero(test).tolist() == [4, 5, 6, 7]
    assert np.flatnonzero(train).tolist() == [0, 1, 2, 9, 10, 11]
    malformed = windows.copy()
    malformed["quality_pass"] = malformed.quality_pass.astype(object)
    malformed.loc[0, "quality_pass"] = "True"
    with pytest.raises(ValueError, match="booleans"):
        select_nonoverlapping_windows(
            malformed, structures, chrom="chr", chrom_length=120, window_bp=10, folds=3
        )


def test_end_to_end_oof_fit_excludes_test_data():
    windows = pd.DataFrame({
        "window_id": [f"w{i}" for i in range(12)], "chrom": ["chr"] * 12,
        "start": np.arange(12) * 10, "end": np.arange(1, 13) * 10,
        "fold": np.repeat(np.arange(3), 4), "label": [0, 0, 1, 1] * 3,
    })
    for column in ["total_contacts_mean", "nonzero_fraction_mean", "near_far_ratio_mean",
                   "decay_slope_mean", "center_enrichment_mean",
                   "coefficient_of_variation_mean", "contact_entropy_mean",
                   "replicate_correlation"]:
        windows[column] = np.array([1.0, 1.0, 2.0, 2.0] * 3)
    # An extreme held-out feature must not affect that fold's scaler.
    windows.loc[0, "total_contacts_mean"] = 1e10
    predicted, result = benchmark_features(windows, purge_bp=0, permutations=2, seed=7)
    first = result["models"]["all_features"]["folds"][0]
    assert set(first["train_window_ids"]).isdisjoint(first["test_window_ids"])
    assert first["fit"]["median"][0] == pytest.approx((np.log(2) + np.log(3)) / 2)
    assert predicted.all_features_probability.notna().all()
    assert len(result["shuffle_control"]["mean_fold_roc_auc"]) == 2
