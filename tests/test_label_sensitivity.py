"""Synthetic interval/math controls, not biological experiment results."""

import numpy as np
import pandas as pd
import pytest

from microc_foundation.discovery_benchmark import benchmark_features
from microc_foundation.label_sensitivity import RULES, audit_label_sensitivity, label_roles


def tables():
    windows = pd.DataFrame({
        "window_id": [f"w{i}" for i in range(36)], "chrom": ["chr"] * 36,
        "start": np.arange(36) * 10, "end": np.arange(1, 37) * 10,
        "center": np.arange(36) * 10 + 5, "fold": np.repeat(np.arange(3), 12),
    })
    starts = np.arange(0, 36, 2) * 10
    lengths = np.tile([1, 5, 9, 1, 5, 9], 3)
    structures = pd.DataFrame({
        "structure_id": [f"s{i}" for i in range(len(starts))], "chrom": "chr",
        "start": starts, "end": starts + lengths, "structure_type": "CHIN",
    })
    rng = np.random.default_rng(21)
    for name in ["total_contacts_mean", "nonzero_fraction_mean", "near_far_ratio_mean",
                 "decay_slope_mean", "center_enrichment_mean",
                 "coefficient_of_variation_mean", "contact_entropy_mean",
                 "replicate_correlation"]:
        windows[name] = rng.random(len(windows)) + 0.1
    return windows, structures


def test_roles_union_center_endpoints_and_no_false_background():
    windows, structures = tables()
    structures.loc[0, "end"] = 4
    structures = pd.concat([structures, pd.DataFrame({
        "structure_id": ["extra"], "chrom": ["chr"], "start": [2], "end": [5],
        "structure_type": ["CHID"],
    })], ignore_index=True)
    # Center at start included; center at end excluded. Union, not sum, is 5 bp.
    structures.loc[1, ["start", "end"]] = [25, 29]
    result = label_roles(windows, structures)
    assert result.loc[0, "known_overlap_fraction"] == 0.5
    assert result.loc[0, "center_inside"] == "excluded"
    assert result.loc[2, "center_inside"] == "positive"
    for rule in RULES:
        assert result.loc[result[rule] == "background", "window_id"].tolist() == [
            f"w{i}" for i in range(1, 36, 2)
        ]
    assert result.loc[0, "coverage_75"] == "excluded"


def test_same_baseline_and_fixed_purge_after_exclusions():
    windows, structures = tables()
    memberships, counts, summary = audit_label_sensitivity(
        windows, structures, purge_bp=10, permutations=2, seed=7
    )
    reference = label_roles(windows, structures)
    reference["label"] = (reference.known_overlap_bp > 0).astype(int)
    predicted, baseline = benchmark_features(reference, purge_bp=10, permutations=2, seed=7)
    any_overlap = memberships.loc[memberships.rule == "any_overlap"]
    for name in baseline["models"]:
        assert np.allclose(any_overlap[f"{name}_probability"], predicted[f"{name}_probability"])
    assert summary["rules"]["any_overlap"]["shuffle_control"]["mean_fold_roc_auc"] == (
        baseline["shuffle_control"]["mean_fold_roc_auc"]
    )
    assert len(counts) == 15
    for result in summary["rules"].values():
        for model in result["models"].values():
            for fold in model["folds"]:
                ref = summary["reference_splits"][fold["fold"]]
                assert set(fold["train_window_ids"]) <= set(ref["train_window_ids"])
                assert set(fold["test_window_ids"]) <= set(ref["test_window_ids"])
                assert set(fold["train_window_ids"]).isdisjoint(fold["test_window_ids"])
    assert "w11" not in summary["rules"]["coverage_25"]["models"]["all_features"][
        "folds"
    ][1]["train_window_ids"]


def test_missing_class_marks_entire_rule_without_partial_auc():
    windows, structures = tables()
    structures["end"] = structures.start + 1
    memberships, counts, summary = audit_label_sensitivity(
        windows, structures, purge_bp=0, permutations=0
    )
    for rule in RULES[1:]:
        result = summary["rules"][rule]
        assert result["status"] == "unsupported"
        assert result["models"] == {} and result["shuffle_control"] is None
        assert result["reasons"]
        assert memberships.loc[memberships.rule == rule, "all_features_probability"].isna().all()
        assert (counts.loc[counts.rule == rule, "test_positive"] == 0).all()
    assert summary["rules"]["any_overlap"]["status"] == "evaluated"


def test_invalid_windows_and_permutations():
    windows, structures = tables()
    with pytest.raises(ValueError, match="permutations"):
        audit_label_sensitivity(windows, structures, permutations=-1)
    windows.loc[0, "window_id"] = "w1"
    with pytest.raises(ValueError, match="unique"):
        label_roles(windows, structures)
