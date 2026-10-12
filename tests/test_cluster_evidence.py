"""Small fixed-table consistency controls, not biological cluster evidence."""

import numpy as np
import pandas as pd
import pytest

from microc_foundation.cluster_evidence import build_cluster_evidence
from microc_foundation.clustering import build_model_features, robust_scale


def tables():
    windows = pd.DataFrame({
        "window_id": [f"w{i}" for i in range(6)], "chrom": "chr",
        "start": np.arange(6) * 10, "end": np.arange(1, 7) * 10,
        "center": np.arange(6) * 10 + 5, "quality_pass": True,
        "cluster": [0, 0, 0, 1, 1, 1], "known_overlap_bp": [0, 1, 1, 0, 1, 0],
        "known_structure_types": ["", "CHIN", "CHID", "", "OPCID", ""],
        "candidate_pass": [True, False, False, True, False, True],
        "novelty_score": [2, 0, 0, 1, 0, 1], "pc1": [-3, -2, -1, 1, 2, 3],
    })
    for column in ["total_contacts_mean", "nonzero_fraction_mean", "near_far_ratio_mean",
                   "decay_slope_mean", "center_enrichment_mean",
                   "coefficient_of_variation_mean", "contact_entropy_mean", "replicate_correlation"]:
        windows[column] = np.arange(6) + 1.0
    features, names = build_model_features(windows)
    _, median, scale = robust_scale(features)
    summary = {"selected_num_clusters": 2, "pca_components": 1, "num_quality_windows": 6,
               "feature_names": names, "feature_median": median.tolist(), "feature_scale": scale.tolist(),
               "cluster_centers": [[-2], [2]], "cluster_counts": {"0": 3, "1": 3},
               "cluster_known_overlap_fractions": {"0": 2 / 3, "1": 1 / 3},
               "num_candidate_regions": 1, "num_candidate_windows_before_merge": 3}
    candidates = pd.DataFrame({
        "candidate_region_id": ["candidate_001"], "chrom": ["chr"], "start": [0], "end": [40],
        "cluster": [0], "representative_window_id": ["w0"],
        "num_supporting_windows": [2], "supporting_window_ids": ["w0;w3"],
    })
    return windows, candidates, summary


def test_reconciles_centroids_mixed_support_and_unmerged_selection():
    members, profiles, centers, support, audit = build_cluster_evidence(*tables())
    assert len(members) == 6 and len(profiles) == 16 and len(centers) == 2
    assert audit["refitted"] is False and audit["max_centroid_absolute_error"] == 0
    assert audit["final_region_support_windows"] == 2
    assert audit["selected_unmerged_windows"] == 1
    assert support.support_cluster.tolist() == [0, 1]  # Representative label is not all support labels.
    assert support.support_count.tolist() == [1, 1]
    assert centers.nearest_window_id.tolist() == ["w1", "w4"]
    assert members.loc[members.window_id == "w5", "selection_stage"].item() == "selected_unmerged"
    assert audit["all_clusters_contain_known"] is True


def test_rejects_stale_centroid_scaling_and_support_ids():
    windows, candidates, summary = tables()
    summary["cluster_centers"][0][0] = -2.1
    with pytest.raises(ValueError, match="centroid"):
        build_cluster_evidence(windows, candidates, summary)
    windows, candidates, summary = tables()
    summary["feature_scale"][0] *= 2
    with pytest.raises(ValueError, match="scaling"):
        build_cluster_evidence(windows, candidates, summary)
    windows, candidates, summary = tables()
    candidates.loc[0, "supporting_window_ids"] = "w0;absent"
    with pytest.raises(ValueError, match="support IDs"):
        build_cluster_evidence(windows, candidates, summary)


def test_rejects_support_known_overlap_and_incorrect_composition():
    windows, candidates, summary = tables()
    windows.loc[0, "known_overlap_bp"] = 1
    with pytest.raises(ValueError, match="support coordinates"):
        build_cluster_evidence(windows, candidates, summary)
    windows, candidates, summary = tables()
    summary["cluster_counts"]["0"] = 4
    with pytest.raises(ValueError, match="composition"):
        build_cluster_evidence(windows, candidates, summary)
