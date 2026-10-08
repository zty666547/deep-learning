import numpy as np
import pandas as pd
import pytest

from microc_foundation.clustering import (
    annotate_known_structures,
    kmeans,
    merge_candidate_windows,
    principal_components,
    robust_scale,
    silhouette_score,
)


def test_robust_scale_uses_median_and_iqr():
    matrix = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    scaled, median, scale = robust_scale(matrix)
    assert median.tolist() == [1.0, 1.0]
    assert scale.tolist() == [1.0, 1.0]
    assert scaled[:, 1].tolist() == [0.0, 0.0, 0.0]


def test_pca_and_kmeans_separate_two_groups():
    matrix = np.array([[-2.0, -2.0], [-1.8, -2.1], [2.0, 2.0], [2.1, 1.8]])
    embedding, _, ratios = principal_components(matrix, maximum_components=2)
    labels, _, _ = kmeans(embedding, 2, seed=7, n_init=3)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert ratios.sum() == pytest.approx(1.0)
    assert silhouette_score(embedding, labels) > 0.8


def test_known_overlap_uses_union_length_and_nearest_distance():
    windows = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [0, 100],
            "end": [50, 150],
            "center": [25, 125],
        }
    )
    structures = pd.DataFrame(
        {
            "structure_id": ["A", "B"],
            "chrom": ["chr1", "chr1"],
            "start": [10, 20],
            "end": [30, 40],
            "structure_type": ["X", "Y"],
        }
    )
    result = annotate_known_structures(windows, structures)
    assert result["known_overlap_bp"].tolist() == [30, 0]
    assert result["nearest_known_distance_bp"].tolist() == [0, 60]
    assert result.loc[0, "known_structure_types"] == "X;Y"


def test_merge_candidates_requires_adjacent_support():
    candidates = pd.DataFrame(
        {
            "window_id": ["w1", "w2", "w3"],
            "chrom": ["chr1"] * 3,
            "start": [0, 10, 100],
            "end": [40, 50, 140],
            "center": [20, 30, 120],
            "cluster": [1, 1, 0],
            "novelty_score": [2.0, 3.0, 4.0],
            "replicate_correlation": [0.95, 0.96, 0.97],
            "nearest_known_distance_bp": [50, 40, 100],
        }
    )
    result = merge_candidate_windows(
        candidates, maximum_start_gap_bp=20, minimum_support=2
    )
    assert len(result) == 1
    assert result.loc[0, "start"] == 0
    assert result.loc[0, "end"] == 50
    assert result.loc[0, "representative_window_id"] == "w2"
