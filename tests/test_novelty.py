import numpy as np
import pandas as pd
import pytest

from microc_foundation.novelty import (
    _kmeans,
    _window_features,
    scan_genome_candidates,
)


def test_window_features_are_finite_and_interpretable():
    matrix = np.eye(8, dtype=float)
    matrix[0, 1] = np.nan
    features = _window_features(matrix)
    assert features["missing_fraction"] > 0
    assert all(np.isfinite(value) for value in features.values())
    assert features["diagonal_mean"] == 1.0


def test_kmeans_is_deterministic_for_separated_points():
    values = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
    labels = _kmeans(values, num_clusters=2)
    assert labels.tolist() in ([0, 0, 1, 1], [1, 1, 0, 0])


def test_scan_requires_two_replicates(tmp_path):
    structures = pd.DataFrame(
        {"structure_id": ["A"], "chrom": ["chr1"], "start": [0], "end": [10]}
    )
    with pytest.raises(ValueError, match="two biological"):
        scan_genome_candidates(["one.cool"], structures, str(tmp_path))
