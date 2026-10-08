import numpy as np
import pytest

from microc_foundation.discovery import (
    diagonal_window_starts,
    extract_contact_features,
    matrix_correlation,
)


def test_diagonal_window_starts_include_rightmost_full_window():
    assert diagonal_window_starts(100, 10, 40, 30) == [0, 30, 60]
    assert diagonal_window_starts(105, 10, 40, 20) == [0, 20, 40, 60]


def test_diagonal_window_starts_require_bin_alignment():
    with pytest.raises(ValueError, match="divisible"):
        diagonal_window_starts(100, 10, 45, 20)


def test_contact_features_capture_intensity_and_decay():
    matrix = np.zeros((8, 8), dtype=float)
    for offset in range(8):
        value = 8 - offset
        indices = np.arange(8 - offset)
        matrix[indices, indices + offset] = value
        matrix[indices + offset, indices] = value
    features = extract_contact_features(matrix)
    assert features["finite_fraction"] == 1.0
    assert features["diagonal_mean"] == 8.0
    assert features["near_band_mean"] > features["mid_band_mean"]
    assert features["decay_slope"] < 0
    assert 0 <= features["contact_entropy"] <= 1
    assert features["zero_bin_fraction"] == 0
    assert features["largest_zero_run_fraction"] == 0


def test_contact_features_detect_consecutive_zero_bins():
    matrix = np.ones((8, 8), dtype=float)
    matrix[2:4, :] = 0
    matrix[:, 2:4] = 0
    features = extract_contact_features(matrix)
    assert features["zero_bin_fraction"] == 0.25
    assert features["largest_zero_run_fraction"] == 0.25


def test_matrix_correlation_uses_log_upper_triangle():
    matrix = np.arange(64, dtype=float).reshape(8, 8)
    assert matrix_correlation(matrix, matrix * 2) > 0.99
    with pytest.raises(ValueError, match="equal square"):
        matrix_correlation(matrix, matrix[:7])
