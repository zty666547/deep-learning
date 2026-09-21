import numpy as np
import pytest

from microc_foundation.normalization import normalize_matrix


def test_minmax_preserves_nan_and_scales_finite_values():
    matrix = np.array([[0.0, 2.0], [4.0, np.nan]])
    result = normalize_matrix(matrix, "minmax")
    np.testing.assert_allclose(result[:2, 0], [0.0, 1.0])
    assert result[0, 1] == 0.5
    assert np.isnan(result[1, 1])


def test_log1p_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        normalize_matrix(np.array([[-1.0, 0.0]]), "log1p")

