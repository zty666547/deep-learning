import numpy as np
import pytest

from microc_foundation.explainability import normalize_saliency


def test_normalize_saliency_clips_outlier_and_preserves_shape():
    result = normalize_saliency(np.array([[0.0, 1.0], [2.0, 100.0]]), percentile=75)
    assert result.shape == (2, 2)
    assert result.min() == 0
    assert result.max() == 1


def test_normalize_saliency_rejects_non_matrix():
    with pytest.raises(ValueError, match="two-dimensional"):
        normalize_saliency(np.ones(3))
