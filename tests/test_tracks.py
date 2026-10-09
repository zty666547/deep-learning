import numpy as np
import pytest

from microc_foundation.tracks import centered_rolling_mean


def test_centered_rolling_mean_is_edge_aware():
    result = centered_rolling_mean(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), window=3)
    assert result.tolist() == pytest.approx([1.5, 2.0, 3.0, 4.0, 4.5])


def test_centered_rolling_mean_requires_odd_window():
    with pytest.raises(ValueError, match="odd"):
        centered_rolling_mean(np.ones(3), window=2)
