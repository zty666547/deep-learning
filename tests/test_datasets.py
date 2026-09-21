from typing import ClassVar

import numpy as np
import pytest

from microc_foundation.datasets import aligned_window_from_center, pool_square_matrix
from microc_foundation.windows import GenomicWindow


class TinyCool:
    binsize = 10
    chromnames: ClassVar[list[str]] = ["chr1"]
    chromsizes: ClassVar[dict[str, int]] = {"chr1": 95}


def test_aligned_window_keeps_fixed_bin_count_at_right_edge():
    assert aligned_window_from_center(TinyCool(), "chr1", 94, 4) == GenomicWindow(
        "chr1", 60, 95
    )


def test_pool_square_matrix_sums_non_overlapping_blocks():
    matrix = np.arange(16).reshape(4, 4)
    result = pool_square_matrix(matrix, factor=2, method="sum")
    assert result.tolist() == [[10, 18], [42, 50]]


def test_pool_square_matrix_rejects_non_divisible_shape():
    with pytest.raises(ValueError, match="divisible"):
        pool_square_matrix(np.ones((3, 3)), factor=2)
