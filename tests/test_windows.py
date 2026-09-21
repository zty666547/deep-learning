import pytest

from microc_foundation.windows import GenomicWindow, parse_region, window_from_center


def test_parse_region_uses_half_open_coordinates():
    assert parse_region("chr1:1,000-2,000") == GenomicWindow("chr1", 1000, 2000)


def test_center_window_shifts_at_left_boundary():
    assert window_from_center("chr1", 100, 1000, 5000) == GenomicWindow("chr1", 0, 1000)


def test_center_window_shifts_at_right_boundary():
    assert window_from_center("chr1", 4900, 1000, 5000) == GenomicWindow(
        "chr1", 4000, 5000
    )


def test_center_outside_chromosome_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        window_from_center("chr1", 5001, 1000, 5000)

