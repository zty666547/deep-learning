import numpy as np
import pytest

from microc_foundation.structure_fidelity import (
    annotation_shape_r,
    compare_profiles,
    distance_residual,
    excess_profile,
)


def test_centering_drops_distance_envelope_and_near_diagonal():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(32, 32))
    x = (x + x.T) / 2
    rows, columns = np.indices(x.shape)
    y = x + abs(rows-columns)*3
    y[np.diag_indices(32)] = 999
    assert np.allclose(distance_residual(x, minimum_offset=3), distance_residual(y, minimum_offset=3))
    assert np.array_equal(distance_residual(np.ones((32, 32)), minimum_offset=3), np.zeros((32, 32)))


def test_profile_is_coordinate_sensitive_and_peaks_are_not_annotation_forced():
    a = np.zeros(40)
    b = np.zeros(40)
    a[10], b[15] = 1, 1
    metrics = compare_profiles(a, b, bin_bp=80)
    assert metrics["peak_displacement_bp"] == 400
    assert compare_profiles(a, a)["profile_correlation"] == pytest.approx(1)
    assert compare_profiles(a, a)["top_profile_jaccard"] == 1


def test_constant_profiles_do_not_emit_fake_localization():
    result = compare_profiles(np.ones(20), np.ones(20))
    assert all(np.isnan(value) for value in result.values())


def test_annotation_only_correlates_pairs_inside_its_interval():
    x = np.random.default_rng(1).normal(size=(32, 32))
    y = x.copy()
    y[20:, :] += 100
    result, n = annotation_shape_r(x, y, start_bp=0, annotation_start=0,
                                 annotation_end=800, minimum_offset=2, maximum_offset=20)
    assert result == pytest.approx(1) and n == 36
    result, n = annotation_shape_r(x, y, start_bp=0, annotation_start=0,
                                 annotation_end=80, minimum_offset=2, maximum_offset=20)
    assert n == 0 and np.isnan(result)


def test_profile_smoothing_is_finite_and_rejects_invalid_band():
    x = np.eye(32)
    assert excess_profile(x, minimum_offset=3, maximum_offset=10).shape == (32,)
    with pytest.raises(ValueError):
        excess_profile(x, minimum_offset=10, maximum_offset=40)
    with pytest.raises(ValueError):
        distance_residual(np.ones((4, 5)))
    with pytest.raises(ValueError):
        compare_profiles(np.ones(4), np.ones(4), bin_bp=0)
