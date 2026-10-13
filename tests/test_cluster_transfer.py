import numpy as np
import pandas as pd
import pytest

from microc_foundation.cluster_transfer import (
    SINGLE_FEATURES,
    assign_partition,
    candidate_guard_mask,
    circular_shift_control,
    fit_partition,
    library_cpm_matrix,
    nonoverlapping_grid,
    partition_agreement,
    single_replicate_features,
)


def test_single_features_exclude_replicate_correlation_and_transform_counts():
    frame = pd.DataFrame({name: [1., 2.] for name in SINGLE_FEATURES})
    frame["replicate_correlation"] = [1., -1.]
    values = single_replicate_features(frame)
    assert values.shape == (2, 7)
    assert values[0, 0] == pytest.approx(np.log(2))
    frame["replicate_correlation"] = [0., 0.]
    assert np.array_equal(values, single_replicate_features(frame))


def test_library_cpm_removes_global_multiplier_not_local_variation():
    counts = np.array([[1., 2.], [2., 4.]])
    assert np.array_equal(library_cpm_matrix(counts, 1e6), counts)
    assert np.array_equal(library_cpm_matrix(counts*10, 1e7), counts)
    changed = counts.copy()
    changed[0, 0] *= 2
    assert not np.array_equal(library_cpm_matrix(changed, 1e6), counts)


def test_library_cpm_rejects_missing_or_invalid_library_counts():
    for total in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            library_cpm_matrix(np.ones((2, 2)), total)
    for matrix in (np.full((2, 2), np.nan), np.full((2, 2), -1)):
        with pytest.raises(ValueError):
            library_cpm_matrix(matrix, 10)


def test_grid_and_guard_respect_half_open_boundaries_and_chromosomes():
    windows = pd.DataFrame({"window_id": ["a", "b", "c", "d"], "chrom": ["x", "x", "x", "y"],
                            "start": [0, 5, 10, 0], "end": [10, 15, 20, 10]})
    assert nonoverlapping_grid(windows).window_id.tolist() == ["a", "c", "d"]
    candidates = pd.DataFrame({"chrom": ["x"], "start": [10], "end": [20]})
    assert candidate_guard_mask(windows, candidates, 0).tolist() == [False, True, True, False]
    assert candidate_guard_mask(windows, candidates, 1).tolist() == [True, True, True, False]


def test_partition_assignment_never_refits_query_distribution():
    x = np.random.default_rng(1).normal(size=(30, 7))
    model = fit_partition(x)
    before = {key: value.copy() for key, value in model.items()}
    first, margin = assign_partition(x[:3], model)
    second, _ = assign_partition(np.vstack([x[:3], np.full((1, 7), 1e8)]), model)
    assert np.array_equal(first, second[:3])
    assert ((margin >= 0) & (margin <= 1)).all()
    assert all(np.array_equal(before[key], model[key]) for key in model)


def test_ari_is_label_invariant_but_same_space_agreement_is_not():
    a = np.array([0, 0, 1, 1])
    result = partition_agreement(a, 1-a)
    assert result["adjusted_rand_index"] == 1
    assert result["agreement"] == 0
    assert result["balanced_reference_agreement"] == 0
    assert partition_agreement(a, np.array([0, 1, 0, 1]))["adjusted_rand_index"] == pytest.approx(-.5)


def test_constant_partitions_and_all_nonzero_shift_control():
    assert partition_agreement(np.zeros(4), np.ones(4))["adjusted_rand_index"] == 1
    result = circular_shift_control(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))
    assert result["shift_count"] == 3
    assert result["shift_control_is_descriptive_not_p_value"] is True


def test_rejects_invalid_features_and_duplicate_coordinates():
    with pytest.raises(ValueError):
        fit_partition(np.ones((3, 7)))
    with pytest.raises(ValueError):
        assign_partition(np.full((2, 7), np.nan), fit_partition(np.random.default_rng(1).normal(size=(20, 7))))
    with pytest.raises(ValueError):
        partition_agreement(np.array([1]), np.array([1]))
    with pytest.raises(ValueError):
        candidate_guard_mask(pd.DataFrame(), pd.DataFrame(), -1)
    with pytest.raises(ValueError):
        nonoverlapping_grid(pd.DataFrame({"window_id": ["a", "a"], "chrom": ["x", "x"],
                                        "start": [0, 10], "end": [10, 20]}))
