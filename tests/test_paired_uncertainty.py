import numpy as np
import pandas as pd
import pytest

from microc_foundation.paired_uncertainty import (
    genomic_groups,
    log_distance_residual_r,
    paired_group_bootstrap,
)


def metadata(starts, ends, chrom=None):
    return pd.DataFrame({"structure_id": [f"s{i}" for i in range(len(starts))],
                         "chrom": chrom or ["x"]*len(starts), "window_start": starts,
                         "window_end": ends})


def test_groups_merge_crossing_intervals_transitively_not_by_midpoints():
    # Middle interval crosses a grid boundary and links all three blocks.
    frame = metadata([0, 9, 19, 30], [2, 21, 22, 32])
    assert genomic_groups(frame, block_bp=10).tolist() == [0, 0, 0, 1]


def test_half_open_overlap_and_chromosome_groups_preserve_input_order():
    frame = metadata([10, 0, 0, 5], [20, 10, 10, 15], ["x", "x", "y", "x"])
    groups = genomic_groups(frame, block_bp=1)
    assert groups[0] == groups[1] == groups[3] and groups[2] != groups[0]
    assert genomic_groups(metadata([0, 10], [10, 20]), block_bp=1).tolist() == [0, 1]


def test_guard_sensitivity_and_invalid_metadata():
    frame = metadata([0, 30], [10, 40])
    assert genomic_groups(frame, block_bp=1, guard_bp=11).tolist() == [0, 0]
    for invalid in (metadata([0, 1.2], [1, 3]), metadata([0], [0]), metadata([-1], [10])):
        with pytest.raises(ValueError):
            genomic_groups(invalid)


def test_paired_bootstrap_deterministic_and_structure_weighted():
    values, groups = np.array([1., 1., 10.]), np.array([0, 0, 1])
    result = paired_group_bootstrap(values, groups, draws=1000)
    assert result == paired_group_bootstrap(values, groups, draws=1000)
    assert result["mean_delta"] == 4  # Not block-weighted 5.5.
    assert result["group_sizes"] == [2, 1]
    assert result["interval_95"] == [1, 10]
    assert result["leave_one_group_out_mean_range"] == [1, 10]


def test_single_spatial_group_does_not_emit_fake_interval():
    result = paired_group_bootstrap(np.array([1., 2.]), np.array([0, 0]))
    assert result["status"] == "insufficient_genomic_groups"
    assert result["interval_95"] is None
    with pytest.raises(ValueError):
        paired_group_bootstrap(np.array([np.nan]), np.array([0]))


def test_distance_residual_drops_envelope_without_second_log():
    rng = np.random.default_rng(3)
    target = rng.uniform(size=(20, 20))
    rows, columns = np.indices(target.shape)
    altered = target + np.abs(rows-columns)*10
    assert log_distance_residual_r(target, altered, 2) == pytest.approx(1)
    assert np.isnan(log_distance_residual_r(np.ones((20, 20)), target, 2))
    with pytest.raises(ValueError):
        log_distance_residual_r(target, target, 19)
