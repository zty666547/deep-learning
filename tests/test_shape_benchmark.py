import cooler
import numpy as np
import pandas as pd
import pytest

from microc_foundation.shape_benchmark import extract_replicate_shapes, pooled_shape_features


def test_pure_distance_envelope_has_no_texture():
    distance = np.abs(np.arange(32)[:, None] - np.arange(32)[None, :])
    values = pooled_shape_features(np.expm1(4 / (distance + 1)), blocks=4, minimum_offset=2)
    assert len(values["raw_log"]) == 10
    assert len(values["distance_texture"]) == 20
    assert np.allclose(values["distance_texture"], 0)


def test_texture_invariant_to_log_distance_envelope_and_diagonal():
    rng = np.random.default_rng(9)
    distance = np.abs(np.arange(32)[:, None] - np.arange(32)[None, :])
    noise = rng.uniform(0, 0.5, (32, 32))
    noise = (noise + noise.T) / 2
    first = np.expm1(4 / (distance + 1) + noise)
    second = np.expm1(8 / (distance + 1) + noise)
    np.fill_diagonal(second, 1e6)
    left = pooled_shape_features(first, blocks=4, minimum_offset=2)
    right = pooled_shape_features(second, blocks=4, minimum_offset=2)
    np.testing.assert_allclose(left["distance_texture"], right["distance_texture"], atol=1e-12)
    assert not np.allclose(left["raw_log"], right["raw_log"])


def test_shapes_validate_counts_and_block_geometry():
    with pytest.raises(ValueError, match="symmetric"):
        pooled_shape_features(np.full((32, 32), np.nan))
    with pytest.raises(ValueError, match="block count"):
        pooled_shape_features(np.ones((31, 31)), blocks=8)
    with pytest.raises(ValueError, match="retain pixels"):
        pooled_shape_features(np.ones((32, 32)), blocks=8, minimum_offset=5)


def test_extract_tiny_coolers_has_replicate_axis_and_checks_coordinates(tmp_path):
    size = 64
    bins = pd.DataFrame({"chrom": ["chr"] * size, "start": np.arange(size) * 10,
                         "end": np.arange(1, size + 1) * 10})
    i, j = np.triu_indices(size)
    pixels = pd.DataFrame({"bin1_id": i, "bin2_id": j,
                           "count": np.random.default_rng(4).integers(1, 100, len(i))})
    path = str(tmp_path / "tiny.cool")
    cooler.create_cooler(path, bins, pixels)
    windows = pd.DataFrame({"window_id": ["a", "b"], "chrom": ["chr"] * 2,
                            "start": [0, 320], "end": [320, 640]})
    features, source = extract_replicate_shapes([path, path], windows, blocks=4, minimum_offset=2)
    assert features["distance_texture"].shape == (2, 2, 20)
    np.testing.assert_allclose(features["raw_log"][:, 0], features["raw_log"][:, 1])
    assert source["minimum_distance_bp"] == 20
    windows.loc[0, "start"] = 1
    with pytest.raises(ValueError, match="bin aligned"):
        extract_replicate_shapes([path, path], windows, blocks=4, minimum_offset=2)
