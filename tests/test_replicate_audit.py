import cooler
import numpy as np
import pandas as pd
import pytest

from microc_foundation.replicate_audit import (
    audit_replicate_consistency,
    distance_controlled_correlations,
)


def test_shared_decay_alone_is_not_residual_replication():
    distance = np.abs(np.arange(20)[:, None] - np.arange(20)[None, :])
    first = np.expm1(4 / (distance + 1))
    second = np.expm1(8 / (distance + 1))
    result = distance_controlled_correlations(first, second)
    assert result["offdiagonal_log_correlation"] == pytest.approx(1)
    assert np.isnan(result["distance_centered_correlation"])


def test_distance_centering_preserves_shared_texture_not_different_envelope():
    rng = np.random.default_rng(2026)
    distance = np.abs(np.arange(20)[:, None] - np.arange(20)[None, :])
    noise = rng.uniform(0, 0.5, (20, 20))
    noise = (noise + noise.T) / 2
    result = distance_controlled_correlations(
        np.expm1(4 / (distance + 1) + noise), np.expm1(8 / (distance + 1) + noise),
    )
    assert result["distance_centered_correlation"] == pytest.approx(1)
    assert result["offdiagonal_log_correlation"] < 1


def test_diagonal_cannot_change_offdiagonal_scores():
    rng = np.random.default_rng(1)
    first, second = rng.uniform(0, 100, (20, 20)), rng.uniform(0, 100, (20, 20))
    before = distance_controlled_correlations(first, second)
    np.fill_diagonal(first, 1e6)
    np.fill_diagonal(second, 1e6)
    after = distance_controlled_correlations(first, second)
    assert before["offdiagonal_log_correlation"] == after["offdiagonal_log_correlation"]
    assert before["distance_centered_correlation"] == after["distance_centered_correlation"]


@pytest.mark.parametrize("offset", [0, -1, 18])
def test_invalid_offsets_rejected(offset):
    with pytest.raises(ValueError, match="offset"):
        distance_controlled_correlations(np.ones((20, 20)), np.ones((20, 20)), minimum_offset=offset)


def test_bad_contacts_rejected():
    with pytest.raises(ValueError, match="finite nonnegative"):
        distance_controlled_correlations(np.ones((20, 20)), np.full((20, 20), np.nan))


@pytest.fixture
def tiny_audit(tmp_path):
    bins = pd.DataFrame({"chrom": ["chr1"] * 60, "start": np.arange(60) * 160,
                         "end": (np.arange(60) + 1) * 160})
    i, j = np.triu_indices(60)
    pixels = pd.DataFrame({"bin1_id": i, "bin2_id": j,
                           "count": np.random.default_rng(1).integers(1, 100, len(i))})
    path = tmp_path / "fixture.cool"
    cooler.create_cooler(str(path), bins, pixels)
    windows = pd.DataFrame({"window_id": ["A", "B", "C"], "chrom": ["chr1"] * 3,
                            "start": [0, 3200, 6400], "end": [3200, 6400, 9600],
                            "quality_pass": [True] * 3, "known_overlap_bp": [0, 10, 0]})
    candidates = pd.DataFrame({"candidate_region_id": ["candidate_001"], "chrom": ["chr1"],
                               "start": [0], "end": [3200], "representative_window_id": ["A"]})
    return [str(path), str(path)], windows, candidates


def test_audit_pipeline_creates_coordinate_registry_and_all_window_groups(tiny_audit, tmp_path):
    summary = audit_replicate_consistency(*tiny_audit, str(tmp_path / "outputs"))
    assert summary["num_windows"] == 3
    assert set(summary["groups"]) == {"candidate_representative", "known_overlap", "unannotated_background"}
    registry = pd.read_csv(tmp_path / "outputs/candidate_registry.csv")
    assert registry.loc[0, "stable_candidate_key"] == "task2-main-v1:chr1:0-3200"
    assert registry.loc[0, "distance_centered_correlation"] == pytest.approx(1)
    assert (tmp_path / "outputs/replication_distance_controls.png").is_file()


def test_audit_rejects_candidate_id_with_wrong_coordinates(tiny_audit, tmp_path):
    paths, windows, candidates = tiny_audit
    candidates.loc[0, "chrom"] = "wrong-chrom"
    with pytest.raises(ValueError, match="coordinates"):
        audit_replicate_consistency(paths, windows, candidates, str(tmp_path / "outputs"))


def test_audit_records_library_cpm_normalization(tiny_audit, tmp_path):
    paths, windows, candidates = tiny_audit
    summary = audit_replicate_consistency(
        paths, windows, candidates, str(tmp_path / "outputs"), normalization="library_cpm",
    )
    assert summary["normalization"] == "library_cpm"
    assert len(summary["library_total_contact_counts"]) == 2
    assert all(value > 0 for value in summary["library_total_contact_counts"])


def test_audit_rejects_unsupported_normalization(tiny_audit, tmp_path):
    paths, windows, candidates = tiny_audit
    with pytest.raises(ValueError, match="normalization"):
        audit_replicate_consistency(
            paths, windows, candidates, str(tmp_path / "outputs"), normalization="balanced",
        )
