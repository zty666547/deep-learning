import cooler
import numpy as np
import pandas as pd
import pytest

from microc_foundation.candidate_resolution_audit import audit_candidate_resolution_robustness


@pytest.fixture
def tiny_resolution_sources(tmp_path):
    bins = pd.DataFrame({"chrom": ["chr1"] * 400, "start": np.arange(400) * 10,
                         "end": (np.arange(400) + 1) * 10})
    paths = {}
    for condition_index, condition in enumerate(("WT", "mutant")):
        paths[condition] = []
        for replicate in range(2):
            rng = np.random.default_rng(100 * condition_index + replicate)
            i, j = np.triu_indices(400)
            pixels = pd.DataFrame({"bin1_id": i, "bin2_id": j,
                                   "count": rng.integers(1, 100, len(i))})
            path = tmp_path / f"{condition}_{replicate}.cool"
            cooler.create_cooler(str(path), bins, pixels)
            paths[condition].append(str(path))
    candidates = pd.DataFrame({
        "candidate_region_id": ["candidate_001"], "chrom": ["chr1"],
        "start": [0], "end": [3200],
        "stable_candidate_key": ["task2-main-v1:chr1:0-3200"],
    })
    return paths, candidates


def test_fixed_candidates_are_compared_on_same_interval_at_each_resolution(
    tiny_resolution_sources, tmp_path,
):
    paths, candidates = tiny_resolution_sources
    summary = audit_candidate_resolution_robustness(
        paths, candidates, str(tmp_path / "outputs"),
    )
    assert summary["requested_resolutions_bp"] == [80, 160, 320]
    assert summary["minimum_offsets_by_resolution"] == {"80": 10, "160": 5, "320": 3}
    pairwise = pd.read_csv(tmp_path / "outputs/candidate_resolution_pairwise.csv")
    assert len(pairwise) == 18
    assert set(pairwise.resolution_bp) == {80, 160, 320}
    assert set(pairwise.actual_minimum_distance_bp) == {800, 960}
    assert pairwise.distance_centered_correlation.between(-1, 1).all()


def test_candidate_must_align_to_all_requested_resolutions(tiny_resolution_sources, tmp_path):
    paths, candidates = tiny_resolution_sources
    candidates.loc[0, "start"] = 80
    with pytest.raises(ValueError, match="align to every"):
        audit_candidate_resolution_robustness(paths, candidates, str(tmp_path / "outputs"))


def test_resolution_sources_must_be_unique(tiny_resolution_sources, tmp_path):
    paths, candidates = tiny_resolution_sources
    with pytest.raises(ValueError, match="unique positive"):
        audit_candidate_resolution_robustness(
            paths, candidates, str(tmp_path / "outputs"), resolutions=(80, 80),
        )
