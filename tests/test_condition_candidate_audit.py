import cooler
import numpy as np
import pandas as pd
import pytest

from microc_foundation.condition_candidate_audit import audit_candidate_condition_transfer


@pytest.fixture
def tiny_conditions(tmp_path):
    paths = {}
    bins = pd.DataFrame({"chrom": ["chr1"] * 40, "start": np.arange(40) * 160,
                         "end": (np.arange(40) + 1) * 160})
    for condition_index, condition in enumerate(("WT", "mutant")):
        paths[condition] = []
        for replicate in range(2):
            rng = np.random.default_rng(100 * condition_index + replicate)
            upper = np.triu(rng.integers(1, 100, (40, 40)))
            i, j = np.triu_indices(40)
            pixels = pd.DataFrame({"bin1_id": i, "bin2_id": j, "count": upper[i, j]})
            path = tmp_path / f"{condition}_{replicate}.cool"
            cooler.create_cooler(str(path), bins, pixels)
            paths[condition].append(str(path))
    candidates = pd.DataFrame({
        "candidate_region_id": ["candidate_001"], "chrom": ["chr1"],
        "start": [0], "end": [3200],
        "stable_candidate_key": ["task2-main-v1:chr1:0-3200"],
    })
    return paths, candidates


def test_fixed_candidate_transfer_emits_within_and_all_cross_rep_pairs(tiny_conditions, tmp_path):
    paths, candidates = tiny_conditions
    summary = audit_candidate_condition_transfer(
        paths, candidates, str(tmp_path / "outputs"), minimum_offset=2,
    )
    assert summary["num_candidate_regions"] == 1
    assert summary["normalization"] == "raw_counts"
    pairwise = pd.read_csv(tmp_path / "outputs/candidate_condition_pairwise.csv")
    assert len(pairwise) == 6
    assert set(pairwise.condition_contrast) == {"within:WT", "within:mutant", "WT_vs_mutant"}
    cross = pairwise[pairwise.condition_contrast == "WT_vs_mutant"]
    assert len(cross) == 4
    assert pairwise.distance_centered_correlation.between(-1, 1).all()
    summary_table = pd.read_csv(tmp_path / "outputs/candidate_condition_summary.csv")
    assert summary_table.loc[summary_table.condition_contrast == "WT_vs_mutant", "num_pairs"].item() == 4


def test_candidate_must_be_grid_aligned(tiny_conditions, tmp_path):
    paths, candidates = tiny_conditions
    candidates.loc[0, "start"] = 1
    with pytest.raises(ValueError, match="bin-aligned"):
        audit_candidate_condition_transfer(paths, candidates, str(tmp_path / "outputs"), minimum_offset=2)


def test_reused_file_cannot_count_as_multiple_replicates(tiny_conditions, tmp_path):
    paths, candidates = tiny_conditions
    paths["mutant"][1] = paths["WT"][0]
    with pytest.raises(ValueError, match="multiple conditions"):
        audit_candidate_condition_transfer(paths, candidates, str(tmp_path / "outputs"), minimum_offset=2)


def test_library_cpm_mode_is_recorded(tiny_conditions, tmp_path):
    paths, candidates = tiny_conditions
    summary = audit_candidate_condition_transfer(
        paths, candidates, str(tmp_path / "outputs"), minimum_offset=2,
        normalization="library_cpm",
    )
    assert summary["normalization"] == "library_cpm"
    assert set(summary["library_total_contact_counts"]) == {"WT", "mutant"}
    assert all(total > 0 for group in summary["library_total_contact_counts"].values()
               for total in group)


def test_unsupported_normalization_is_rejected(tiny_conditions, tmp_path):
    paths, candidates = tiny_conditions
    with pytest.raises(ValueError, match="normalization"):
        audit_candidate_condition_transfer(
            paths, candidates, str(tmp_path / "outputs"), minimum_offset=2,
            normalization="balanced",
        )
