import pandas as pd
import pytest

from microc_foundation.candidate_background_audit import select_candidate_and_background_windows


@pytest.fixture
def candidate_controls():
    windows = pd.DataFrame({
        "window_id": ["rep", "neighbor", "background", "known", "failed"],
        "chrom": ["chr1"] * 5,
        "start": [100, 250, 500, 700, 900],
        "end": [200, 350, 600, 800, 1000],
        "quality_pass": [True, True, True, True, False],
        "known_overlap_bp": [0, 0, 0, 20, 0],
    })
    candidates = pd.DataFrame({
        "candidate_region_id": ["candidate_001"], "chrom": ["chr1"],
        "start": [100], "end": [300], "representative_window_id": ["rep"],
    })
    return windows, candidates


def test_selector_keeps_candidate_representative_and_clean_background(candidate_controls):
    windows, candidates = candidate_controls
    selected = select_candidate_and_background_windows(windows, candidates)
    assert selected.window_id.tolist() == ["rep", "background"]


def test_selector_requires_qc_passed_representative(candidate_controls):
    windows, candidates = candidate_controls
    candidates.loc[0, "representative_window_id"] = "failed"
    with pytest.raises(ValueError, match="QC-passed"):
        select_candidate_and_background_windows(windows, candidates)


def test_selector_rejects_duplicate_window_ids(candidate_controls):
    windows, candidates = candidate_controls
    windows.loc[1, "window_id"] = "rep"
    with pytest.raises(ValueError, match="unique"):
        select_candidate_and_background_windows(windows, candidates)
