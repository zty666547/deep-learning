import pandas as pd

from microc_foundation.candidate_validation import evaluate_candidate_sensitivity


def test_candidate_sensitivity_reports_persistence():
    windows = pd.DataFrame(
        {
            "window_id": ["w1", "w2", "w3", "w4"],
            "chrom": ["chr1"] * 4,
            "start": [0, 10, 100, 200],
            "end": [40, 50, 140, 240],
            "center": [20, 30, 120, 220],
            "cluster": [0, 0, 1, 1],
            "known_overlap_bp": [0, 0, 0, 10],
            "replicate_correlation": [0.95, 0.96, 0.93, 0.99],
            "novelty_score": [3.0, 2.5, 1.0, 0.0],
            "nearest_known_distance_bp": [100, 90, 20, 0],
        }
    )
    baseline = pd.DataFrame(
        {
            "candidate_region_id": ["candidate_001"],
            "chrom": ["chr1"],
            "representative_center": [20],
        }
    )
    result = evaluate_candidate_sensitivity(
        windows,
        baseline,
        correlation_thresholds=(0.90,),
        novelty_quantiles=(0.50,),
        minimum_support_values=(2,),
        maximum_start_gap_bp=20,
    )
    assert result["summary"]["num_settings"] == 1
    assert result["settings"][0]["num_candidate_regions"] == 1
    assert result["persistence"][0]["persistence_fraction"] == 1.0
