import numpy as np
import pytest

from microc_foundation.error_analysis import (
    find_cross_class_annotation_overlaps,
    summarize_prediction_stability,
)


def test_prediction_stability_groups_and_consensus():
    targets = np.array([0, 1, 1])
    predictions = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]])
    probabilities = np.array(
        [
            [[0.8, 0.2], [0.7, 0.3], [0.3, 0.7]],
            [[0.9, 0.1], [0.4, 0.6], [0.6, 0.4]],
            [[0.6, 0.4], [0.8, 0.2], [0.7, 0.3]],
        ]
    )
    summary = summarize_prediction_stability(targets, predictions, probabilities)
    assert summary["correct_count"].tolist() == [3, 1, 1]
    assert summary["stability_group"].tolist() == ["always_correct", "mixed", "mixed"]
    assert summary["consensus_prediction"].tolist() == [0, 0, 0]
    assert summary["prediction_agreement"].tolist() == pytest.approx([1.0, 2 / 3, 2 / 3])


def test_prediction_stability_rejects_misaligned_shapes():
    with pytest.raises(ValueError, match="do not align"):
        summarize_prediction_stability(
            np.array([0, 1]),
            np.array([[0, 1]]),
            np.ones((1, 1, 2)),
        )


def test_cross_class_annotation_overlaps_exclude_same_class_and_touching():
    rows = [
        {
            "structure_id": "A",
            "true_class": "CHID",
            "chrom": "chr1",
            "annotation_start": 10,
            "annotation_end": 30,
        },
        {
            "structure_id": "B",
            "true_class": "CHIN",
            "chrom": "chr1",
            "annotation_start": 20,
            "annotation_end": 40,
        },
        {
            "structure_id": "C",
            "true_class": "CHID",
            "chrom": "chr1",
            "annotation_start": 25,
            "annotation_end": 35,
        },
        {
            "structure_id": "D",
            "true_class": "OPCID",
            "chrom": "chr1",
            "annotation_start": 40,
            "annotation_end": 50,
        },
    ]
    overlaps = find_cross_class_annotation_overlaps(rows)
    assert overlaps == [
        {
            "left_structure_id": "A",
            "left_class": "CHID",
            "right_structure_id": "B",
            "right_class": "CHIN",
            "overlap_bp": 10,
        },
        {
            "left_structure_id": "B",
            "left_class": "CHIN",
            "right_structure_id": "C",
            "right_class": "CHID",
            "overlap_bp": 10,
        },
    ]
