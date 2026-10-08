import numpy as np
import pytest

from microc_foundation.error_analysis import summarize_prediction_stability


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
