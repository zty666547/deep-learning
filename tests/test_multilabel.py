import numpy as np
import pandas as pd
import pytest

from microc_foundation.datasets import prepare_multilabel_regions
from microc_foundation.metrics import multilabel_metrics
from microc_foundation.multilabel import (
    select_label_thresholds,
    summarize_multilabel_runs,
)


def test_prepare_multilabel_regions_encodes_sorted_classes():
    frame = pd.DataFrame(
        {
            "region_id": ["r1", "r2"],
            "chrom": ["chr1", "chr1"],
            "start": [10, 50],
            "end": [30, 90],
            "label_set": ["B;A", "B"],
            "splits": ["train", "test"],
        }
    )
    selected, class_names, targets = prepare_multilabel_regions(frame)
    assert class_names == ["A", "B"]
    assert targets.tolist() == [[1.0, 1.0], [0.0, 1.0]]
    assert selected["center"].tolist() == [20, 70]
    assert selected["split"].tolist() == ["train", "test"]


def test_prepare_multilabel_regions_rejects_cross_split_region():
    frame = pd.DataFrame(
        {
            "region_id": ["r1"],
            "chrom": ["chr1"],
            "start": [10],
            "end": [30],
            "label_set": ["A;B"],
            "splits": ["train;validation"],
        }
    )
    with pytest.raises(ValueError, match="more than one"):
        prepare_multilabel_regions(frame)


def test_multilabel_metrics_report_exact_match_and_per_label_counts():
    result = multilabel_metrics(
        np.array([[1, 0], [1, 1], [0, 1]]),
        np.array([[1, 0], [0, 1], [0, 1]]),
        ["A", "B"],
    )
    assert result["exact_match_ratio"] == pytest.approx(2 / 3)
    assert result["hamming_loss"] == pytest.approx(1 / 6)
    assert result["per_label"]["A"]["false_negative"] == 1
    assert result["per_label"]["B"]["recall"] == 1.0


def test_thresholds_are_selected_per_label():
    targets = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    probabilities = np.array(
        [[0.4, 0.1], [0.3, 0.2], [0.2, 0.6], [0.1, 0.7]]
    )
    thresholds = select_label_thresholds(
        targets, probabilities, candidates=np.array([0.25, 0.5, 0.75])
    )
    assert thresholds.tolist() == [0.25, 0.5]


def test_multilabel_summary_retains_seed_runs():
    result = {
        "seed": 1,
        "best_epoch": 2,
        "epochs_ran": 3,
        "class_names": ["A"],
        "thresholds": {"A": 0.4},
        "test": {
            "macro_f1": 0.5,
            "micro_f1": 0.5,
            "exact_match_ratio": 0.5,
            "hamming_loss": 0.5,
            "per_label": {"A": {"f1": 0.5, "recall": 1.0}},
        },
    }
    summary = summarize_multilabel_runs([result])
    assert summary["num_runs"] == 1
    assert summary["aggregate"]["test_macro_f1"]["mean"] == 0.5
