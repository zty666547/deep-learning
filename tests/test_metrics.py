import numpy as np

from microc_foundation.metrics import classification_metrics, confusion_matrix


def test_confusion_matrix_uses_true_rows_and_predicted_columns():
    result = confusion_matrix(
        np.array([0, 0, 1, 2]),
        np.array([0, 1, 1, 2]),
        num_classes=3,
    )
    assert result.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_classification_metrics_reports_macro_f1_and_support():
    result = classification_metrics(
        np.array([0, 0, 1, 2]),
        np.array([0, 1, 1, 2]),
        ["A", "B", "C"],
    )
    assert result["accuracy"] == 0.75
    assert result["per_class"]["A"]["support"] == 2
    assert 0 < result["macro_f1"] < 1
    assert 0 < result["macro_recall"] <= 1
