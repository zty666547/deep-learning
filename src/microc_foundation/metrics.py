"""Dependency-light classification metrics for the Task 1 baseline."""

from __future__ import annotations

import numpy as np


def confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Return a rows=true, columns=predicted integer confusion matrix."""

    true = np.asarray(targets, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    if true.shape != predicted.shape:
        raise ValueError("targets and predictions must have the same shape")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if true.ndim != 1:
        raise ValueError("targets and predictions must be one-dimensional")
    if ((true < 0) | (true >= num_classes)).any():
        raise ValueError("target index is outside the class range")
    if ((predicted < 0) | (predicted >= num_classes)).any():
        raise ValueError("prediction index is outside the class range")

    matrix = np.zeros((num_classes, num_classes), dtype="int64")
    np.add.at(matrix, (true, predicted), 1)
    return matrix


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    """Calculate accuracy, macro F1 and per-class precision/recall/F1."""

    matrix = confusion_matrix(targets, predictions, len(class_names))
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, class_name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted_count = int(matrix[:, index].sum())
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted_count if predicted_count else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_class[class_name] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "macro_recall": (
            float(np.mean([values["recall"] for values in per_class.values()]))
            if per_class
            else 0.0
        ),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
