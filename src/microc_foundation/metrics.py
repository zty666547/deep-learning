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


def multilabel_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    """Calculate per-label and aggregate metrics for binary label vectors."""

    true = np.asarray(targets, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    expected_shape = (true.shape[0], len(class_names)) if true.ndim == 2 else None
    if true.shape != predicted.shape:
        raise ValueError("targets and predictions must have the same shape")
    if true.ndim != 2 or true.shape != expected_shape:
        raise ValueError("targets must have one column per class name")
    if not np.isin(true, (0, 1)).all() or not np.isin(predicted, (0, 1)).all():
        raise ValueError("targets and predictions must be binary")

    per_label: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    total_tp = total_fp = total_fn = 0
    for index, class_name in enumerate(class_names):
        label_true = true[:, index]
        label_predicted = predicted[:, index]
        true_positive = int(((label_true == 1) & (label_predicted == 1)).sum())
        false_positive = int(((label_true == 0) & (label_predicted == 1)).sum())
        false_negative = int(((label_true == 1) & (label_predicted == 0)).sum())
        true_negative = int(((label_true == 0) & (label_predicted == 0)).sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        total_tp += true_positive
        total_fp += false_positive
        total_fn += false_negative
        per_label[class_name] = {
            "support": int(label_true.sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        }

    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "micro_f1": micro_f1,
        "exact_match_ratio": float(np.all(true == predicted, axis=1).mean()),
        "hamming_loss": float((true != predicted).mean()),
        "per_label": per_label,
    }
