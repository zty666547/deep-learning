"""Spatially held-out checks of scan coverage and Task 2 feature separability.

Unannotated windows are background, not verified biological negatives. This
benchmark does not change the unsupervised candidate selection protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microc_foundation.clustering import (
    annotate_known_structures,
    build_model_features,
    robust_scale,
)


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    """Compute tie-aware ROC AUC and fixed-0.5-threshold binary metrics."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if (labels.ndim != 1 or scores.shape != labels.shape or
            not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all() or
            not ((scores >= 0) & (scores <= 1)).all()):
        raise ValueError("expected binary labels and finite probability scores")
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        raise ValueError("both classes are required for ROC AUC")
    comparisons = positive[:, None] - negative[None, :]
    auc = float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))
    predictions = scores >= 0.5
    recall = float(predictions[labels == 1].mean())
    specificity = float((~predictions[labels == 0]).mean())
    return {
        "n": len(labels), "positive": len(positive), "background": len(negative),
        "roc_auc": auc, "recall": recall, "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "accuracy": float((predictions == labels).mean()),
    }


def fit_logistic(
    features: np.ndarray, labels: np.ndarray, *, penalty: float = 0.01,
) -> tuple[np.ndarray, dict[str, object]]:
    """Class-balanced L2 logistic regression, with train-only median/IQR scaling.

The regularizer and decision threshold are fixed, not chosen on held-out data.
Newton steps use backtracking on stable log-loss; the intercept is unpenalized.
"""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    if (features.ndim != 2 or labels.shape != (len(features),) or
            not np.isfinite(features).all() or not np.isin(labels, [0, 1]).all() or
            len(np.unique(labels)) != 2 or penalty <= 0):
        raise ValueError("expected finite features, both binary classes, and positive penalty")
    scaled, median, scale = robust_scale(features)
    design = np.column_stack((np.ones(len(features)), scaled))
    counts = np.bincount(labels.astype(int), minlength=2)
    weights = 1 / (2 * counts[labels.astype(int)])
    regularizer = np.full(design.shape[1], penalty)
    regularizer[0] = 0
    coefficients = np.zeros(design.shape[1])

    def objective(values: np.ndarray) -> float:
        logits = np.einsum("ij,j->i", design, values, optimize=False)
        return float(np.sum(weights * (np.logaddexp(0, logits) - labels * logits)) +
                     0.5 * np.sum(regularizer * values**2))

    converged = False
    for iteration in range(100):
        logits = np.einsum("ij,j->i", design, coefficients, optimize=False)
        probabilities = np.exp(-np.logaddexp(0, -logits))
        gradient = np.einsum(
            "ij,i->j", design, weights * (probabilities - labels), optimize=False
        ) + regularizer * coefficients
        if np.max(np.abs(gradient)) < 1e-8:
            converged = True
            break
        curvature = weights * probabilities * (1 - probabilities)
        hessian = np.einsum(
            "ni,n,nj->ij", design, curvature, design, optimize=False
        ) + np.diag(regularizer)
        step = np.linalg.solve(hessian, gradient)
        previous_loss = objective(coefficients)
        rate = 1.0
        for _ in range(40):
            updated = coefficients - rate * step
            if objective(updated) <= previous_loss - 1e-4 * rate * float(
                np.einsum("i,i->", gradient, step, optimize=False)
            ):
                coefficients = updated
                break
            rate /= 2
        else:
            raise RuntimeError("logistic backtracking failed")
    if not converged:
        raise RuntimeError("logistic fit did not converge")
    return coefficients, {
        "median": median.tolist(), "scale": scale.tolist(),
        "coefficients": coefficients.tolist(), "iterations": iteration + 1,
        "penalty": penalty, "converged": converged,
    }


def predict_logistic(features: np.ndarray, fit: dict[str, object]) -> np.ndarray:
    values = (np.asarray(features, dtype=float) - np.asarray(fit["median"])) / np.asarray(
        fit["scale"]
    )
    coefficients = np.asarray(fit["coefficients"])
    logits = coefficients[0] + np.einsum("ij,j->i", values, coefficients[1:], optimize=False)
    return np.exp(-np.logaddexp(0, -logits))


def select_nonoverlapping_windows(
    windows: pd.DataFrame, structures: pd.DataFrame, *, chrom: str,
    chrom_length: int, window_bp: int = 20_480, folds: int = 5,
) -> pd.DataFrame:
    """Use a fixed origin-aligned grid, QC pass, and contiguous genomic folds."""
    required = {"window_id", "chrom", "start", "end", "center", "quality_pass"}
    if (not required.issubset(windows.columns) or folds < 2 or
            chrom_length <= 0 or window_bp <= 0):
        raise ValueError("invalid windows or grid configuration")
    quality = windows.quality_pass
    if quality.isna().any() or not quality.isin([True, False]).all():
        raise ValueError("quality_pass must contain booleans")
    if (windows.window_id.duplicated().any() or (windows.start < 0).any() or
            (windows.end <= windows.start).any()):
        raise ValueError("invalid window IDs or intervals")
    if set(windows.chrom.astype(str)) != {chrom}:
        raise ValueError("benchmark requires one matching chromosome")
    if (windows.end > chrom_length).any():
        raise ValueError("window exceeds chromosome")
    if set(structures.chrom.astype(str)) != {chrom} or (structures.end > chrom_length).any():
        raise ValueError("annotations do not match benchmark chromosome")
    if not np.allclose(windows.center, (windows.start + windows.end) / 2, atol=0.5):
        raise ValueError("window centers must match interval midpoints")
    selected = windows.loc[
        quality & ((windows.start % window_bp) == 0) &
        ((windows.end - windows.start) == window_bp)
    ].sort_values("start").reset_index(drop=True)
    if selected.empty or (selected.start.to_numpy()[1:] < selected.end.to_numpy()[:-1]).any():
        raise ValueError("selected grid is empty or overlaps")
    selected = annotate_known_structures(selected, structures)
    selected["label"] = (selected.known_overlap_bp > 0).astype(int)
    selected["fold"] = np.floor(selected.center / chrom_length * folds).astype(int)
    if set(selected.fold) != set(range(folds)):
        raise ValueError("every genomic fold must have windows")
    return selected


def spatial_splits(windows: pd.DataFrame, *, purge_bp: int = 20_480):
    """Leave out each block and remove training windows within its buffer."""
    if purge_bp < 0:
        raise ValueError("purge buffer must be nonnegative")
    for fold in sorted(windows.fold.unique()):
        test = windows.fold.to_numpy() == fold
        left = int(windows.loc[test, "start"].min()) - purge_bp
        right = int(windows.loc[test, "end"].max()) + purge_bp
        train = (~test) & ((windows.end.to_numpy() <= left) |
                          (windows.start.to_numpy() >= right))
        if len(np.unique(windows.loc[train, "label"])) != 2 or len(
            np.unique(windows.loc[test, "label"])
        ) != 2:
            raise ValueError("each train/test fold must contain both classes")
        yield int(fold), train, test


def structure_coverage(structures: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Fraction of each annotation covered by the UNION of supplied windows."""
    records = []
    for item in structures.itertuples(index=False):
        matching = windows.loc[
            (windows.chrom == item.chrom) & (windows.end > item.start) &
            (windows.start < item.end), ["start", "end"]
        ].sort_values("start")
        covered = 0
        previous_end = int(item.start)
        for interval in matching.itertuples(index=False):
            left = max(int(interval.start), int(item.start), previous_end)
            right = min(int(interval.end), int(item.end))
            covered += max(0, right - left)
            previous_end = max(previous_end, right)
        records.append({
            "structure_id": item.structure_id, "structure_type": item.structure_type,
            "chrom": item.chrom, "start": int(item.start), "end": int(item.end),
            "covered_bp": covered, "coverage_fraction": covered / (item.end - item.start),
        })
    return pd.DataFrame(records)


def benchmark_features(
    windows: pd.DataFrame, *, purge_bp: int = 20_480,
    permutations: int = 100, seed: int = 2026,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate fixed feature sets with out-of-fold predictions and a shuffle control."""
    if permutations < 0:
        raise ValueError("permutations must be nonnegative")
    matrix, names = build_model_features(windows)
    labels = windows.label.to_numpy(int)
    splits = list(spatial_splits(windows, purge_bp=purge_bp))
    feature_sets = {"density_only": [0, 1], "shape_repeat": list(range(2, 8)),
                    "all_features": list(range(8))}
    predictions = windows.copy()
    models = {}
    for name, indices in feature_sets.items():
        selected = matrix[:, indices]
        scores = np.full(len(windows), np.nan)
        fold_results = []
        for fold, train, test in splits:
            _, fitted = fit_logistic(selected[train], labels[train])
            scores[test] = predict_logistic(selected[test], fitted)
            fold_results.append({
                "fold": fold, "train_n": int(train.sum()),
                "train_window_ids": windows.loc[train, "window_id"].tolist(),
                "test_window_ids": windows.loc[test, "window_id"].tolist(),
                "metrics": binary_metrics(labels[test], scores[test]), "fit": fitted,
            })
        predictions[f"{name}_probability"] = scores
        aucs = [item["metrics"]["roc_auc"] for item in fold_results]
        balanced = [item["metrics"]["balanced_accuracy"] for item in fold_results]
        models[name] = {
            "features": [names[index] for index in indices], "folds": fold_results,
            "mean_fold_roc_auc": float(np.mean(aucs)),
            "sd_fold_roc_auc": float(np.std(aucs, ddof=1)),
            "mean_fold_balanced_accuracy": float(np.mean(balanced)),
            "pooled_oof_metrics": binary_metrics(labels, scores),
        }
    rng = np.random.default_rng(seed)
    null_aucs = []
    for _ in range(permutations):
        shuffled = labels.copy()
        # Preserve each genomic block's positive/background count.
        for _, _, test in splits:
            shuffled[test] = rng.permutation(labels[test])
        fold_aucs = []
        for _, train, test in splits:
            _, fitted = fit_logistic(matrix[train], shuffled[train])
            fold_aucs.append(binary_metrics(
                shuffled[test], predict_logistic(matrix[test], fitted)
            )["roc_auc"])
        null_aucs.append(float(np.mean(fold_aucs)))
    summary = {
        "protocol": "task2-feature-benchmark-v1", "n": len(windows),
        "positive": int(labels.sum()), "unannotated_background": int((labels == 0).sum()),
        "label_rule": "any positive-length overlap with supplied known annotations",
        "purge_bp": purge_bp, "threshold": 0.5, "penalty": 0.01, "models": models,
        "constant_positive_baseline": binary_metrics(labels, np.ones(len(labels))),
        "shuffle_control": {
            "seed": seed, "permutations": permutations, "mean_fold_roc_auc": null_aucs,
            "null_mean": float(np.mean(null_aucs)) if null_aucs else None,
            "null_2_5_percentile": float(np.percentile(null_aucs, 2.5)) if null_aucs else None,
            "null_97_5_percentile": float(np.percentile(null_aucs, 97.5)) if null_aucs else None,
            "interpretation": "descriptive shuffled-label control; not biological significance",
        },
    }
    return predictions, summary
