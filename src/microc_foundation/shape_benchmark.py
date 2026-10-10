"""Exploratory spatial texture features after within-window distance centering."""

from __future__ import annotations

from contextlib import ExitStack

import numpy as np
import pandas as pd

from .discovery_benchmark import binary_metrics, fit_logistic, predict_logistic, spatial_splits
from .io import open_cooler


def pooled_shape_features(
    counts: np.ndarray, *, blocks: int = 8, minimum_offset: int = 5,
) -> dict[str, np.ndarray]:
    """Pool upper-triangle log counts and unit-RMS distance-centered texture.

Each offset is centered independently within this one sample. Residual mean
and RMS in each spatial block retain position and local texture strength.
This is not genome-wide O/E, and per-window scaling does not remove all bias.
"""
    counts = np.asarray(counts, dtype=float)
    if (counts.ndim != 2 or counts.shape[0] != counts.shape[1] or
            not np.isfinite(counts).all() or (counts < 0).any() or
            not np.allclose(counts, counts.T)):
        raise ValueError("requires a finite nonnegative symmetric square matrix")
    size = len(counts)
    if blocks < 2 or size % blocks or not 1 <= minimum_offset <= size - 3:
        raise ValueError("invalid block count or distance offset")
    log_counts = np.log1p(counts)
    residual = np.zeros_like(log_counts)
    valid = np.zeros_like(log_counts, dtype=bool)
    for offset in range(minimum_offset, size - 2):
        rows = np.arange(size - offset)
        values = log_counts[rows, rows + offset]
        residual[rows, rows + offset] = values - values.mean()
        valid[rows, rows + offset] = True
    rms = float(np.sqrt(np.mean(residual[valid] ** 2)))
    # Pure distance envelopes legitimately have zero texture, not fake signal.
    normalized = residual / rms if rms > 1e-12 else np.zeros_like(residual)
    raw_means, residual_means, residual_rms = [], [], []
    width = size // blocks
    for row in range(blocks):
        for column in range(row, blocks):
            region = np.s_[row * width:(row + 1) * width,
                           column * width:(column + 1) * width]
            mask = valid[region]
            if not mask.any():
                raise ValueError("every upper spatial block must retain pixels")
            raw_means.append(float(log_counts[region][mask].mean()))
            values = normalized[region][mask]
            residual_means.append(float(values.mean()))
            residual_rms.append(float(np.sqrt(np.mean(values**2))))
    return {"raw_log": np.asarray(raw_means),
            "distance_texture": np.asarray(residual_means + residual_rms)}


def extract_replicate_shapes(
    paths: list[str], windows: pd.DataFrame, *, blocks: int = 8, minimum_offset: int = 5,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Read identical, bin-aligned local windows from two matching COOL files."""
    if len(paths) != 2 or windows.empty or windows.window_id.duplicated().any():
        raise ValueError("requires two replicates and unique nonempty windows")
    collected = {"raw_log": [], "distance_texture": []}
    with ExitStack() as stack:
        coolers = [stack.enter_context(open_cooler(path)) for path in paths]
        resolution = int(coolers[0].binsize)
        if any(int(cool.binsize) != resolution or
               not cool.chromsizes.equals(coolers[0].chromsizes) for cool in coolers):
            raise ValueError("replicate resolutions and references must agree")
        sources = [{"path": path, "binsize": int(cool.binsize), "sum": int(cool.info["sum"]),
                    "chromsizes": {str(k): int(v) for k, v in cool.chromsizes.items()}}
                   for path, cool in zip(paths, coolers)]
        for window in windows.itertuples(index=False):
            if (window.start % resolution or window.end % resolution or
                    window.end <= window.start or window.chrom not in coolers[0].chromnames or
                    window.start < 0 or window.end > coolers[0].chromsizes[window.chrom]):
                raise ValueError("window must be inside reference and bin aligned")
            region = f"{window.chrom}:{int(window.start)}-{int(window.end)}"
            features = []
            for cool in coolers:
                counts = np.asarray(cool.matrix(balance=False).fetch(region))
                size = (window.end - window.start) // resolution
                if counts.shape != (size, size):
                    raise ValueError("fetched shape differs from window interval")
                features.append(pooled_shape_features(
                    counts, blocks=blocks, minimum_offset=minimum_offset
                ))
            for name, values in collected.items():
                values.append(np.stack([item[name] for item in features]))
    return {name: np.stack(values) for name, values in collected.items()}, {
        "sources": sources, "blocks": blocks, "minimum_offset": minimum_offset,
        "minimum_distance_bp": minimum_offset * resolution,
        "feature_counts": {name: len(values[0][0]) for name, values in collected.items()},
    }


def evaluate_spatial_representation(
    training_features: np.ndarray, testing_features: np.ndarray, windows: pd.DataFrame,
    *, purge_bp: int = 20_480, permutations: int = 0, seed: int = 2026,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit each held-out genomic fold, optionally transferring between replicates."""
    training_features = np.asarray(training_features, dtype=float)
    testing_features = np.asarray(testing_features, dtype=float)
    if (training_features.shape != testing_features.shape or training_features.ndim != 2 or
            len(training_features) != len(windows) or permutations < 0):
        raise ValueError("feature arrays must match window rows and each other")
    labels = windows.label.to_numpy(int)
    splits = list(spatial_splits(windows, purge_bp=purge_bp))
    scores = np.full(len(windows), np.nan)
    folds = []
    for fold, train, test in splits:
        _, fitted = fit_logistic(training_features[train], labels[train])
        scores[test] = predict_logistic(testing_features[test], fitted)
        folds.append({
            "fold": fold, "train_window_ids": windows.loc[train, "window_id"].tolist(),
            "test_window_ids": windows.loc[test, "window_id"].tolist(),
            "metrics": binary_metrics(labels[test], scores[test]), "fit": fitted,
        })
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(permutations):
        shuffled = labels.copy()
        for _, _, test in splits:
            shuffled[test] = rng.permutation(labels[test])
        aucs = []
        for _, train, test in splits:
            _, fitted = fit_logistic(training_features[train], shuffled[train])
            aucs.append(binary_metrics(
                shuffled[test], predict_logistic(testing_features[test], fitted)
            )["roc_auc"])
        null.append(float(np.mean(aucs)))
    return scores, {
        "folds": folds, "mean_fold_roc_auc": float(np.mean([
            item["metrics"]["roc_auc"] for item in folds
        ])), "sd_fold_roc_auc": float(np.std([
            item["metrics"]["roc_auc"] for item in folds
        ], ddof=1)), "pooled_oof_metrics": binary_metrics(labels, scores),
        "shuffle_mean_fold_auc": null, "shuffle_seed": seed,
        "shuffle_2_5_percentile": float(np.percentile(null, 2.5)) if null else None,
        "shuffle_97_5_percentile": float(np.percentile(null, 97.5)) if null else None,
    }
