"""Candidate-excluded, single-replicate frozen partition checks (not novelty proof)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .clustering import kmeans, principal_components, robust_scale

SINGLE_FEATURES = (
    "total_contacts", "nonzero_fraction", "near_far_ratio", "decay_slope",
    "center_enrichment", "coefficient_of_variation", "contact_entropy",
)


def single_replicate_features(frame: pd.DataFrame) -> np.ndarray:
    """No pooled replicate means or replicate-correlation features are allowed."""
    values = frame[list(SINGLE_FEATURES)].to_numpy(float).copy()
    if not np.isfinite(values).all() or (values[:, [0, 2]] < 0).any():
        raise ValueError("single-replicate features must be finite with nonnegative counts/ratios")
    values[:, [0, 2]] = np.log1p(values[:, [0, 2]])
    return values


def candidate_guard_mask(windows: pd.DataFrame, candidates: pd.DataFrame,
                         guard_bp: int = 20_480) -> np.ndarray:
    if guard_bp < 0:
        raise ValueError("guard must be nonnegative")
    protected = np.zeros(len(windows), dtype=bool)
    for row in candidates.itertuples(index=False):
        protected |= (windows.chrom.eq(row.chrom).to_numpy()
                      & windows.start.lt(int(row.end) + guard_bp).to_numpy()
                      & windows.end.gt(int(row.start) - guard_bp).to_numpy())
    return protected


def nonoverlapping_grid(windows: pd.DataFrame) -> pd.DataFrame:
    """Greedy coordinate-only thinning, independent of feature values or labels."""
    required = {"window_id", "chrom", "start", "end"}
    if (not required.issubset(windows) or windows.window_id.duplicated().any()
            or (windows.start < 0).any() or (windows.end <= windows.start).any()):
        raise ValueError("invalid or duplicate window coordinates")
    selected, previous_end = [], {}
    for index, row in windows.sort_values(["chrom", "start", "window_id"]).iterrows():
        if row.start >= previous_end.get(row.chrom, -1):
            selected.append(index)
            previous_end[row.chrom] = row.end
    return windows.loc[selected].reset_index(drop=True)


def fit_partition(features: np.ndarray, *, seed: int = 2026) -> dict[str, np.ndarray]:
    """Fit a separate diagnostic k=2/PC=3 partition on training rows only."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != 7 or len(x) < 8 or not np.isfinite(x).all():
        raise ValueError("need at least eight finite seven-feature training rows")
    scaled, median, scale = robust_scale(x)
    projected, components, ratios = principal_components(
        scaled, minimum_components=3, maximum_components=3,
    )
    labels, centers, _ = kmeans(projected, 2, seed=seed)
    if len(np.unique(labels)) != 2:
        raise ValueError("degenerate training partition")
    return {"median": median, "scale": scale, "scaled_mean": scaled.mean(axis=0),
            "components": components, "centers": centers, "variance_ratios": ratios}


def assign_partition(features: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(model["median"]) or not np.isfinite(x).all():
        raise ValueError("invalid evaluation features")
    scaled = (x - model["median"]) / model["scale"] - model["scaled_mean"]
    projected = np.einsum("ij,kj->ik", scaled, model["components"], optimize=False)
    distances = np.linalg.norm(projected[:, None, :] - model["centers"][None, :, :], axis=2)
    labels = distances.argmin(axis=1)
    ordered = np.sort(distances, axis=1)
    # Zero for ties; this is geometric margin, NOT class probability.
    margin = (ordered[:, 1] - ordered[:, 0]) / np.maximum(ordered.sum(axis=1), 1e-12)
    return labels, margin


def partition_agreement(first: np.ndarray, second: np.ndarray) -> dict:
    """Same frozen label space; ARI is invariant to arbitrary cluster numbering."""
    a, b = np.asarray(first), np.asarray(second)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError("need equal label vectors with at least two observations")
    ua, ai = np.unique(a, return_inverse=True)
    ub, bi = np.unique(b, return_inverse=True)
    counts = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(counts, (ai, bi), 1)
    def pairs(n):
        return np.sum(n * (n - 1) / 2)
    total = len(a) * (len(a) - 1) / 2
    expected = pairs(counts.sum(axis=0)) * pairs(counts.sum(axis=1)) / total
    upper = (pairs(counts.sum(axis=0)) + pairs(counts.sum(axis=1))) / 2
    ari = ((pairs(counts) - expected) / (upper - expected)
           if upper > expected else 1.0)
    return {"n": len(a), "agreement": float(np.mean(a == b)),
            "balanced_reference_agreement": float(np.mean([np.mean(b[a == x] == x) for x in ua])),
            "adjusted_rand_index": float(ari),
            "reference_cluster_counts": {str(x): int((a == x).sum()) for x in ua},
            "query_cluster_counts": {str(x): int((b == x).sum()) for x in ub}}


def circular_shift_control(first: np.ndarray, second: np.ndarray) -> dict:
    """All nonzero cyclic shifts; descriptive, not an independent-sample p-value."""
    observed = partition_agreement(first, second)
    controls = [partition_agreement(first, np.roll(second, shift))["adjusted_rand_index"]
                for shift in range(1, len(first))]
    return {**observed, "shift_count": len(controls),
            "shift_ari_quantiles": np.quantile(controls, [.025, .5, .975]).tolist(),
            "shift_ari_max": float(max(controls)),
            "shift_control_is_descriptive_not_p_value": True}
