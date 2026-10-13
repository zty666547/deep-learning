"""Paired fixed-test uncertainty, using genomic groups rather than seed/replicate rows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def genomic_groups(metadata: pd.DataFrame, *, block_bp: int = 100_000,
                   guard_bp: int = 0) -> np.ndarray:
    """Merge all fixed grid blocks touched by an interval (optionally expanded).

    Crossing intervals merge blocks transitively; overlapping windows can never
    be split between bootstrap units. These are spatial units, not independent
    biological replicates, and residual long-range dependence can remain.
    """
    needed = {"structure_id", "chrom", "window_start", "window_end"}
    if (not needed.issubset(metadata) or metadata.empty or block_bp <= 0 or guard_bp < 0
            or metadata.structure_id.duplicated().any()):
        raise ValueError("invalid metadata, block size or guard")
    for name in ("window_start", "window_end"):
        values = metadata[name].to_numpy(float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("coordinates must be finite integers")
    if (metadata.window_start < 0).any() or (metadata.window_end <= metadata.window_start).any():
        raise ValueError("invalid genomic intervals")
    groups = np.empty(len(metadata), dtype=int)
    next_group = 0
    ordered = metadata.reset_index(drop=True).sort_values(["chrom", "window_start", "window_end"])
    previous_chrom, highest_block = None, -1
    for index, row in ordered.iterrows():
        low = max(0, int(row.window_start) - guard_bp) // block_bp
        high = (int(row.window_end) + guard_bp - 1) // block_bp
        if row.chrom != previous_chrom or low > highest_block:
            current = next_group
            next_group += 1
            highest_block = high
        else:
            highest_block = max(highest_block, high)
        groups[index] = current
        previous_chrom = row.chrom
    return groups


def paired_group_bootstrap(delta: np.ndarray, groups: np.ndarray, *,
                           draws: int = 10_000, seed: int = 2026) -> dict:
    """Resample whole genomic groups with replacement, keeping methods paired.

    Statistic is the structure-weighted mean, not the equally weighted block
    mean. No p-value; an interval describes these fixed-test spatial units only.
    """
    values, labels = np.asarray(delta, dtype=float), np.asarray(groups)
    if (values.ndim != 1 or values.shape != labels.shape or not len(values)
            or not np.isfinite(values).all() or draws < 100):
        raise ValueError("need finite paired deltas, matching groups and >=100 draws")
    unique = np.unique(labels)
    result = {"structure_count": len(values), "group_count": len(unique),
              "mean_delta": float(values.mean()), "improved_fraction": float((values > 0).mean()),
              "draws": draws, "seed": seed, "interval_95": None,
              "leave_one_group_out_mean_range": None}
    if len(unique) < 2:
        result["status"] = "insufficient_genomic_groups"
        return result
    counts = np.array([(labels == g).sum() for g in unique])
    sums = np.array([values[labels == g].sum() for g in unique])
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(unique), size=(draws, len(unique)))
    means = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    leave_one_out = (values.sum() - sums) / (len(values) - counts)
    result.update(status="descriptive_fixed_test_group_bootstrap",
                  interval_95=np.quantile(means, [.025, .975]).tolist(),
                  leave_one_group_out_mean_range=[float(leave_one_out.min()), float(leave_one_out.max())],
                  group_sizes=counts.tolist())
    return result


def log_distance_residual_r(target: np.ndarray, prediction: np.ndarray,
                            minimum_offset: int = 10) -> float:
    """Pearson r on distance-centered *already log1p* maps; no second log transform."""
    a, b = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if (a.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or minimum_offset < 1 or minimum_offset > len(a) - 3):
        raise ValueError("need equal finite square log maps and a valid offset")
    x, y = [], []
    for offset in range(minimum_offset, len(a) - 2):
        u, v = np.diagonal(a, offset), np.diagonal(b, offset)
        x.append(u - u.mean())
        y.append(v - v.mean())
    u, v = np.concatenate(x), np.concatenate(y)
    if u.std() < 1e-12 or v.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(u, v)[0, 1])
