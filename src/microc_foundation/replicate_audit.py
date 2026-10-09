"""Descriptive replication audit after removing short-range and distance trends."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import open_cooler


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or left.std() < 1e-12 or right.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def distance_controlled_correlations(
    first: np.ndarray, second: np.ndarray, *, minimum_offset: int = 5,
) -> dict[str, float]:
    """Correlate log1p maps before/after removing each distance's mean.

    Centering is performed separately per replicate inside each window. It
    removes the within-window distance envelope; it is NOT genome-wide O/E.
    Undefined correlations (constant maps/residuals) are NaN, not fake zeros.
    """
    x, y = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if x.ndim != 2 or x.shape != y.shape or x.shape[0] != x.shape[1]:
        raise ValueError("requires equal square matrices")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (x < 0).any() or (y < 0).any():
        raise ValueError("requires finite nonnegative contact counts")
    if minimum_offset < 1 or minimum_offset > len(x) - 3:
        raise ValueError("offset must exclude the main diagonal and leave at least three pixels")
    x, y = np.log1p(x), np.log1p(y)
    triangle = np.triu_indices(len(x))
    selected_x, selected_y, residual_x, residual_y = [], [], [], []
    # Each retained offset has at least three pixels; tiny outer diagonals
    # would produce unstable/degenerate distance-wise moments.
    for offset in range(minimum_offset, len(x) - 2):
        a, b = np.diagonal(x, offset), np.diagonal(y, offset)
        selected_x.append(a)
        selected_y.append(b)
        residual_x.append(a - a.mean())
        residual_y.append(b - b.mean())
    return {
        "full_log_correlation": _pearson(x[triangle], y[triangle]),
        "offdiagonal_log_correlation": _pearson(np.concatenate(selected_x), np.concatenate(selected_y)),
        "distance_centered_correlation": _pearson(np.concatenate(residual_x), np.concatenate(residual_y)),
    }


def plot_replication_audit(table: pd.DataFrame, output: Path, minimum_distance_bp: int) -> None:
    """Plot descriptive distributions on shared axes with readable group labels."""
    names = sorted(table["group"].unique())
    short_names = {"candidate_representative": "Candidate", "candidate_neighborhood": "Neighbor",
                   "known_overlap": "Known", "unannotated_background": "Unannotated"}
    labels = [f"{short_names[name]}\nn={int((table['group'] == name).sum())}" for name in names]
    metrics = ("full_log_correlation", "offdiagonal_log_correlation", "distance_centered_correlation")
    titles = ("Full log-count r", f"r excluding <{minimum_distance_bp} bp", "Distance-centered r")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for axis, metric, title in zip(axes, metrics, titles):
        values = [table.loc[table["group"] == name, metric].dropna().to_numpy() for name in names]
        axis.boxplot(values, showfliers=False)
        axis.set_xticks(np.arange(1, len(names) + 1))
        axis.set_xticklabels(labels)
        axis.set_title(title)
        axis.set_ylabel("Pearson r (descriptive)")
        axis.set_ylim(-1, 1)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def audit_replicate_consistency(
    cool_paths: list[str], windows: pd.DataFrame, candidates: pd.DataFrame,
    output_dir: str, *, minimum_offset: int = 5,
) -> dict:
    """Audit all QC-passed windows without reselecting candidates from results."""
    if len(cool_paths) != 2:
        raise ValueError("requires exactly two biological replicates")
    if {"window_id", "chrom", "start", "end", "quality_pass", "known_overlap_bp"}.difference(windows.columns):
        raise ValueError("window table is missing required audit columns")
    if {"candidate_region_id", "chrom", "start", "end", "representative_window_id"}.difference(candidates.columns):
        raise ValueError("candidate table is missing required audit columns")
    if windows["window_id"].duplicated().any() or candidates["candidate_region_id"].duplicated().any():
        raise ValueError("window and candidate identifiers must be unique")
    quality = windows["quality_pass"].astype(str).str.lower().eq("true")
    selected = windows.loc[quality].copy()
    if selected.empty or candidates.empty:
        raise ValueError("requires QC-passed windows and candidate regions")
    representatives = set(candidates["representative_window_id"].astype(str))
    if not representatives.issubset(set(selected["window_id"].astype(str))):
        raise ValueError("candidate representative missing from QC-passed windows")
    lookup = selected.set_index("window_id")
    for candidate in candidates.itertuples(index=False):
        window = lookup.loc[candidate.representative_window_id]
        if candidate.chrom != window.chrom or candidate.start > window.start or candidate.end < window.end:
            raise ValueError("candidate coordinates must contain its representative window")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with ExitStack() as stack:
        coolers = [stack.enter_context(open_cooler(path)) for path in cool_paths]
        bin_size = int(coolers[0].binsize)
        if any(int(cool.binsize) != bin_size or not cool.chromsizes.equals(coolers[0].chromsizes) for cool in coolers):
            raise ValueError("replicate resolutions and chromosome lengths must agree")
        for window in selected.itertuples(index=False):
            if window.start % bin_size or window.end % bin_size or window.end <= window.start:
                raise ValueError("windows must be positive and bin aligned")
            region = f"{window.chrom}:{int(window.start)}-{int(window.end)}"
            matrices = [np.asarray(cool.matrix(balance=False).fetch(region)) for cool in coolers]
            expected_size = (int(window.end) - int(window.start)) // bin_size
            if any(matrix.shape != (expected_size, expected_size) for matrix in matrices):
                raise ValueError("matrix shape does not match requested coordinates")
            near_candidate = (
                (candidates["chrom"] == window.chrom)
                & (candidates["start"] < window.end) & (candidates["end"] > window.start)
            ).any()
            group = (
                "candidate_representative" if str(window.window_id) in representatives else
                "candidate_neighborhood" if near_candidate else
                "known_overlap" if window.known_overlap_bp > 0 else "unannotated_background"
            )
            rows.append({
                "window_id": window.window_id, "chrom": window.chrom,
                "start": int(window.start), "end": int(window.end), "group": group,
                **distance_controlled_correlations(*matrices, minimum_offset=minimum_offset),
            })
    table = pd.DataFrame(rows)
    table.to_csv(output / "window_replication_audit.csv", index=False)
    registry = candidates[["candidate_region_id", "chrom", "start", "end", "representative_window_id"]].copy()
    registry["stable_candidate_key"] = [
        f"task2-main-v1:{row.chrom}:{int(row.start)}-{int(row.end)}"
        for row in registry.itertuples(index=False)
    ]
    registry = registry.merge(table, left_on="representative_window_id", right_on="window_id", suffixes=("", "_window"), validate="one_to_one")
    registry.to_csv(output / "candidate_registry.csv", index=False)
    metrics = ("full_log_correlation", "offdiagonal_log_correlation", "distance_centered_correlation")
    groups = {}
    for name, group in table.groupby("group"):
        groups[name] = {"num_windows": len(group), "metrics": {}}
        for metric in metrics:
            values = group[metric].to_numpy(float)
            finite = values[np.isfinite(values)]
            groups[name]["metrics"][metric] = {
                "num_defined": len(finite), "median": float(np.median(finite)) if len(finite) else None,
                "q25": float(np.quantile(finite, 0.25)) if len(finite) else None,
                "q75": float(np.quantile(finite, 0.75)) if len(finite) else None,
            }
    summary = {
        "protocol": "task2-main-v1 descriptive within-window distance centering",
        "bin_size_bp": bin_size, "minimum_offset_bins": minimum_offset,
        "minimum_distance_bp": minimum_offset * bin_size,
        "num_windows": len(table), "num_candidate_regions": len(registry), "groups": groups,
        "limitations": ["No candidate reselection", "Background is not a biological negative control",
                        "Overlapping windows are not independent", "Not genome-wide observed/expected normalization"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    plot_replication_audit(table, output / "replication_distance_controls.png", minimum_offset * bin_size)
    return summary
