"""Parameterised full-genome candidate scanning for Task 2."""

from __future__ import annotations

import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import pool_square_matrix
from .io import open_cooler
from .normalization import normalize_matrix
from .visualization import render_heatmap
from .windows import GenomicWindow, fetch_window


def _window_features(matrix: np.ndarray) -> dict[str, float]:
    """Extract robust, interpretable features from one pooled contact map."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("contact matrix must be square")
    missing_fraction = float((~np.isfinite(values)).mean())
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    size = values.shape[0]
    distances = np.abs(np.subtract.outer(np.arange(size), np.arange(size)))
    upper = values[np.triu_indices(size, k=1)]
    near = values[(distances > 0) & (distances <= max(2, size // 32))]
    far = values[distances >= max(4, size // 4)]
    diagonal = np.diag(values)
    mean_value = float(values.mean())
    symmetry_error = float(np.abs(values - values.T).mean() / (mean_value + 1e-8))
    distance_means = np.asarray(
        [values[distances == distance].mean() for distance in range(1, size)],
        dtype=float,
    )
    positive_distances = np.arange(1, size)[distance_means > 0]
    if len(positive_distances) >= 2:
        decay_slope = float(
            np.polyfit(
                np.log1p(positive_distances),
                np.log1p(distance_means[positive_distances - 1]),
                1,
            )[0]
        )
    else:
        decay_slope = 0.0
    return {
        "mean_contact": mean_value,
        "median_contact": float(np.median(upper)) if len(upper) else 0.0,
        "max_contact": float(values.max()),
        "diagonal_mean": float(diagonal.mean()),
        "near_diagonal_mean": float(near.mean()) if len(near) else 0.0,
        "far_contact_mean": float(far.mean()) if len(far) else 0.0,
        "near_far_ratio": float(near.mean() / (far.mean() + 1e-8)) if len(near) else 0.0,
        "symmetry_error": symmetry_error,
        "zero_fraction": float((values <= 0).mean()),
        "missing_fraction": missing_fraction,
        "distance_decay_slope": decay_slope,
    }


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (values - mean) / std, mean, std


def _pca(values: np.ndarray, n_components: int = 2) -> np.ndarray:
    centered = values - values.mean(axis=0)
    active = np.std(centered, axis=0) > 1e-8
    if not active.any():
        return np.zeros((len(values), min(n_components, 1)), dtype=float)
    _, _, right_singular_vectors = np.linalg.svd(
        centered[:, active], full_matrices=False
    )
    components = min(n_components, right_singular_vectors.shape[0])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        projected = centered[:, active] @ right_singular_vectors[:components].T
    return np.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)


def _kmeans(values: np.ndarray, num_clusters: int, max_iter: int = 100) -> np.ndarray:
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("k-means input must be a non-empty matrix")
    cluster_count = min(max(1, num_clusters), len(values))
    order = np.argsort(values[:, 0], kind="stable")
    seed_indices = order[np.linspace(0, len(values) - 1, cluster_count).astype(int)]
    centroids = values[seed_indices].copy()
    labels = np.zeros(len(values), dtype=int)
    for _ in range(max_iter):
        distances = ((values[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(cluster_count):
            members = values[labels == cluster]
            if len(members):
                centroids[cluster] = members.mean(axis=0)
    return labels


def _known_overlap_bp(
    chrom: str, start: int, end: int, structures: pd.DataFrame
) -> tuple[int, str]:
    subset = structures.loc[structures["chrom"].astype(str) == chrom]
    overlap_ids: list[str] = []
    overlap_bp = 0
    for row in subset.itertuples(index=False):
        overlap = max(0, min(end, int(row.end)) - max(start, int(row.start)))
        if overlap:
            overlap_bp += overlap
            overlap_ids.append(str(row.structure_id))
    return overlap_bp, ";".join(sorted(overlap_ids))


def _replicate_correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float).ravel()
    right = np.asarray(second, dtype=float).ravel()
    if np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def scan_genome_candidates(
    cool_paths: list[str],
    structures: pd.DataFrame,
    output_dir: str,
    *,
    window_size_bp: int = 20_480,
    step_bp: int = 20_480,
    pool_factor: int = 16,
    normalization: str = "log1p",
    num_clusters: int = 8,
    min_total_contacts: float = 0.0,
    max_missing_fraction: float = 0.5,
    min_replicate_correlation: float = 0.0,
    min_cluster_size: int = 2,
    top_k: int = 12,
) -> Path:
    """Scan all complete windows shared by replicates and write candidate tables."""

    required = {"structure_id", "chrom", "start", "end"}
    missing = sorted(required.difference(structures.columns))
    if missing:
        raise ValueError(f"structures are missing columns: {', '.join(missing)}")
    paths = [str(Path(path).expanduser()) for path in cool_paths]
    if len(paths) < 2:
        raise ValueError("at least two biological replicates are required")
    if window_size_bp <= 0 or step_bp <= 0 or pool_factor <= 0:
        raise ValueError("window size, step and pool factor must be positive")
    if not 0 <= max_missing_fraction <= 1:
        raise ValueError("max_missing_fraction must lie in [0, 1]")
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    # The current course data are single-chromosome; this loop also handles
    # additional chromosomes without changing coordinate conventions.
    with open_cooler(paths[0]) as first_cool:
        bin_size = int(first_cool.binsize)
        if window_size_bp % bin_size or step_bp % bin_size:
            raise ValueError("window size and step must be divisible by COOL bin size")
        chromosomes = [str(chrom) for chrom in first_cool.chromnames]
        chromosome_sizes = {str(chrom): int(first_cool.chromsizes[chrom]) for chrom in chromosomes}
    window_bins = window_size_bp // bin_size
    if window_bins % pool_factor:
        raise ValueError("window bin count must be divisible by pool_factor")

    windows: list[GenomicWindow] = []
    for chrom in chromosomes:
        chrom_length = chromosome_sizes[chrom]
        if chrom_length < window_size_bp:
            continue
        for start in range(0, chrom_length - window_size_bp + 1, step_bp):
            windows.append(GenomicWindow(chrom, start, start + window_size_bp))
    if not windows:
        raise ValueError("no complete genomic windows were generated")

    matrices_by_replicate: list[list[np.ndarray]] = [[] for _ in paths]
    feature_rows: list[dict[str, object]] = []
    with ExitStack() as stack:
        coolers = [stack.enter_context(open_cooler(path)) for path in paths]
        for window_index, window in enumerate(windows):
            row: dict[str, object] = {
                "window_id": f"window_{window_index + 1:05d}",
                "chrom": window.chrom,
                "start": window.start,
                "end": window.end,
                "center": (window.start + window.end) // 2,
            }
            pooled_replicates: list[np.ndarray] = []
            for replicate_index, cool in enumerate(coolers, start=1):
                matrix, current_bin_size = fetch_window(
                    cool, window, balance=False, fill_value=None
                )
                if current_bin_size != bin_size or matrix.shape != (window_bins, window_bins):
                    raise ValueError(f"unexpected matrix shape for {window.region}")
                missing_fraction = float((~np.isfinite(matrix)).mean())
                dense = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
                pooled = pool_square_matrix(dense, pool_factor, method="sum")
                pooled_replicates.append(normalize_matrix(pooled, method=normalization))
                features = _window_features(pooled)
                row[f"total_contacts_rep{replicate_index}"] = float(dense.sum())
                row[f"missing_fraction_rep{replicate_index}"] = missing_fraction
                for name, value in features.items():
                    row[f"rep{replicate_index}_{name}"] = value
                matrices_by_replicate[replicate_index - 1].append(pooled_replicates[-1])
            row["replicate_correlation"] = _replicate_correlation(
                pooled_replicates[0], pooled_replicates[1]
            )
            overlap_bp, overlap_ids = _known_overlap_bp(
                window.chrom, window.start, window.end, structures
            )
            row["known_overlap_bp"] = overlap_bp
            row["known_structure_ids"] = overlap_ids
            row["quality_pass"] = bool(
                all(
                    float(row[f"total_contacts_rep{index}"]) >= min_total_contacts
                    and float(row[f"missing_fraction_rep{index}"]) <= max_missing_fraction
                    for index in range(1, len(paths) + 1)
                )
            )
            feature_rows.append(row)

    frame = pd.DataFrame(feature_rows)
    feature_names = [
        name
        for name in frame.columns
        if name.startswith(("rep1_", "rep2_"))
    ]
    quality_mask = frame["quality_pass"].to_numpy(bool)
    if quality_mask.sum() < max(2, num_clusters):
        raise ValueError("too few quality-passing windows for clustering")
    feature_values = frame.loc[quality_mask, feature_names].to_numpy(float)
    standardized, feature_mean, feature_std = _standardize(feature_values)
    coordinates = _pca(standardized, n_components=2)
    labels = _kmeans(standardized, num_clusters)
    frame["pca1"] = np.nan
    frame["pca2"] = np.nan
    frame["cluster"] = -1
    frame.loc[quality_mask, "pca1"] = coordinates[:, 0]
    frame.loc[quality_mask, "pca2"] = coordinates[:, 1] if coordinates.shape[1] > 1 else 0.0
    frame.loc[quality_mask, "cluster"] = labels
    cluster_sizes = Counter(labels.tolist())
    frame["cluster_size"] = frame["cluster"].map(cluster_sizes).fillna(0).astype(int)
    frame["candidate_score"] = (
        frame["replicate_correlation"].clip(lower=0)
        * np.log1p(frame["cluster_size"])
    )
    frame["passes_candidate_rule"] = (
        frame["quality_pass"]
        & (frame["known_overlap_bp"] == 0)
        & (frame["replicate_correlation"] >= min_replicate_correlation)
        & (frame["cluster_size"] >= min_cluster_size)
    )

    candidates = frame.loc[frame["passes_candidate_rule"]].copy()
    candidates = candidates.sort_values(
        ["candidate_score", "replicate_correlation"], ascending=False
    ).head(top_k)
    candidates.insert(
        0,
        "candidate_region_id",
        [f"candidate_{index:03d}" for index in range(1, len(candidates) + 1)],
    )
    frame.to_csv(output / "genome_windows.csv", index=False)
    candidates.to_csv(output / "candidate_regions.csv", index=False)
    with (output / "feature_scaler.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "feature_names": feature_names,
                "mean": feature_mean.tolist(),
                "std": feature_std.tolist(),
                "pca_components": coordinates.shape[1],
            },
            handle,
            indent=2,
        )

    heatmap_dir = output / "candidate_heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    first_matrices = matrices_by_replicate[0]
    second_matrices = matrices_by_replicate[1]
    for candidate in candidates.itertuples(index=False):
        index = int(str(candidate.window_id).split("_")[-1]) - 1
        window = GenomicWindow(str(candidate.chrom), int(candidate.start), int(candidate.end))
        for replicate_name, matrix in (
            ("rep1", first_matrices[index]),
            ("rep2", second_matrices[index]),
        ):
            render_heatmap(
                matrix,
                window,
                str(heatmap_dir / f"{candidate.candidate_region_id}_{replicate_name}.png"),
                title=f"{candidate.candidate_region_id} {replicate_name} cluster={candidate.cluster}",
            )

    summary = {
        "num_windows": len(frame),
        "num_quality_pass": int(quality_mask.sum()),
        "num_candidate_regions": len(candidates),
        "num_clusters": int(frame.loc[quality_mask, "cluster"].nunique()),
        "cluster_sizes": {str(key): int(value) for key, value in sorted(cluster_sizes.items())},
        "window_size_bp": window_size_bp,
        "step_bp": step_bp,
        "pool_factor": pool_factor,
        "normalization": normalization,
        "min_total_contacts": min_total_contacts,
        "max_missing_fraction": max_missing_fraction,
        "min_replicate_correlation": min_replicate_correlation,
        "min_cluster_size": min_cluster_size,
        "candidate_rule": "quality_pass AND no known overlap AND replicate correlation threshold AND cluster size threshold",
        "candidate_interpretation": "ranked candidates for review, not validated new biological structures",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output
