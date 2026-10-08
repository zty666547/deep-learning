"""Dependency-light clustering and novelty rules for Task 2 windows."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_FEATURE_NAMES = (
    "log_total_contacts",
    "nonzero_fraction",
    "log_near_far_ratio",
    "decay_slope",
    "center_enrichment",
    "coefficient_of_variation",
    "contact_entropy",
    "replicate_correlation",
)


def build_model_features(windows: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build the explicitly transformed, interpretable clustering matrix."""

    source_columns = {
        "total_contacts_mean",
        "nonzero_fraction_mean",
        "near_far_ratio_mean",
        "decay_slope_mean",
        "center_enrichment_mean",
        "coefficient_of_variation_mean",
        "contact_entropy_mean",
        "replicate_correlation",
    }
    missing = sorted(source_columns.difference(windows.columns))
    if missing:
        raise ValueError(f"window table is missing columns: {', '.join(missing)}")
    matrix = np.column_stack(
        (
            np.log1p(windows["total_contacts_mean"].to_numpy(float)),
            windows["nonzero_fraction_mean"].to_numpy(float),
            np.log1p(windows["near_far_ratio_mean"].to_numpy(float)),
            windows["decay_slope_mean"].to_numpy(float),
            windows["center_enrichment_mean"].to_numpy(float),
            windows["coefficient_of_variation_mean"].to_numpy(float),
            windows["contact_entropy_mean"].to_numpy(float),
            windows["replicate_correlation"].to_numpy(float),
        )
    )
    if not np.isfinite(matrix).all():
        raise ValueError("model features must be finite")
    return matrix, list(MODEL_FEATURE_NAMES)


def robust_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale columns by median and interquartile range."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("matrix must be a non-empty two-dimensional array")
    median = np.median(values, axis=0)
    first_quartile, third_quartile = np.percentile(values, (25, 75), axis=0)
    scale = third_quartile - first_quartile
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (values - median) / scale, median, scale


def principal_components(
    matrix: np.ndarray,
    *,
    explained_variance_target: float = 0.9,
    minimum_components: int = 2,
    maximum_components: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA by SVD and retain enough components for the variance target."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("matrix must contain at least two rows")
    if not 0 < explained_variance_target <= 1:
        raise ValueError("explained variance target must lie in (0, 1]")
    centered = values - values.mean(axis=0)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2 / max(len(values) - 1, 1)
    ratios = variance / variance.sum() if variance.sum() else np.zeros_like(variance)
    required = int(np.searchsorted(np.cumsum(ratios), explained_variance_target) + 1)
    count = max(minimum_components, min(maximum_components, required, values.shape[1]))
    transformed = np.einsum(
        "ij,kj->ik", centered, components[:count], optimize=False
    )
    return transformed, components[:count], ratios[:count]


def _kmeans_once(
    matrix: np.ndarray,
    num_clusters: int,
    rng: np.random.Generator,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    first = int(rng.integers(len(matrix)))
    center_indices = [first]
    while len(center_indices) < num_clusters:
        distances = np.min(
            np.sum(
                (matrix[:, None, :] - matrix[center_indices][None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
        if distances.sum() <= 0:
            remaining = [index for index in range(len(matrix)) if index not in center_indices]
            center_indices.append(int(rng.choice(remaining)))
        else:
            center_indices.append(int(rng.choice(len(matrix), p=distances / distances.sum())))
    centers = matrix[center_indices].copy()
    labels = np.zeros(len(matrix), dtype=int)
    for _ in range(max_iterations):
        distances = np.sum((matrix[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        updated_labels = distances.argmin(axis=1)
        if np.array_equal(updated_labels, labels) and _ > 0:
            break
        labels = updated_labels
        for cluster in range(num_clusters):
            members = matrix[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
            else:
                centers[cluster] = matrix[int(rng.integers(len(matrix)))]
    inertia = float(np.sum((matrix - centers[labels]) ** 2))
    return labels, centers, inertia


def kmeans(
    matrix: np.ndarray,
    num_clusters: int,
    *,
    seed: int = 2026,
    n_init: int = 10,
    max_iterations: int = 200,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run deterministic multi-start K-means++ without an sklearn dependency."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or num_clusters < 2 or num_clusters >= len(values):
        raise ValueError("num_clusters must be between 2 and number of rows minus 1")
    if n_init <= 0 or max_iterations <= 0:
        raise ValueError("n_init and max_iterations must be positive")
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for initialization in range(n_init):
        rng = np.random.default_rng(seed + initialization)
        result = _kmeans_once(values, num_clusters, rng, max_iterations)
        if best is None or result[2] < best[2]:
            best = result
    assert best is not None
    return best


def silhouette_score(matrix: np.ndarray, labels: np.ndarray) -> float:
    """Calculate the mean silhouette score from Euclidean distances."""

    values = np.asarray(matrix, dtype=float)
    assignments = np.asarray(labels, dtype=int)
    if values.ndim != 2 or assignments.shape != (len(values),):
        raise ValueError("labels must match matrix rows")
    clusters = np.unique(assignments)
    if len(clusters) < 2:
        raise ValueError("silhouette requires at least two clusters")
    distances = np.sqrt(
        np.maximum(
            0.0,
            np.sum(values**2, axis=1)[:, None]
            + np.sum(values**2, axis=1)[None, :]
            - 2 * np.einsum("ij,kj->ik", values, values, optimize=False),
        )
    )
    scores = np.zeros(len(values), dtype=float)
    for index in range(len(values)):
        own_mask = assignments == assignments[index]
        own_count = int(own_mask.sum())
        if own_count <= 1:
            scores[index] = 0.0
            continue
        within = float(distances[index, own_mask].sum() / (own_count - 1))
        between = min(
            float(distances[index, assignments == cluster].mean())
            for cluster in clusters
            if cluster != assignments[index]
        )
        scores[index] = (between - within) / max(within, between, 1e-12)
    return float(scores.mean())


def annotate_known_structures(
    windows: pd.DataFrame, structures: pd.DataFrame
) -> pd.DataFrame:
    """Add union overlap, nearest-distance and known-label fields to windows."""

    required_windows = {"chrom", "start", "end", "center"}
    required_structures = {"structure_id", "chrom", "start", "end", "structure_type"}
    if required_windows.difference(windows.columns):
        raise ValueError("window table is missing genomic coordinates")
    if required_structures.difference(structures.columns):
        raise ValueError("structure table is missing required columns")
    result = windows.copy()
    overlap_bp: list[int] = []
    nearest_distance: list[int] = []
    known_ids: list[str] = []
    known_types: list[str] = []
    for window in result.itertuples(index=False):
        same_chrom = structures.loc[structures["chrom"].astype(str) == str(window.chrom)]
        overlapping = same_chrom.loc[
            (same_chrom["start"].astype(int) < int(window.end))
            & (same_chrom["end"].astype(int) > int(window.start))
        ]
        intervals = sorted(
            (
                max(int(window.start), int(row.start)),
                min(int(window.end), int(row.end)),
            )
            for row in overlapping.itertuples(index=False)
        )
        merged: list[list[int]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        overlap_bp.append(sum(end - start for start, end in merged))
        known_ids.append(";".join(overlapping["structure_id"].astype(str)))
        known_types.append(
            ";".join(sorted(overlapping["structure_type"].astype(str).unique()))
        )
        if len(overlapping):
            nearest_distance.append(0)
        elif len(same_chrom):
            distances = np.maximum(
                np.maximum(
                    same_chrom["start"].to_numpy(int) - int(window.end),
                    int(window.start) - same_chrom["end"].to_numpy(int),
                ),
                0,
            )
            nearest_distance.append(int(distances.min()))
        else:
            nearest_distance.append(-1)
    result["known_overlap_bp"] = overlap_bp
    result["known_overlap_fraction"] = (
        result["known_overlap_bp"]
        / (result["end"].astype(int) - result["start"].astype(int))
    )
    result["nearest_known_distance_bp"] = nearest_distance
    result["known_structure_ids"] = known_ids
    result["known_structure_types"] = known_types
    return result


def nearest_reference_distances(
    embedding: np.ndarray, reference_mask: np.ndarray
) -> np.ndarray:
    """Return Euclidean distance to the nearest known-overlapping window."""

    values = np.asarray(embedding, dtype=float)
    mask = np.asarray(reference_mask, dtype=bool)
    if values.ndim != 2 or mask.shape != (len(values),):
        raise ValueError("reference mask must match embedding rows")
    if not mask.any():
        raise ValueError("at least one reference window is required")
    references = values[mask]
    distances = np.empty(len(values), dtype=float)
    for start in range(0, len(values), 256):
        batch = values[start : start + 256]
        squared = np.sum((batch[:, None, :] - references[None, :, :]) ** 2, axis=2)
        distances[start : start + len(batch)] = np.sqrt(squared.min(axis=1))
    return distances


def merge_candidate_windows(
    candidates: pd.DataFrame,
    *,
    maximum_start_gap_bp: int,
    minimum_support: int,
) -> pd.DataFrame:
    """Merge adjacent selected windows and keep regions with repeated support."""

    if maximum_start_gap_bp <= 0 or minimum_support <= 0:
        raise ValueError("merge gap and minimum support must be positive")
    rows: list[dict[str, object]] = []
    ordered = candidates.sort_values(["chrom", "start", "end"])
    groups: list[list[object]] = []
    for row in ordered.itertuples(index=False):
        if (
            not groups
            or str(groups[-1][-1].chrom) != str(row.chrom)
            or int(row.start) - int(groups[-1][-1].start) > maximum_start_gap_bp
        ):
            groups.append([row])
        else:
            groups[-1].append(row)
    for group in groups:
        if len(group) < minimum_support:
            continue
        representative = max(group, key=lambda row: float(row.novelty_score))
        rows.append(
            {
                "candidate_region_id": f"candidate_{len(rows) + 1:03d}",
                "chrom": str(group[0].chrom),
                "start": min(int(row.start) for row in group),
                "end": max(int(row.end) for row in group),
                "num_supporting_windows": len(group),
                "representative_window_id": str(representative.window_id),
                "representative_center": int(representative.center),
                "cluster": int(representative.cluster),
                "novelty_score": float(representative.novelty_score),
                "replicate_correlation": float(representative.replicate_correlation),
                "nearest_known_distance_bp": int(
                    representative.nearest_known_distance_bp
                ),
                "supporting_window_ids": ";".join(
                    str(row.window_id) for row in group
                ),
            }
        )
    return pd.DataFrame(rows)


def cluster_and_select_candidates(
    windows: pd.DataFrame,
    structures: pd.DataFrame,
    *,
    k_min: int = 2,
    k_max: int = 8,
    seed: int = 2026,
    minimum_replicate_correlation: float = 0.9,
    novelty_quantile: float = 0.9,
    maximum_start_gap_bp: int = 5_120,
    minimum_support: int = 2,
) -> dict[str, object]:
    """Cluster windows and apply a transparent, strict candidate definition."""

    if not 0 <= minimum_replicate_correlation <= 1:
        raise ValueError("minimum replicate correlation must lie in [0, 1]")
    if not 0 < novelty_quantile < 1:
        raise ValueError("novelty quantile must lie in (0, 1)")
    quality = windows["quality_pass"].astype(str).str.lower().isin(("true", "1"))
    selected = windows.loc[quality].copy().reset_index(drop=True)
    if len(selected) <= k_max:
        raise ValueError("not enough quality windows for requested cluster range")

    raw_features, feature_names = build_model_features(selected)
    scaled, feature_median, feature_scale = robust_scale(raw_features)
    embedding, _components, variance_ratios = principal_components(scaled)
    cluster_scores: dict[int, float] = {}
    cluster_results: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    for num_clusters in range(k_min, k_max + 1):
        result = kmeans(embedding, num_clusters, seed=seed)
        cluster_results[num_clusters] = result
        cluster_scores[num_clusters] = silhouette_score(embedding, result[0])
    selected_k = max(cluster_scores, key=cluster_scores.get)
    labels, centers, inertia = cluster_results[selected_k]

    annotated = annotate_known_structures(selected, structures)
    annotated["cluster"] = labels
    for index in range(embedding.shape[1]):
        annotated[f"pc{index + 1}"] = embedding[:, index]
    known_mask = annotated["known_overlap_bp"].to_numpy(int) > 0
    novelty_scores = nearest_reference_distances(embedding, known_mask)
    annotated["novelty_score"] = novelty_scores
    eligible = (
        ~known_mask
        & (annotated["replicate_correlation"].to_numpy(float) >= minimum_replicate_correlation)
    )
    if not eligible.any():
        raise ValueError("no windows satisfy known-overlap and replicate criteria")
    novelty_threshold = float(np.quantile(novelty_scores[eligible], novelty_quantile))
    annotated["candidate_pass"] = eligible & (novelty_scores >= novelty_threshold)
    candidate_windows = annotated.loc[annotated["candidate_pass"]].copy()
    candidate_regions = merge_candidate_windows(
        candidate_windows,
        maximum_start_gap_bp=maximum_start_gap_bp,
        minimum_support=minimum_support,
    )
    cluster_counts = annotated["cluster"].value_counts().sort_index()
    cluster_known_fractions = (
        annotated.assign(known_overlap=known_mask)
        .groupby("cluster")["known_overlap"]
        .mean()
        .sort_index()
    )
    summary = {
        "num_quality_windows": len(annotated),
        "feature_names": feature_names,
        "feature_median": feature_median.tolist(),
        "feature_scale": feature_scale.tolist(),
        "pca_components": int(embedding.shape[1]),
        "pca_explained_variance_ratio": variance_ratios.tolist(),
        "pca_explained_variance_total": float(variance_ratios.sum()),
        "cluster_silhouette_scores": {
            str(key): value for key, value in cluster_scores.items()
        },
        "selected_num_clusters": int(selected_k),
        "selected_inertia": inertia,
        "cluster_centers": centers.tolist(),
        "cluster_counts": {str(key): int(value) for key, value in cluster_counts.items()},
        "cluster_known_overlap_fractions": {
            str(key): float(value) for key, value in cluster_known_fractions.items()
        },
        "num_no_known_overlap": int((~known_mask).sum()),
        "minimum_replicate_correlation": minimum_replicate_correlation,
        "novelty_quantile_among_eligible": novelty_quantile,
        "novelty_threshold": novelty_threshold,
        "num_candidate_windows_before_merge": len(candidate_windows),
        "maximum_start_gap_bp": maximum_start_gap_bp,
        "minimum_supporting_windows": minimum_support,
        "num_candidate_regions": len(candidate_regions),
    }
    return {
        "windows": annotated,
        "candidate_windows": candidate_windows,
        "candidate_regions": candidate_regions,
        "summary": summary,
    }


def write_clustering_results(result: dict[str, object], output_dir: str) -> Path:
    """Write clustering tables, summary and PCA overview."""

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    windows = result["windows"]
    candidates = result["candidate_windows"]
    regions = result["candidate_regions"]
    windows.to_csv(output / "window_clusters.csv", index=False)
    candidates.to_csv(output / "candidate_windows.csv", index=False)
    regions.to_csv(output / "candidate_regions.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    scatter = axis.scatter(
        windows["pc1"],
        windows["pc2"],
        c=windows["cluster"],
        cmap="tab10",
        s=14,
        alpha=0.55,
    )
    known = windows["known_overlap_bp"] > 0
    axis.scatter(
        windows.loc[known, "pc1"],
        windows.loc[known, "pc2"],
        facecolors="none",
        edgecolors="black",
        linewidths=0.25,
        s=18,
        label="Overlaps known",
    )
    axis.scatter(
        candidates["pc1"],
        candidates["pc2"],
        marker="*",
        c="red",
        edgecolors="black",
        linewidths=0.3,
        s=65,
        label="Candidate window",
    )
    axis.set(
        xlabel="PC1",
        ylabel="PC2",
        title="Task 2 window clusters and strict candidates",
    )
    axis.legend()
    figure.colorbar(scatter, ax=axis, label="Cluster")
    figure.savefig(output / "pca_clusters.png", dpi=180)
    plt.close(figure)
    return output
