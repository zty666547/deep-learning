"""Distance- and annotation-aware checks for saved Task 5 reconstructions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_DISTANCE_BINS = (
    (10, 20, "0.8-1.6 kb"),
    (20, 40, "1.6-3.2 kb"),
    (40, 80, "3.2-6.4 kb"),
    (80, 128, "6.4-10.24 kb"),
    (128, 256, "10.24-20.48 kb"),
)


def _mse(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(np.mean((target[mask] - prediction[mask]) ** 2))


def _load_reconstruction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "structure_id", "label", "chrom", "window_start", "window_end",
            "annotation_start", "annotation_end", "replicate", "target", "bicubic", "cnn",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")
        result = {key: archive[key].copy() for key in required}
    target = result["target"]
    if target.ndim != 4 or target.shape[1] != 2 or target.shape[2] != target.shape[3]:
        raise ValueError("reconstruction maps must have shape (samples, 2, bins, bins)")
    if any(result[name].shape != target.shape for name in ("bicubic", "cnn")):
        raise ValueError("target, bicubic and CNN arrays must have equal shapes")
    if not all(np.isfinite(result[name]).all() for name in ("target", "bicubic", "cnn")):
        raise ValueError("reconstruction arrays must be finite")
    if len(result["structure_id"]) != len(target):
        raise ValueError("metadata rows must match reconstruction maps")
    return result


def evaluate_reconstructions(
    reconstruction_paths: list[str],
    output_dir: str,
    *,
    bin_size_bp: int = 80,
    near_diagonal_exclusion_bp: int = 160,
) -> dict[str, object]:
    """Compare bicubic and CNN MSE by genomic distance and known interval.

    Replicates and seeds are averaged within each structure ID before the
    across-structure summary. This avoids treating either as extra biological
    samples. Results remain descriptive for the fixed test split.
    """
    if not reconstruction_paths or bin_size_bp <= 0 or near_diagonal_exclusion_bp < 0:
        raise ValueError("provide reconstruction files and valid positive resolutions")
    loaded = [_load_reconstruction(Path(path)) for path in reconstruction_paths]
    reference = loaded[0]
    reference_ids = reference["structure_id"].astype(str)
    for item in loaded[1:]:
        if not np.array_equal(item["structure_id"].astype(str), reference_ids):
            raise ValueError("all seed files must contain the same ordered test IDs")
        for field in (
            "label", "chrom", "window_start", "window_end", "annotation_start",
            "annotation_end", "replicate",
        ):
            if not np.array_equal(item[field], reference[field]):
                raise ValueError(f"test metadata differs between seeds: {field}")

    seeds = []
    for path in reconstruction_paths:
        parent = Path(path).parent.name
        try:
            seeds.append(int(parent.removeprefix("seed_")))
        except ValueError:
            seeds.append(len(seeds))

    per_sample_distance: list[dict[str, object]] = []
    per_sample_structure: list[dict[str, object]] = []
    ids = reference_ids
    labels = reference["label"].astype(str)
    starts = reference["window_start"].astype(int)
    annotation_starts = reference["annotation_start"].astype(int)
    annotation_ends = reference["annotation_end"].astype(int)
    resolution = int(bin_size_bp)
    diagonal_bins = int(np.ceil(near_diagonal_exclusion_bp / resolution))

    for sample_index, structure_id in enumerate(ids):
        distance_values: dict[str, list[float]] = {
            name: [] for name in ("bicubic", "cnn")
        }
        structure_values: dict[str, list[float]] = {
            name: [] for name in ("bicubic", "cnn")
        }
        for seed_result in loaded:
            target = seed_result["target"][sample_index].mean(axis=0)
            predictions = {
                "bicubic": seed_result["bicubic"][sample_index].mean(axis=0),
                "cnn": seed_result["cnn"][sample_index].mean(axis=0),
            }
            rows, columns = np.triu_indices(target.shape[0], k=max(diagonal_bins, 1))
            offsets = columns - rows
            for low, high, name in _DISTANCE_BINS:
                mask = (offsets >= low) & (offsets < high)
                if not mask.any():
                    continue
                for method, prediction in predictions.items():
                    distance_values[method].append(
                        _mse(target[rows, columns], prediction[rows, columns], mask)
                    )

            ann_start, ann_end = annotation_starts[sample_index], annotation_ends[sample_index]
            if ann_start >= 0 and ann_end > ann_start:
                bin_starts = starts[sample_index] + np.arange(target.shape[0]) * resolution
                inside = (bin_starts < ann_end) & (bin_starts + resolution > ann_start)
                row_mask = inside[rows] & inside[columns] & (offsets >= max(diagonal_bins, 1))
                for method, prediction in predictions.items():
                    structure_values[method].append(
                        _mse(target[rows, columns], prediction[rows, columns], row_mask)
                    )

        for name, (low, high, distance_name) in enumerate(_DISTANCE_BINS):
            if distance_values["cnn"]:
                # Distances are appended in fixed bin order for each seed.
                index = name
                for method in ("bicubic", "cnn"):
                    values = distance_values[method][index::len(_DISTANCE_BINS)]
                    per_sample_distance.append({
                        "structure_id": structure_id,
                        "label": labels[sample_index],
                        "distance": distance_name,
                        "method": method,
                        "mse": float(np.nanmean(values)) if values else float("nan"),
                        "seed_count": len(values),
                    })
        if structure_values["cnn"]:
            for method in ("bicubic", "cnn"):
                values = np.asarray(structure_values[method], dtype=float)
                per_sample_structure.append({
                    "structure_id": structure_id,
                    "label": labels[sample_index],
                    "method": method,
                    "mse": float(np.nanmean(values)) if np.isfinite(values).any() else float("nan"),
                    "seed_count": int(np.isfinite(values).sum()),
                })

    distance_frame = pd.DataFrame(per_sample_distance)
    structure_frame = pd.DataFrame(per_sample_structure)
    if distance_frame.empty:
        raise ValueError("no evaluable distance strata found")
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    distance_frame.to_csv(output / "distance_metrics_by_sample.csv", index=False)
    structure_frame.to_csv(output / "known_structure_metrics_by_sample.csv", index=False)
    distance_summary = (
        distance_frame.groupby(["distance", "method"], sort=False)["mse"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    structure_summary = (
        structure_frame.groupby(["label", "method"])["mse"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    distance_summary.to_csv(output / "distance_metrics_summary.csv", index=False)
    structure_summary.to_csv(output / "known_structure_metrics_summary.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    ordered_distances = [name for _, _, name in _DISTANCE_BINS]
    x = np.arange(len(ordered_distances))
    width = 0.36
    for offset, method, color, label in (
        (-width / 2, "bicubic", "#6b7280", "Bicubic"),
        (width / 2, "cnn", "#1769aa", "CNN"),
    ):
        subset = distance_summary.loc[distance_summary["method"] == method].set_index("distance")
        axes[0].bar(
            x + offset,
            [subset.loc[name, "mean"] for name in ordered_distances],
            width,
            yerr=[subset.loc[name, "std"] for name in ordered_distances],
            color=color,
            label=label,
            capsize=3,
        )
    axes[0].set_xticks(x, ordered_distances, rotation=25, ha="right")
    axes[0].set_ylabel("Upper-triangle MSE (log1p counts)")
    axes[0].set_title("By genomic separation")
    axes[0].legend(frameon=False)
    ordered_labels = [label for label in ("CHID", "CHIN", "OPCID") if label in set(structure_summary["label"])]
    x = np.arange(len(ordered_labels))
    for offset, method, color, label in (
        (-width / 2, "bicubic", "#6b7280", "Bicubic"),
        (width / 2, "cnn", "#1769aa", "CNN"),
    ):
        subset = structure_summary.loc[structure_summary["method"] == method].set_index("label")
        axes[1].bar(
            x + offset,
            [subset.loc[name, "mean"] for name in ordered_labels],
            width,
            yerr=[subset.loc[name, "std"] for name in ordered_labels],
            color=color,
            label=label,
            capsize=3,
        )
    axes[1].set_xticks(x, [f"{name} (n={int(structure_summary.loc[structure_summary['label'] == name, 'count'].max())})" for name in ordered_labels])
    axes[1].set_ylabel("Within-annotation MSE (log1p counts)")
    axes[1].set_title("Pairs within known intervals")
    axes[1].legend(frameon=False)
    figure.savefig(output / "task5_structure_recovery.png", dpi=180)
    plt.close(figure)
    summary: dict[str, object] = {
        "status": "descriptive_structure_recovery_evaluation_completed",
        "input_files": [str(Path(path)) for path in reconstruction_paths],
        "seeds": seeds,
        "test_structure_count": len(ids),
        "replicate_aggregation": "average two channels within each map before MSE",
        "seed_aggregation": "average seed-specific MSE within each structure ID",
        "distance_bin_size_bp": resolution,
        "near_diagonal_exclusion_bp": diagonal_bins * resolution,
        "annotation_metric": "upper-triangle MSE for bin pairs both overlapping the known interval",
        "limitations": [
            "descriptive evaluation on one fixed test split",
            "test windows may be spatially related and are not independent biological replicates",
            "annotation coordinate origin follows the project normalized table and remains unverified at source",
            "same-condition derived low-resolution maps are not independent low-depth measurements",
        ],
        "distance_summary": distance_summary.to_dict(orient="records"),
        "known_structure_summary": structure_summary.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
