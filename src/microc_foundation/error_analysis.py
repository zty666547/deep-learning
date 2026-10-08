"""Cross-seed error analysis for Task 1 classification models."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import MicroCCNN


def summarize_prediction_stability(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    """Summarize correctness and agreement across repeated model runs."""

    target_values = np.asarray(targets, dtype=int)
    predicted_values = np.asarray(predictions, dtype=int)
    probability_values = np.asarray(probabilities, dtype=float)
    if target_values.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    if predicted_values.ndim != 2:
        raise ValueError("predictions must have shape runs by samples")
    if probability_values.ndim != 3:
        raise ValueError("probabilities must have shape runs by samples by classes")
    if predicted_values.shape != probability_values.shape[:2]:
        raise ValueError("prediction and probability shapes do not align")
    if predicted_values.shape[1] != len(target_values):
        raise ValueError("target and prediction sample counts do not align")
    if len(predicted_values) == 0:
        raise ValueError("at least one model run is required")

    correct_count = (predicted_values == target_values[None, :]).sum(axis=0)
    mean_probabilities = probability_values.mean(axis=0)
    consensus_prediction = mean_probabilities.argmax(axis=1)
    consensus_confidence = mean_probabilities.max(axis=1)
    mean_true_probability = mean_probabilities[
        np.arange(len(target_values)), target_values
    ]
    vote_count = np.apply_along_axis(
        lambda values: np.bincount(
            values, minlength=probability_values.shape[2]
        ).max(),
        axis=0,
        arr=predicted_values,
    )
    stability_group = np.full(len(target_values), "mixed", dtype="U16")
    stability_group[correct_count == 0] = "always_wrong"
    stability_group[correct_count == len(predicted_values)] = "always_correct"
    return {
        "correct_count": correct_count,
        "consensus_prediction": consensus_prediction,
        "consensus_confidence": consensus_confidence,
        "mean_true_probability": mean_true_probability,
        "prediction_agreement": vote_count / len(predicted_values),
        "stability_group": stability_group,
        "mean_probabilities": mean_probabilities,
    }


def find_cross_class_annotation_overlaps(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return positive-length annotation overlaps between different classes."""

    overlaps: list[dict[str, object]] = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if left["chrom"] != right["chrom"]:
                continue
            if left["true_class"] == right["true_class"]:
                continue
            overlap_start = max(
                int(left["annotation_start"]), int(right["annotation_start"])
            )
            overlap_end = min(
                int(left["annotation_end"]), int(right["annotation_end"])
            )
            if overlap_start < overlap_end:
                overlaps.append(
                    {
                        "left_structure_id": left["structure_id"],
                        "left_class": left["true_class"],
                        "right_structure_id": right["structure_id"],
                        "right_class": right["true_class"],
                        "overlap_bp": overlap_end - overlap_start,
                    }
                )
    return overlaps


def _load_checkpoint_predictions(
    matrices: np.ndarray,
    checkpoint_path: str,
) -> tuple[int, list[str], np.ndarray, np.ndarray]:
    checkpoint_file = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    class_names = [str(name) for name in checkpoint["class_names"]]
    input_shape = tuple(int(value) for value in checkpoint["input_shape"])
    if tuple(matrices.shape[1:]) != input_shape:
        raise ValueError(f"checkpoint input shape does not match {checkpoint_path}")
    model = MicroCCNN(matrices.shape[1], len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    channel_mean = np.asarray(checkpoint["channel_mean"], dtype="float32").reshape(
        1, -1, 1, 1
    )
    channel_std = np.asarray(checkpoint["channel_std"], dtype="float32").reshape(
        1, -1, 1, 1
    )
    standardized = (matrices - channel_mean) / channel_std
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(standardized)), dim=1)
    predictions = probabilities.argmax(dim=1).numpy()
    seed = checkpoint.get("seed")
    if seed is None:
        metrics_path = checkpoint_file.with_name("metrics.json")
        if not metrics_path.exists():
            raise ValueError(f"checkpoint seed is unavailable for {checkpoint_path}")
        seed = json.loads(metrics_path.read_text(encoding="utf-8"))["seed"]
    return int(seed), class_names, predictions, probabilities.numpy()


def _plot_stability_summary(
    rows: list[dict[str, object]],
    class_names: list[str],
    output_path: Path,
) -> None:
    groups = ("always_correct", "mixed", "always_wrong")
    colors = ("#2ca02c", "#ffbf00", "#d62728")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    bottoms = np.zeros(len(class_names), dtype=float)
    for group, color in zip(groups, colors):
        counts = np.asarray(
            [
                sum(
                    row["true_class"] == class_name
                    and row["stability_group"] == group
                    for row in rows
                )
                for class_name in class_names
            ],
            dtype=float,
        )
        axes[0].bar(class_names, counts, bottom=bottoms, label=group, color=color)
        bottoms += counts
    axes[0].set(
        xlabel="True class",
        ylabel="Test samples",
        title="Cross-seed correctness stability",
    )
    axes[0].legend()

    group_colors = dict(zip(groups, colors))
    for row in rows:
        axes[1].scatter(
            float(row["annotation_length_bp"]),
            float(row["mean_true_probability"]),
            color=group_colors[str(row["stability_group"])],
            alpha=0.8,
        )
    axes[1].set(
        xlabel="Annotation length (bp)",
        ylabel="Mean probability of true class",
        title="Length and true-class confidence",
        ylim=(0, 1),
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_chid_cases(
    matrices: np.ndarray,
    rows: list[dict[str, object]],
    structure_ids: np.ndarray,
    output_path: Path,
) -> None:
    chid_positions = [
        position for position, row in enumerate(rows) if row["true_class"] == "CHID"
    ]
    if not chid_positions:
        return
    index_by_id = {
        str(structure_id): index for index, structure_id in enumerate(structure_ids)
    }
    maps = [
        matrices[index_by_id[str(rows[position]["structure_id"])]].mean(axis=0)
        for position in chid_positions
    ]
    vmax = float(np.percentile(np.concatenate([value.ravel() for value in maps]), 99))
    figure, axes = plt.subplots(
        1, len(chid_positions), figsize=(4 * len(chid_positions), 4), constrained_layout=True
    )
    axes_values = np.atleast_1d(axes)
    for axis, contact_map, position in zip(axes_values, maps, chid_positions):
        row = rows[position]
        start = int(row["window_start"])
        end = int(row["window_end"])
        extent = (start, end, start, end)
        axis.imshow(
            contact_map,
            origin="lower",
            cmap="Reds",
            extent=extent,
            vmin=0,
            vmax=vmax,
        )
        annotation_start = int(row["annotation_start"])
        annotation_end = int(row["annotation_end"])
        for boundary in (annotation_start, annotation_end):
            axis.axvline(boundary, color="navy", linewidth=0.8, linestyle="--")
            axis.axhline(boundary, color="navy", linewidth=0.8, linestyle="--")
        axis.set_title(
            f"{row['structure_id']}\n{row['correct_count']}/{row['num_runs']} correct"
        )
        axis.tick_params(axis="x", rotation=30)
    figure.suptitle("CHID test cases across repeated models")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def analyze_repeated_model_errors(
    dataset_path: str,
    checkpoint_paths: list[str],
    output_dir: str,
) -> dict[str, object]:
    """Analyze per-sample test errors across multiple trained checkpoints."""

    if not checkpoint_paths:
        raise ValueError("at least one checkpoint is required")
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(Path(dataset_path).expanduser()) as dataset:
        matrices = dataset["matrices"].astype("float32")
        labels_text = dataset["labels"].astype(str)
        splits = dataset["split"].astype(str)
        structure_ids = dataset["structure_id"].astype(str)
        chrom = dataset["chrom"].astype(str)
        annotation_start = dataset["annotation_start"].astype("int64")
        annotation_end = dataset["annotation_end"].astype("int64")
        window_start = dataset["window_start"].astype("int64")
        window_end = dataset["window_end"].astype("int64")

    test_indices = np.flatnonzero(splits == "test")
    if not len(test_indices):
        raise ValueError("dataset contains no test samples")
    seeds: list[int] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    class_names: list[str] | None = None
    for checkpoint_path in checkpoint_paths:
        seed, current_classes, current_predictions, current_probabilities = (
            _load_checkpoint_predictions(matrices[test_indices], checkpoint_path)
        )
        if class_names is None:
            class_names = current_classes
        elif current_classes != class_names:
            raise ValueError("checkpoints do not use the same class order")
        seeds.append(seed)
        predictions.append(current_predictions)
        probabilities.append(current_probabilities)
    assert class_names is not None
    if len(set(seeds)) != len(seeds):
        raise ValueError("checkpoint seeds must be unique")

    class_to_index = {name: index for index, name in enumerate(class_names)}
    test_labels = labels_text[test_indices]
    targets = np.asarray([class_to_index[value] for value in test_labels], dtype=int)
    prediction_array = np.stack(predictions)
    probability_array = np.stack(probabilities)
    stability = summarize_prediction_stability(
        targets, prediction_array, probability_array
    )

    rows: list[dict[str, object]] = []
    for position, dataset_index in enumerate(test_indices):
        row: dict[str, object] = {
            "structure_id": str(structure_ids[dataset_index]),
            "true_class": str(test_labels[position]),
            "chrom": str(chrom[dataset_index]),
            "annotation_start": int(annotation_start[dataset_index]),
            "annotation_end": int(annotation_end[dataset_index]),
            "annotation_length_bp": int(
                annotation_end[dataset_index] - annotation_start[dataset_index]
            ),
            "window_start": int(window_start[dataset_index]),
            "window_end": int(window_end[dataset_index]),
            "num_runs": len(seeds),
            "correct_count": int(stability["correct_count"][position]),
            "stability_group": str(stability["stability_group"][position]),
            "consensus_prediction": class_names[
                int(stability["consensus_prediction"][position])
            ],
            "consensus_confidence": float(
                stability["consensus_confidence"][position]
            ),
            "mean_true_probability": float(
                stability["mean_true_probability"][position]
            ),
            "prediction_agreement": float(
                stability["prediction_agreement"][position]
            ),
        }
        for run_index, seed in enumerate(seeds):
            predicted_index = int(prediction_array[run_index, position])
            row[f"seed_{seed}_prediction"] = class_names[predicted_index]
            row[f"seed_{seed}_confidence"] = float(
                probability_array[run_index, position, predicted_index]
            )
            row[f"seed_{seed}_true_probability"] = float(
                probability_array[run_index, position, targets[position]]
            )
        rows.append(row)

    cross_class_overlaps = find_cross_class_annotation_overlaps(rows)
    overlap_neighbors: dict[str, list[str]] = {
        str(row["structure_id"]): [] for row in rows
    }
    for overlap in cross_class_overlaps:
        left_id = str(overlap["left_structure_id"])
        right_id = str(overlap["right_structure_id"])
        overlap_neighbors[left_id].append(right_id)
        overlap_neighbors[right_id].append(left_id)
    for row in rows:
        neighbors = sorted(overlap_neighbors[str(row["structure_id"])])
        row["cross_class_overlap_count"] = len(neighbors)
        row["cross_class_overlap_ids"] = ";".join(neighbors)

    rows.sort(
        key=lambda row: (
            int(row["correct_count"]),
            float(row["mean_true_probability"]),
            str(row["structure_id"]),
        )
    )
    with (output / "sample_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_class: dict[str, dict[str, object]] = {}
    for class_name in class_names:
        class_rows = [row for row in rows if row["true_class"] == class_name]
        group_counts = Counter(str(row["stability_group"]) for row in class_rows)
        per_class[class_name] = {
            "support": len(class_rows),
            "always_correct": group_counts["always_correct"],
            "mixed": group_counts["mixed"],
            "always_wrong": group_counts["always_wrong"],
            "consensus_accuracy": float(
                np.mean(
                    [row["consensus_prediction"] == class_name for row in class_rows]
                )
            ),
            "mean_true_probability": float(
                np.mean([row["mean_true_probability"] for row in class_rows])
            ),
        }
    overall_groups = Counter(str(row["stability_group"]) for row in rows)
    always_wrong_rows = [
        row for row in rows if row["stability_group"] == "always_wrong"
    ]
    confusion_counts = Counter(
        f"{row['true_class']}->{row['consensus_prediction']}"
        for row in always_wrong_rows
    )
    ordered_wrong = sorted(
        always_wrong_rows,
        key=lambda row: (
            str(row["chrom"]),
            int(row["window_start"]),
            int(row["window_end"]),
        ),
    )
    overlap_components: list[dict[str, object]] = []
    for row in ordered_wrong:
        if (
            not overlap_components
            or overlap_components[-1]["chrom"] != row["chrom"]
            or int(row["window_start"]) >= int(overlap_components[-1]["end"])
        ):
            overlap_components.append(
                {
                    "chrom": row["chrom"],
                    "start": int(row["window_start"]),
                    "end": int(row["window_end"]),
                    "structure_ids": [row["structure_id"]],
                }
            )
        else:
            overlap_components[-1]["end"] = max(
                int(overlap_components[-1]["end"]), int(row["window_end"])
            )
            overlap_components[-1]["structure_ids"].append(row["structure_id"])
    summary: dict[str, object] = {
        "dataset": str(Path(dataset_path).expanduser()),
        "checkpoints": [str(Path(path).expanduser()) for path in checkpoint_paths],
        "seeds": seeds,
        "num_test_samples": len(rows),
        "always_correct": overall_groups["always_correct"],
        "mixed": overall_groups["mixed"],
        "always_wrong": overall_groups["always_wrong"],
        "per_class": per_class,
        "always_wrong_confusions": dict(sorted(confusion_counts.items())),
        "num_cross_class_annotation_overlaps": len(cross_class_overlaps),
        "cross_class_annotation_overlaps": cross_class_overlaps,
        "always_wrong_samples": [
            row["structure_id"]
            for row in rows
            if row["stability_group"] == "always_wrong"
        ],
        "overlapping_always_wrong_groups": [
            component
            for component in overlap_components
            if len(component["structure_ids"]) > 1
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot_stability_summary(rows, class_names, output / "error_stability.png")
    _plot_chid_cases(
        matrices, rows, structure_ids, output / "chid_test_cases.png"
    )
    return summary
