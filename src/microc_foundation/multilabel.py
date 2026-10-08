"""Region-level multi-label training for overlapping Task 1 annotations."""

from __future__ import annotations

import copy
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import multilabel_metrics
from .model import MicroCCNN


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return device


def select_label_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    candidates: np.ndarray | None = None,
) -> np.ndarray:
    """Select each label threshold on validation data by maximum binary F1."""

    true = np.asarray(targets, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if true.shape != scores.shape or true.ndim != 2:
        raise ValueError("targets and probabilities must be equal two-dimensional arrays")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    grid = (
        np.asarray(candidates, dtype=float)
        if candidates is not None
        else np.arange(0.05, 1.0, 0.05)
    )
    if grid.ndim != 1 or not len(grid) or ((grid <= 0) | (grid >= 1)).any():
        raise ValueError("threshold candidates must lie strictly between zero and one")

    thresholds: list[float] = []
    for label_index in range(true.shape[1]):
        target = true[:, label_index]
        best_key: tuple[float, float, float] | None = None
        best_threshold = 0.5
        for threshold in grid:
            prediction = scores[:, label_index] >= threshold
            true_positive = int(((target == 1) & prediction).sum())
            false_positive = int(((target == 0) & prediction).sum())
            false_negative = int(((target == 1) & ~prediction).sum())
            denominator = 2 * true_positive + false_positive + false_negative
            f1 = 2 * true_positive / denominator if denominator else 0.0
            key = (f1, -abs(float(threshold) - 0.5), float(threshold))
            if best_key is None or key > best_key:
                best_key = key
                best_threshold = float(threshold)
        thresholds.append(best_threshold)
    return np.asarray(thresholds, dtype="float32")


def _evaluate_probabilities(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            total_loss += float(criterion(logits, labels).item()) * len(labels)
            targets.append(labels.cpu().numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        np.concatenate(targets),
        np.concatenate(probabilities),
    )


def _plot_history(history: list[dict[str, float]], output: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="BCE objective", title="Loss")
    axes[0].legend()
    axes[1].plot(
        epochs,
        [row["validation_macro_f1"] for row in history],
        label="Macro F1",
    )
    axes[1].plot(
        epochs,
        [row["validation_exact_match"] for row in history],
        label="Exact match",
    )
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation", ylim=(0, 1))
    axes[1].legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_label_metrics(metrics: dict[str, object], output: Path) -> None:
    per_label = metrics["per_label"]
    class_names = list(per_label)
    values = np.asarray(
        [
            [per_label[name][metric] for metric in ("precision", "recall", "f1")]
            for name in class_names
        ]
    )
    figure, axis = plt.subplots(figsize=(6.5, 4.3), constrained_layout=True)
    image = axis.imshow(values, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(3), ["Precision", "Recall", "F1"])
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_title("Region-level multi-label test metrics")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def train_multilabel(
    dataset_path: str,
    output_dir: str,
    *,
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    seed: int = 2026,
    device_name: str = "cpu",
) -> dict[str, object]:
    """Train a region-level classifier and tune thresholds on validation only."""

    if epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("epochs, batch_size and patience must be positive")
    _set_seed(seed)
    device = _resolve_device(device_name)
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    with np.load(Path(dataset_path).expanduser()) as dataset:
        matrices = dataset["matrices"].astype("float32")
        targets = dataset["targets"].astype("float32")
        splits = dataset["split"].astype(str)
        region_ids = dataset["region_id"].astype(str)
        class_names = dataset["class_names"].astype(str).tolist()
    if targets.shape != (len(matrices), len(class_names)):
        raise ValueError("target shape does not match samples and class names")

    split_masks = {name: splits == name for name in ("train", "validation", "test")}
    for name, mask in split_masks.items():
        if not mask.any():
            raise ValueError(f"dataset contains no {name} samples")
    train_targets = targets[split_masks["train"]]
    positive_counts = train_targets.sum(axis=0)
    if (positive_counts == 0).any():
        raise ValueError("every label must have a positive training sample")

    train_values = matrices[split_masks["train"]]
    channel_mean = train_values.mean(axis=(0, 2, 3), keepdims=True)
    channel_std = train_values.std(axis=(0, 2, 3), keepdims=True)
    channel_std = np.where(channel_std < 1e-6, 1.0, channel_std)
    matrices = (matrices - channel_mean) / channel_std

    tensors = {
        name: TensorDataset(
            torch.from_numpy(matrices[mask]), torch.from_numpy(targets[mask])
        )
        for name, mask in split_masks.items()
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            tensors["train"], batch_size=batch_size, shuffle=True, generator=generator
        ),
        "validation": DataLoader(tensors["validation"], batch_size=batch_size),
        "test": DataLoader(tensors["test"], batch_size=batch_size),
    }

    negative_counts = len(train_targets) - positive_counts
    pos_weight = torch.tensor(
        negative_counts / positive_counts, dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = MicroCCNN(matrices.shape[1], len(class_names)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, labels in loaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)

        validation_loss, validation_targets, validation_probabilities = (
            _evaluate_probabilities(model, loaders["validation"], criterion, device)
        )
        thresholds = select_label_thresholds(
            validation_targets, validation_probabilities
        )
        validation_predictions = validation_probabilities >= thresholds
        validation_metrics = multilabel_metrics(
            validation_targets, validation_predictions, class_names
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / len(tensors["train"]),
                "validation_loss": validation_loss,
                "validation_macro_f1": float(validation_metrics["macro_f1"]),
                "validation_exact_match": float(
                    validation_metrics["exact_match_ratio"]
                ),
            }
        )
        current_macro_f1 = float(validation_metrics["macro_f1"])
        if current_macro_f1 > best_macro_f1 + 1e-8:
            best_macro_f1 = current_macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    validation_loss, validation_targets, validation_probabilities = (
        _evaluate_probabilities(model, loaders["validation"], criterion, device)
    )
    thresholds = select_label_thresholds(validation_targets, validation_probabilities)
    test_loss, test_targets, test_probabilities = _evaluate_probabilities(
        model, loaders["test"], criterion, device
    )
    validation_predictions = validation_probabilities >= thresholds
    test_predictions = test_probabilities >= thresholds
    validation_metrics = multilabel_metrics(
        validation_targets, validation_predictions, class_names
    )
    test_metrics = multilabel_metrics(test_targets, test_predictions, class_names)
    validation_metrics["loss"] = validation_loss
    test_metrics["loss"] = test_loss

    prevalence_prediction = (train_targets.mean(axis=0) >= 0.5).astype(int)
    prevalence_baseline = {
        "rule": "predict labels with training prevalence at least 0.5",
        "predicted_labels": [
            name
            for index, name in enumerate(class_names)
            if prevalence_prediction[index]
        ],
        "validation": multilabel_metrics(
            validation_targets,
            np.tile(prevalence_prediction, (len(validation_targets), 1)),
            class_names,
        ),
        "test": multilabel_metrics(
            test_targets,
            np.tile(prevalence_prediction, (len(test_targets), 1)),
            class_names,
        ),
    }

    checkpoint = {
        "model_state": best_state,
        "class_names": class_names,
        "thresholds": thresholds.tolist(),
        "channel_mean": channel_mean.reshape(-1).tolist(),
        "channel_std": channel_std.reshape(-1).tolist(),
        "input_shape": list(matrices.shape[1:]),
        "seed": seed,
        "task": "region_multilabel",
    }
    torch.save(checkpoint, output / "model.pt")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    test_ids = region_ids[split_masks["test"]]
    with (output / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = ["region_id"]
        for name in class_names:
            fieldnames.extend((f"true_{name}", f"probability_{name}", f"predicted_{name}"))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, region_id in enumerate(test_ids):
            row: dict[str, object] = {"region_id": region_id}
            for class_index, name in enumerate(class_names):
                row[f"true_{name}"] = int(test_targets[row_index, class_index])
                row[f"probability_{name}"] = float(
                    test_probabilities[row_index, class_index]
                )
                row[f"predicted_{name}"] = int(
                    test_predictions[row_index, class_index]
                )
            writer.writerow(row)

    results: dict[str, object] = {
        "status": "region_multilabel_completed",
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "device": str(device),
        "seed": seed,
        "class_names": class_names,
        "positive_class_weights": {
            name: float(pos_weight[index].cpu())
            for index, name in enumerate(class_names)
        },
        "thresholds": {
            name: float(thresholds[index]) for index, name in enumerate(class_names)
        },
        "sample_counts": {
            name: int(mask.sum()) for name, mask in split_masks.items()
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "prevalence_baseline": prevalence_baseline,
    }
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot_history(history, output / "training_curves.png")
    _plot_label_metrics(test_metrics, output / "label_metrics.png")
    return results


def summarize_multilabel_runs(results: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate repeated multi-label runs while retaining seed-level values."""

    if not results:
        raise ValueError("at least one seed result is required")
    class_names = list(results[0]["class_names"])
    rows: list[dict[str, float | int]] = []
    for result in results:
        if list(result["class_names"]) != class_names:
            raise ValueError("all runs must use the same class order")
        test = result["test"]
        row: dict[str, float | int] = {
            "seed": int(result["seed"]),
            "best_epoch": int(result["best_epoch"]),
            "epochs_ran": int(result["epochs_ran"]),
            "test_macro_f1": float(test["macro_f1"]),
            "test_micro_f1": float(test["micro_f1"]),
            "test_exact_match_ratio": float(test["exact_match_ratio"]),
            "test_hamming_loss": float(test["hamming_loss"]),
        }
        for name in class_names:
            row[f"test_f1_{name}"] = float(test["per_label"][name]["f1"])
            row[f"test_recall_{name}"] = float(test["per_label"][name]["recall"])
            row[f"threshold_{name}"] = float(result["thresholds"][name])
        rows.append(row)

    aggregate: dict[str, dict[str, float]] = {}
    for key in rows[0]:
        if key in {"seed", "best_epoch", "epochs_ran"}:
            continue
        values = np.asarray([float(row[key]) for row in rows])
        aggregate[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "num_runs": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
        "class_names": class_names,
        "aggregate": aggregate,
        "runs": rows,
    }


def write_multilabel_summary(summary: dict[str, object], output_dir: str) -> Path:
    """Write repeated-run tables and a compact stability figure."""

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = summary["runs"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    seeds = [int(row["seed"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for key, label in (
        ("test_macro_f1", "Macro F1"),
        ("test_micro_f1", "Micro F1"),
        ("test_exact_match_ratio", "Exact match"),
    ):
        axis.plot(seeds, [float(row[key]) for row in rows], marker="o", label=label)
    axis.set(
        xlabel="Random seed",
        ylabel="Test score",
        title="Region-level multi-label stability",
        ylim=(0, 1),
    )
    axis.set_xticks(seeds)
    axis.legend()
    figure.savefig(output / "seed_stability.png", dpi=180)
    plt.close(figure)
    return output
