"""Training utilities for the reproducible Task 1 CNN baseline."""

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

from .metrics import classification_metrics
from .model import MicroCCNN


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return device


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            total_loss += float(criterion(logits, labels).item()) * len(labels)
            targets.append(labels.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        np.concatenate(targets),
        np.concatenate(predictions),
    )


def _plot_history(history: list[dict[str, float]], output: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Weighted cross-entropy", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], label="Accuracy")
    axes[1].plot(epochs, [row["validation_macro_f1"] for row in history], label="Macro F1")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation metrics", ylim=(0, 1))
    axes[1].legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_confusion(
    matrix: list[list[int]],
    class_names: list[str],
    output: Path,
) -> None:
    values = np.asarray(matrix, dtype=int)
    figure, axis = plt.subplots(figsize=(5.4, 4.8), constrained_layout=True)
    image = axis.imshow(values, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set(xlabel="Predicted class", ylabel="True class", title="Test confusion matrix")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, str(values[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def train_baseline(
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
    """Train once on train, select on validation, and evaluate test once."""

    if epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("epochs, batch_size and patience must be positive")
    _set_seed(seed)
    device = _resolve_device(device_name)
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    with np.load(Path(dataset_path).expanduser()) as dataset:
        matrices = dataset["matrices"].astype("float32")
        labels_text = dataset["labels"].astype(str)
        splits = dataset["split"].astype(str)
        structure_ids = dataset["structure_id"].astype(str)

    class_names = sorted(np.unique(labels_text).tolist())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    labels = np.asarray([class_to_index[value] for value in labels_text], dtype="int64")
    split_masks = {
        name: splits == name for name in ("train", "validation", "test")
    }
    for name, mask in split_masks.items():
        if not mask.any():
            raise ValueError(f"dataset contains no {name} samples")

    train_values = matrices[split_masks["train"]]
    channel_mean = train_values.mean(axis=(0, 2, 3), keepdims=True)
    channel_std = train_values.std(axis=(0, 2, 3), keepdims=True)
    channel_std = np.where(channel_std < 1e-6, 1.0, channel_std)
    matrices = (matrices - channel_mean) / channel_std

    tensors: dict[str, TensorDataset] = {}
    for name, mask in split_masks.items():
        tensors[name] = TensorDataset(
            torch.from_numpy(matrices[mask]),
            torch.from_numpy(labels[mask]),
        )
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            tensors["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        ),
        "validation": DataLoader(tensors["validation"], batch_size=batch_size),
        "test": DataLoader(tensors["test"], batch_size=batch_size),
    }

    train_counts = np.bincount(labels[split_masks["train"]], minlength=len(class_names))
    class_weights = len(tensors["train"]) / (len(class_names) * train_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
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
        for inputs, target in loaders["train"]:
            inputs = inputs.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(target)

        validation_loss, validation_targets, validation_predictions = _evaluate(
            model, loaders["validation"], criterion, device
        )
        validation_metrics = classification_metrics(
            validation_targets, validation_predictions, class_names
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / len(tensors["train"]),
                "validation_loss": validation_loss,
                "validation_accuracy": float(validation_metrics["accuracy"]),
                "validation_macro_f1": float(validation_metrics["macro_f1"]),
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
    validation_loss, validation_targets, validation_predictions = _evaluate(
        model, loaders["validation"], criterion, device
    )
    test_loss, test_targets, test_predictions = _evaluate(
        model, loaders["test"], criterion, device
    )
    validation_metrics = classification_metrics(
        validation_targets, validation_predictions, class_names
    )
    test_metrics = classification_metrics(test_targets, test_predictions, class_names)
    validation_metrics["loss"] = validation_loss
    test_metrics["loss"] = test_loss

    majority_index = int(np.argmax(train_counts))
    majority_baseline = {
        "predicted_class": class_names[majority_index],
        "validation": classification_metrics(
            validation_targets,
            np.full_like(validation_targets, majority_index),
            class_names,
        ),
        "test": classification_metrics(
            test_targets,
            np.full_like(test_targets, majority_index),
            class_names,
        ),
    }

    checkpoint = {
        "model_state": best_state,
        "class_names": class_names,
        "channel_mean": channel_mean.reshape(-1).tolist(),
        "channel_std": channel_std.reshape(-1).tolist(),
        "input_shape": list(matrices.shape[1:]),
        "replicate_mode": "channels",
    }
    torch.save(checkpoint, output / "model.pt")

    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    results: dict[str, object] = {
        "status": "baseline_completed",
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "device": str(device),
        "seed": seed,
        "replicate_mode": "two_replicates_as_channels",
        "class_names": class_names,
        "class_weights": {
            name: float(class_weights[index]) for index, name in enumerate(class_names)
        },
        "sample_counts": {
            name: int(mask.sum()) for name, mask in split_masks.items()
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "majority_baseline": majority_baseline,
        "test_structure_ids": structure_ids[split_masks["test"]].tolist(),
        "test_targets": [class_names[index] for index in test_targets.tolist()],
        "test_predictions": [class_names[index] for index in test_predictions.tolist()],
    }
    (output / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot_history(history, output / "training_curves.png")
    _plot_confusion(
        test_metrics["confusion_matrix"], class_names, output / "confusion_matrix.png"
    )
    return results
