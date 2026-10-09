"""Lightweight contact-map super-resolution baseline for optional Task 5."""

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
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class ContactSuperResolutionCNN(nn.Module):
    """Residual CNN that upsamples a low-resolution contact map by two."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        upsampled = F.interpolate(
            inputs, scale_factor=2, mode="bilinear", align_corners=False
        )
        residual = F.interpolate(
            self.encoder(inputs), scale_factor=2, mode="bilinear", align_corners=False
        )
        return upsampled + residual


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


def psnr(target: np.ndarray, prediction: np.ndarray) -> float:
    """Compute PSNR using each target map's observed dynamic range."""

    truth = np.asarray(target, dtype=float)
    estimate = np.asarray(prediction, dtype=float)
    if truth.shape != estimate.shape:
        raise ValueError("target and prediction must have the same shape")
    mse = float(np.mean((truth - estimate) ** 2))
    if mse == 0:
        return float("inf")
    data_range = float(truth.max() - truth.min())
    if data_range <= 0:
        data_range = 1.0
    return float(20 * np.log10(data_range / np.sqrt(mse)))


def global_ssim(target: np.ndarray, prediction: np.ndarray) -> float:
    """Compute a dependency-light global SSIM for one contact-map pair."""

    truth = np.asarray(target, dtype=float)
    estimate = np.asarray(prediction, dtype=float)
    if truth.shape != estimate.shape:
        raise ValueError("target and prediction must have the same shape")
    data_range = float(truth.max() - truth.min())
    if data_range <= 0:
        data_range = 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mean_truth = float(truth.mean())
    mean_estimate = float(estimate.mean())
    variance_truth = float(truth.var())
    variance_estimate = float(estimate.var())
    covariance = float(np.mean((truth - mean_truth) * (estimate - mean_estimate)))
    numerator = (2 * mean_truth * mean_estimate + c1) * (2 * covariance + c2)
    denominator = (
        (mean_truth**2 + mean_estimate**2 + c1)
        * (variance_truth + variance_estimate + c2)
    )
    return float(numerator / denominator) if denominator else 0.0


def _batch_metrics(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    values = [(psnr(truth, estimate), global_ssim(truth, estimate)) for truth, estimate in zip(target, prediction)]
    return (
        float(np.mean([value[0] for value in values])),
        float(np.mean([value[1] for value in values])),
        float(np.mean((target - prediction) ** 2)),
    )


def _plot_example(
    low: np.ndarray, baseline: np.ndarray, prediction: np.ndarray, target: np.ndarray, output: Path
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.5), constrained_layout=True)
    images = (low, baseline, prediction, target)
    titles = ("Low-resolution", "Bicubic baseline", "CNN output", "Target")
    for axis, image, title in zip(axes, images, titles):
        plot = axis.imshow(image[0, 0], cmap="Reds", origin="lower", interpolation="nearest")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(plot, ax=axis, shrink=0.75)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def train_super_resolution(
    low_dataset_path: str,
    high_dataset_path: str,
    output_dir: str,
    *,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    seed: int = 2026,
    device_name: str = "cpu",
) -> dict[str, object]:
    """Train on train split and compare CNN with bicubic upsampling on test."""

    if epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("epochs, batch_size and patience must be positive")
    _set_seed(seed)
    device = _resolve_device(device_name)
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(Path(low_dataset_path).expanduser()) as low_data, np.load(
        Path(high_dataset_path).expanduser()
    ) as high_data:
        low_ids = low_data["structure_id"].astype(str)
        high_ids = high_data["structure_id"].astype(str)
        low_index = {value: index for index, value in enumerate(low_ids)}
        high_index = {value: index for index, value in enumerate(high_ids)}
        common_ids = [value for value in low_ids if value in high_index]
        low = low_data["matrices"][[low_index[value] for value in common_ids]].astype("float32")
        high = high_data["matrices"][[high_index[value] for value in common_ids]].astype("float32")
        splits = low_data["split"][[low_index[value] for value in common_ids]].astype(str)
        splits = np.where(np.isin(splits, ("train", "validation", "test")), splits, "excluded")

    split_masks = {name: splits == name for name in ("train", "validation", "test")}
    for name, mask in split_masks.items():
        if not mask.any():
            raise ValueError(f"no {name} samples remain after alignment")

    high_train = high[split_masks["train"]]
    high_mean = high_train.mean(axis=(0, 2, 3), keepdims=True)
    high_std = high_train.std(axis=(0, 2, 3), keepdims=True)
    high_std = np.where(high_std < 1e-6, 1.0, high_std)
    low_train = low[split_masks["train"]]
    low_mean = low_train.mean(axis=(0, 2, 3), keepdims=True)
    low_std = low_train.std(axis=(0, 2, 3), keepdims=True)
    low_std = np.where(low_std < 1e-6, 1.0, low_std)
    low_scaled = (low - low_mean) / low_std
    high_scaled = (high - high_mean) / high_std

    tensors = {
        name: TensorDataset(
            torch.from_numpy(low_scaled[mask]), torch.from_numpy(high_scaled[mask])
        )
        for name, mask in split_masks.items()
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(tensors["train"], batch_size=batch_size, shuffle=True, generator=generator),
        "validation": DataLoader(tensors["validation"], batch_size=batch_size),
        "test": DataLoader(tensors["test"], batch_size=batch_size),
    }
    model = ContactSuperResolutionCNN(low.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    best_epoch = 0
    without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets in loaders["train"]:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(inputs)
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, targets in loaders["validation"]:
                validation_loss += float(F.mse_loss(model(inputs), targets).item()) * len(inputs)
        validation_loss /= len(tensors["validation"])
        history.append({"epoch": float(epoch), "train_loss": train_loss / len(tensors["train"]), "validation_loss": validation_loss})
        if validation_loss < best_validation - 1e-8:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            without_improvement = 0
        else:
            without_improvement += 1
            if without_improvement >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    low_test = low_scaled[split_masks["test"]]
    with torch.no_grad():
        prediction = model(torch.from_numpy(low_test).to(device)).cpu().numpy()
    low_tensor = torch.from_numpy(low_test).to(device)
    with torch.no_grad():
        baseline = F.interpolate(low_tensor, scale_factor=2, mode="bicubic", align_corners=False).cpu().numpy()
    baseline_original = baseline * low_std + low_mean
    prediction_original = prediction * high_std + high_mean
    target_original = high[split_masks["test"]]
    baseline_scores = _batch_metrics(target_original, baseline_original)
    model_scores = _batch_metrics(target_original, prediction_original)

    torch.save(
        {
            "model_state": best_state,
            "seed": seed,
            "low_mean": low_mean.reshape(-1).tolist(),
            "low_std": low_std.reshape(-1).tolist(),
            "high_mean": high_mean.reshape(-1).tolist(),
            "high_std": high_std.reshape(-1).tolist(),
            "task": "contact_super_resolution",
        },
        output / "model.pt",
    )
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    _plot_example(
        low_test[0:1], baseline_original[0:1], prediction_original[0:1], target_original[0:1], output / "example_comparison.png"
    )
    results = {
        "status": "super_resolution_completed",
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "device": str(device),
        "sample_counts": {name: int(mask.sum()) for name, mask in split_masks.items()},
        "excluded_after_alignment": int((splits == "excluded").sum()),
        "input_shape": list(low.shape[1:]),
        "target_shape": list(high.shape[1:]),
        "bicubic_baseline": {"psnr": baseline_scores[0], "ssim": baseline_scores[1], "mse": baseline_scores[2]},
        "cnn": {"psnr": model_scores[0], "ssim": model_scores[1], "mse": model_scores[2]},
    }
    (output / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results
