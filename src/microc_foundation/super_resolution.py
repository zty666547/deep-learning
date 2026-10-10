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
        result = upsampled + residual
        return (result + result.transpose(-1, -2)) / 2


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


def local_ssim(target: np.ndarray, prediction: np.ndarray, window_size: int = 7) -> float:
    """Average uniform-window SSIM over valid windows of a single 2-D map.

    Population moments, K1=0.01 and K2=0.03 are used. This is distinct
    from the original implementation's whole-image moment approximation.
    """
    truth = np.asarray(target, dtype=float)
    estimate = np.asarray(prediction, dtype=float)
    if truth.shape != estimate.shape or truth.ndim != 2:
        raise ValueError("SSIM requires equal two-dimensional maps")
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ValueError("SSIM requires finite maps")
    if window_size < 3 or window_size % 2 != 1 or min(truth.shape) < window_size:
        raise ValueError("SSIM window must be odd, at least 3, and fit the map")
    x = torch.tensor(truth, dtype=torch.float64)[None, None]
    y = torch.tensor(estimate, dtype=torch.float64)[None, None]
    mean_x = F.avg_pool2d(x, window_size, stride=1)
    mean_y = F.avg_pool2d(y, window_size, stride=1)
    var_x = (F.avg_pool2d(x * x, window_size, stride=1) - mean_x * mean_x).clamp_min(0)
    var_y = (F.avg_pool2d(y * y, window_size, stride=1) - mean_y * mean_y).clamp_min(0)
    covariance = F.avg_pool2d(x * y, window_size, stride=1) - mean_x * mean_y
    data_range = max(float(truth.max() - truth.min()), 1e-8)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    value = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2)
    )
    return float(value.mean())


def correct_count_scale(log_values: np.ndarray, spatial_factor: int) -> np.ndarray:
    """Convert log1p(sum counts) to log1p(mean fine-bin counts).

    A factor-two coarse pixel covers four fine pixels. Dividing after
    exponentiation preserves the physical count scale before interpolation.
    """
    values = np.asarray(log_values, dtype=np.float64)
    if spatial_factor <= 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("positive factor and finite non-negative log counts required")
    return np.log1p(np.expm1(values) / spatial_factor**2).astype("float32")


def load_aligned_super_resolution_data(low_path: str, high_path: str) -> dict[str, object]:
    """Validate metadata, paired coordinates, split agreement and spatial leakage."""
    with np.load(Path(low_path).expanduser(), allow_pickle=False) as low_data, np.load(
        Path(high_path).expanduser(), allow_pickle=False
    ) as high_data:
        for dataset in (low_data, high_data):
            if str(dataset["normalization"]) != "log1p" or str(dataset["pooling"]) != "sum":
                raise ValueError("requires log1p sum-pooled source datasets")
            if len(np.unique(dataset["structure_id"])) != len(dataset["structure_id"]):
                raise ValueError("structure IDs must be unique")
        low_ids = low_data["structure_id"].astype(str)
        high_lookup = {str(value): index for index, value in enumerate(high_data["structure_id"])}
        low_indices = np.array([index for index, value in enumerate(low_ids) if value in high_lookup])
        if len(low_indices) == 0:
            raise ValueError("no common samples")
        ids = low_ids[low_indices]
        high_indices = np.array([high_lookup[value] for value in ids])
        for key in ("chrom", "window_start", "window_end"):
            if not np.array_equal(low_data[key][low_indices], high_data[key][high_indices]):
                raise ValueError(f"paired coordinates differ: {key}")
        if not np.array_equal(low_data["replicate"], high_data["replicate"]):
            raise ValueError("replicate channel order differs")
        low = low_data["matrices"][low_indices].astype("float32")
        high = high_data["matrices"][high_indices].astype("float32")
        if low.ndim != 4 or low.shape[2] != low.shape[3] or high.shape != (len(low), low.shape[1], low.shape[2] * 2, low.shape[3] * 2):
            raise ValueError("requires equal channels and exactly factor-two paired square maps")
        if int(low_data["pooled_bin_size"]) != 2 * int(high_data["pooled_bin_size"]):
            raise ValueError("paired resolutions must differ by a factor of two")
        low_splits = low_data["split"][low_indices].astype(str)
        high_splits = high_data["split"][high_indices].astype(str)
        valid = np.isin(low_splits, ("train", "validation", "test")) & np.isin(
            high_splits, ("train", "validation", "test")
        )
        if (low_splits[valid] != high_splits[valid]).any():
            raise ValueError("paired samples have inconsistent retained splits")
        splits = np.where(valid, low_splits, "excluded")
        chroms = low_data["chrom"][low_indices].astype(str)
        starts = low_data["window_start"][low_indices]
        ends = low_data["window_end"][low_indices]
        for index in np.flatnonzero(valid):
            overlap = valid & (chroms == chroms[index]) & (splits != splits[index])
            overlap &= (starts < ends[index]) & (ends > starts[index])
            if overlap.any():
                raise ValueError("genomic windows overlap across data splits")
        if not np.isfinite(high).all() or (high < 0).any():
            raise ValueError("targets must be finite non-negative log counts")
        return {
            "low": correct_count_scale(low, 2), "high": high, "split": splits,
            "ids": ids, "chrom": chroms, "start": starts, "end": ends,
            "labels": low_data["labels"][low_indices].astype(str)
            if "labels" in low_data.files
            else np.full(len(ids), "unknown"),
            "annotation_start": low_data["annotation_start"][low_indices].astype(np.int64)
            if "annotation_start" in low_data.files
            else np.full(len(ids), -1, dtype=np.int64),
            "annotation_end": low_data["annotation_end"][low_indices].astype(np.int64)
            if "annotation_end" in low_data.files
            else np.full(len(ids), -1, dtype=np.int64),
            "replicates": low_data["replicate"].astype(str),
        }


def _batch_metrics(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    values = [
        (psnr(truth, estimate), local_ssim(truth, estimate))
        for sample, prediction_sample in zip(target, prediction)
        for truth, estimate in zip(sample, prediction_sample)
    ]
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
    vmax = float(np.percentile(target[0, 0], 99.5))
    for axis, image, title in zip(axes, images, titles):
        plot = axis.imshow(image[0, 0], cmap="Reds", origin="lower", interpolation="nearest", vmin=0, vmax=vmax)
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
    aligned = load_aligned_super_resolution_data(low_dataset_path, high_dataset_path)
    low, high, splits = aligned["low"], aligned["high"], aligned["split"]

    split_masks = {name: splits == name for name in ("train", "validation", "test")}
    for name, mask in split_masks.items():
        if not mask.any():
            raise ValueError(f"no {name} samples remain after alignment")

    high_train = high[split_masks["train"]]
    high_mean = high_train.mean(axis=(0, 2, 3), keepdims=True)
    high_std = high_train.std(axis=(0, 2, 3), keepdims=True)
    high_std = np.where(high_std < 1e-6, 1.0, high_std)
    low_mean, low_std = high_mean, high_std
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
                inputs, targets = inputs.to(device), targets.to(device)
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
    prediction_original = np.maximum(prediction * high_std + high_mean, 0)
    baseline_original = np.maximum(baseline_original, 0)
    target_original = high[split_masks["test"]]
    baseline_scores = _batch_metrics(target_original, baseline_original)
    model_scores = _batch_metrics(target_original, prediction_original)
    test_indices = np.flatnonzero(split_masks["test"])
    np.savez_compressed(
        output / "test_reconstruction.npz",
        structure_id=aligned["ids"][test_indices],
        label=aligned["labels"][test_indices],
        chrom=aligned["chrom"][test_indices],
        window_start=aligned["start"][test_indices],
        window_end=aligned["end"][test_indices],
        annotation_start=aligned["annotation_start"][test_indices],
        annotation_end=aligned["annotation_end"][test_indices],
        replicate=aligned["replicates"],
        target=target_original.astype("float32"),
        bicubic=baseline_original.astype("float32"),
        cnn=prediction_original.astype("float32"),
    )

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
        low[split_masks["test"]][0:1], baseline_original[0:1], prediction_original[0:1], target_original[0:1], output / "example_comparison.png"
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
        "protocol": "count-corrected factor-two; shared train-only scale; symmetric CNN",
        "ssim_definition": "per-replicate 7x7 valid uniform-window SSIM, population moments",
        "intensity_correction": "log1p(expm1(low_log_counts)/4)",
        "bicubic_baseline": {"psnr": baseline_scores[0], "ssim": baseline_scores[1], "mse": baseline_scores[2]},
        "cnn": {"psnr": model_scores[0], "ssim": model_scores[1], "mse": model_scores[2]},
    }
    test_ids = aligned["ids"][split_masks["test"]]
    with (output / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["structure_id", "replicate_index", "method", "psnr", "ssim", "mse"])
        writer.writeheader()
        for index, structure_id in enumerate(test_ids):
            for channel in range(target_original.shape[1]):
                truth = target_original[index, channel]
                for method, estimate in (("bicubic", baseline_original), ("cnn", prediction_original)):
                    predicted = estimate[index, channel]
                    writer.writerow({"structure_id": structure_id, "replicate_index": channel + 1,
                                     "method": method, "psnr": psnr(truth, predicted),
                                     "ssim": local_ssim(truth, predicted), "mse": float(np.mean((truth - predicted)**2))})
    (output / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results
