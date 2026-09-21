"""Input-gradient saliency for inspecting the first Task 1 baseline."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .model import MicroCCNN


def normalize_saliency(values: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Scale a non-negative saliency map to zero through one robustly."""

    saliency = np.asarray(values, dtype=float)
    if saliency.ndim != 2:
        raise ValueError("saliency map must be two-dimensional")
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    saliency = np.nan_to_num(saliency, nan=0.0, posinf=0.0, neginf=0.0)
    saliency = np.maximum(saliency, 0.0)
    scale = float(np.percentile(saliency, percentile))
    if scale <= 0:
        return np.zeros_like(saliency)
    return np.clip(saliency / scale, 0.0, 1.0)


def generate_test_saliency(
    dataset_path: str,
    checkpoint_path: str,
    output_dir: str,
    *,
    per_class: int = 1,
) -> list[dict[str, object]]:
    """Generate predicted-class saliency maps for representative test samples."""

    if per_class <= 0:
        raise ValueError("per_class must be positive")
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    with np.load(Path(dataset_path).expanduser()) as dataset:
        matrices = dataset["matrices"].astype("float32")
        labels = dataset["labels"].astype(str)
        splits = dataset["split"].astype(str)
        structure_ids = dataset["structure_id"].astype(str)
        chrom = dataset["chrom"].astype(str)
        window_start = dataset["window_start"].astype("int64")
        window_end = dataset["window_end"].astype("int64")

    checkpoint = torch.load(
        Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=True
    )
    class_names = [str(name) for name in checkpoint["class_names"]]
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
        predictions = model(torch.from_numpy(standardized)).argmax(dim=1).numpy()
    predicted_names = np.asarray([class_names[index] for index in predictions])

    selected_indices: list[int] = []
    for class_name in class_names:
        candidates = np.flatnonzero((splits == "test") & (labels == class_name))
        correct = candidates[predicted_names[candidates] == class_name]
        ordered = np.concatenate((correct, candidates[predicted_names[candidates] != class_name]))
        selected_indices.extend(ordered[:per_class].tolist())

    manifest: list[dict[str, object]] = []
    for index in selected_indices:
        inputs = torch.from_numpy(standardized[index : index + 1]).requires_grad_(True)
        logits = model(inputs)
        predicted_index = int(logits.argmax(dim=1).item())
        model.zero_grad(set_to_none=True)
        logits[0, predicted_index].backward()
        attribution = (inputs.grad * inputs).abs().mean(dim=1)[0].detach().numpy()
        saliency = normalize_saliency(attribution)
        contact_map = matrices[index].mean(axis=0)
        start_mb = float(window_start[index]) / 1e6
        end_mb = float(window_end[index]) / 1e6
        extent = (start_mb, end_mb, start_mb, end_mb)

        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
        contact_image = axes[0].imshow(
            contact_map,
            origin="lower",
            cmap="Reds",
            extent=extent,
            vmax=float(np.percentile(contact_map, 99)),
        )
        axes[0].set_title("Mean log1p contact map")
        figure.colorbar(contact_image, ax=axes[0], shrink=0.8)
        saliency_image = axes[1].imshow(
            saliency,
            origin="lower",
            cmap="magma",
            extent=extent,
            vmin=0,
            vmax=1,
        )
        axes[1].set_title("Absolute input × gradient")
        figure.colorbar(saliency_image, ax=axes[1], shrink=0.8)
        for axis in axes:
            axis.set_xlabel(f"{chrom[index]} position (Mb)")
            axis.set_ylabel(f"{chrom[index]} position (Mb)")
        predicted_name = class_names[predicted_index]
        figure.suptitle(
            f"{structure_ids[index]} | true={labels[index]} | predicted={predicted_name}"
        )
        image_path = output / f"saliency_{structure_ids[index]}.png"
        figure.savefig(image_path, dpi=180)
        plt.close(figure)
        manifest.append(
            {
                "structure_id": str(structure_ids[index]),
                "true_class": str(labels[index]),
                "predicted_class": predicted_name,
                "correct": bool(predicted_name == labels[index]),
                "window": f"{chrom[index]}:{window_start[index]}-{window_end[index]}",
                "image": image_path.name,
            }
        )

    (output / "saliency_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
