"""All-test-sample attribution stability and parameter-reset sanity controls."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from microc_foundation.model import MicroCCNN


def absolute_input_gradient(
    model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """True/fixed-target logit attribution, channel average and symmetric pairs."""
    if (inputs.ndim != 4 or inputs.shape[-1] != inputs.shape[-2] or
            targets.shape != (len(inputs),) or not torch.isfinite(inputs).all()):
        raise ValueError("expected finite square batches and one target per sample")
    model.eval()
    values = inputs.detach().clone().requires_grad_(True)
    logits = model(values)
    if (targets.dtype != torch.long or (targets < 0).any() or
            (targets >= logits.shape[1]).any()):
        raise ValueError("targets must be valid integer class indices")
    selected = logits.gather(1, targets[:, None]).sum()
    gradients = torch.autograd.grad(selected, values)[0]
    maps = (gradients * values).abs().mean(dim=1).detach().numpy()
    maps = (maps + maps.transpose(0, 2, 1)) / 2
    return maps, logits.detach().numpy()


def _validate_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[0] != values.shape[1] or
            not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError("expected finite nonnegative square attribution")
    return values


def attribution_mass(
    values: np.ndarray, *, window_start: int, window_end: int,
    annotation_start: int, annotation_end: int, minimum_offset: int = 5,
) -> dict[str, object]:
    """Unique upper pairs, bin-center annotation membership, raw (unclipped) mass."""
    values = _validate_map(values)
    size = len(values)
    if (not 0 < minimum_offset < size or window_start < 0 or window_end <= window_start or
            not window_start <= annotation_start < annotation_end <= window_end):
        raise ValueError("invalid annotation/window or distance threshold")
    upper = np.triu(np.ones(values.shape, dtype=bool))
    row, col = np.indices(values.shape)
    near = upper & ((col - row) < minimum_offset)
    centers = window_start + (np.arange(size) + 0.5) * (window_end - window_start) / size
    inside = (centers >= annotation_start) & (centers < annotation_end)
    region = upper & inside[:, None] & inside[None, :]
    total = float(values[upper].sum())
    area = float(region.sum() / upper.sum())
    region_mass = float(values[region].sum() / total) if total > 0 else None
    return {
        "attribution_status": "evaluated" if total > 0 else "zero_mass",
        "near_diagonal_area_fraction": float(near.sum() / upper.sum()),
        "near_diagonal_mass_fraction": float(values[near].sum() / total) if total > 0 else None,
        "annotation_area_fraction": area, "annotation_mass_fraction": region_mass,
        "annotation_area_enrichment": region_mass / area if total > 0 and area > 0 else None,
    }


def fractional_top_weights(values: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    """Equal weights within quantile ties, avoiding arbitrary pixel-index ranking."""
    values = np.asarray(values, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all() or
            not 0 < fraction < 1):
        raise ValueError("expected finite vector and fraction in (0, 1)")
    target = fraction * len(values)
    boundary = np.sort(values)[-int(np.ceil(target))]
    weights = (values > boundary).astype(float)
    tied = values == boundary
    weights[tied] = (target - weights.sum()) / tied.sum()
    return weights


def compare_attributions(
    left: np.ndarray, right: np.ndarray, *, minimum_offset: int = 5,
) -> dict[str, object]:
    """Off-diagonal Spearman and fractional top-10% Jaccard; constants abstain."""
    left, right = _validate_map(left), _validate_map(right)
    if left.shape != right.shape or not 0 < minimum_offset < len(left):
        raise ValueError("maps or distance threshold do not match")
    mask = np.triu(np.ones(left.shape, dtype=bool), k=minimum_offset)
    a, b = left[mask], right[mask]
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return {"status": "constant_or_insufficient", "spearman": None, "top10_jaccard": None}
    rank_a = pd.Series(a).rank(method="average").to_numpy()
    rank_b = pd.Series(b).rank(method="average").to_numpy()
    weights_a, weights_b = fractional_top_weights(a), fractional_top_weights(b)
    return {
        "status": "evaluated", "spearman": float(np.corrcoef(rank_a, rank_b)[0, 1]),
        "top10_jaccard": float(np.minimum(weights_a, weights_b).sum() /
                                np.maximum(weights_a, weights_b).sum()),
    }


def audit_checkpoint_saliency(
    dataset_path: str, checkpoint_paths: list[str], output_dir: str,
    *, minimum_offset: int = 5, batch_size: int = 4,
) -> dict[str, object]:
    """Audit three fixed trained models and paired deterministic random networks."""
    if len(checkpoint_paths) != 3 or batch_size <= 0:
        raise ValueError("exactly three checkpoints and positive batch size required")
    if len(set(checkpoint_paths)) != 3:
        raise ValueError("checkpoints must be distinct")
    with np.load(dataset_path, allow_pickle=False) as data:
        test = data["split"].astype(str) == "test"
        matrices = data["matrices"][test].astype(np.float32)
        fields = {name: data[name][test] for name in (
            "structure_id", "labels", "chrom", "window_start", "window_end",
            "annotation_start", "annotation_end",
        )}
        pooled_bin_size = int(data["pooled_bin_size"])
    if not len(matrices) or not np.isfinite(matrices).all():
        raise ValueError("expected finite nonempty test matrices")
    if len(set(fields["structure_id"])) != len(matrices):
        raise ValueError("test structure IDs must be unique")
    if not np.all(fields["window_end"] - fields["window_start"] ==
                  matrices.shape[-1] * pooled_bin_size):
        raise ValueError("window extent does not match pooled bins")
    trained_maps, random_maps, records, class_names = [], [], [], None
    controls = [9101, 9102, 9103]
    for model_index, checkpoint_path in enumerate(checkpoint_paths):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        names = [str(name) for name in checkpoint["class_names"]]
        if class_names is not None and names != class_names:
            raise ValueError("all checkpoints must use the same class order")
        class_names = names
        targets = torch.tensor([names.index(str(label)) for label in fields["labels"]],
                               dtype=torch.long)
        mean = np.asarray(checkpoint["channel_mean"], dtype=np.float32).reshape(1, -1, 1, 1)
        std = np.asarray(checkpoint["channel_std"], dtype=np.float32).reshape(1, -1, 1, 1)
        if (mean.shape[1] != matrices.shape[1] or std.shape[1] != matrices.shape[1] or
                not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
            raise ValueError("invalid checkpoint training scaler")
        standardized = torch.from_numpy((matrices - mean) / std)
        trained = MicroCCNN(matrices.shape[1], len(names))
        trained.load_state_dict(checkpoint["model_state"])
        with torch.random.fork_rng():
            torch.manual_seed(controls[model_index])
            random = MicroCCNN(matrices.shape[1], len(names))
        model_maps, model_logits = [], []
        for model in (trained, random):
            maps, logits = [], []
            for start in range(0, len(matrices), batch_size):
                attribution, scores = absolute_input_gradient(
                    model, standardized[start:start + batch_size], targets[start:start + batch_size]
                )
                maps.append(attribution)
                logits.append(scores)
            model_maps.append(np.concatenate(maps))
            model_logits.append(np.concatenate(logits))
        trained_maps.append(model_maps[0])
        random_maps.append(model_maps[1])
        for index in range(len(matrices)):
            predicted = names[int(model_logits[0][index].argmax())]
            record = {
                "model_index": model_index, "checkpoint": str(checkpoint_path),
                "random_seed": controls[model_index], "structure_id": str(fields["structure_id"][index]),
                "true_class": str(fields["labels"][index]), "predicted_class": predicted,
                "correct": bool(predicted == str(fields["labels"][index])),
                "chrom": str(fields["chrom"][index]),
                **{name: int(fields[name][index]) for name in (
                    "window_start", "window_end", "annotation_start", "annotation_end"
                )},
            }
            kwargs = {name: int(fields[name][index]) for name in (
                "window_start", "window_end", "annotation_start", "annotation_end"
            )}
            for prefix, maps in zip(("trained", "random"), model_maps):
                record.update({f"{prefix}_{key}": value for key, value in attribution_mass(
                    maps[index], **kwargs, minimum_offset=minimum_offset
                ).items()})
            records.append(record)
    trained_maps, random_maps = np.stack(trained_maps), np.stack(random_maps)
    pair_records = []
    pairs = [("trained_vs_random", i, i) for i in range(3)] + [
        ("cross_trained_seed", i, j) for i, j in itertools.combinations(range(3), 2)
    ]
    for kind, left, right in pairs:
        right_maps = random_maps[right] if kind == "trained_vs_random" else trained_maps[right]
        for index in range(len(matrices)):
            pair_records.append({
                "comparison": kind, "left_model": left, "right_model": right,
                "structure_id": str(fields["structure_id"][index]),
                "true_class": str(fields["labels"][index]),
                **compare_attributions(trained_maps[left, index], right_maps[index],
                                       minimum_offset=minimum_offset),
            })
    rows, pair_rows = pd.DataFrame(records), pd.DataFrame(pair_records)
    # The 3 seeds/pairs describe each structure; they are not 3 new samples.
    structure_pairs = pair_rows.groupby(
        ["comparison", "structure_id", "true_class"], observed=True, as_index=False
    )[["spearman", "top10_jaccard"]].median()
    mass_columns = [name for name in rows if name.endswith(("mass_fraction", "area_enrichment"))]
    structure_mass = rows.groupby("structure_id", observed=True)[mass_columns].median()
    group_summary = []
    for label, group in rows.groupby(["true_class", "correct"], observed=True):
        group_summary.append({
            "true_class": str(label[0]), "correct": bool(label[1]), "model_sample_rows": len(group),
            **{name + "_median": float(group[name].median()) if group[name].notna().any() else None
               for name in rows.columns if name.endswith(("mass_fraction", "area_enrichment"))},
        })
    comparison_summary = {
        str(kind): {
            "model_pair_sample_rows": len(group),
            "evaluated_rows": int((group.status == "evaluated").sum()),
            "structure_n": len(structure_pairs.loc[structure_pairs.comparison == kind]),
            **{name + "_structure_median": float(
                structure_pairs.loc[structure_pairs.comparison == kind, name].median()
            ) if structure_pairs.loc[structure_pairs.comparison == kind, name].notna().any() else None
               for name in ("spearman", "top10_jaccard")},
            **{name + "_median": float(group[name].median()) if group[name].notna().any() else None
               for name in ("spearman", "top10_jaccard")},
        } for kind, group in pair_rows.groupby("comparison", observed=True)
    }
    summary = {
        "protocol": "task1-saliency-audit-v1", "test_structures": len(matrices),
        "target": "true-class logit including incorrect predictions",
        "attribution": "abs(input * gradient), channel mean, symmetric pair average",
        "minimum_offset_bins": minimum_offset,
        "minimum_distance_bp": minimum_offset * pooled_bin_size,
        "class_names": class_names, "random_control_seeds": controls,
        "checkpoints": checkpoint_paths, "groups": group_summary,
        "comparisons": comparison_summary,
        "structure_level_mass_medians": {
            name: float(structure_mass[name].median()) if structure_mass[name].notna().any() else None
            for name in mass_columns
        },
        "example_structure_ids": [
            min(str(value) for value, label in zip(fields["structure_id"], fields["labels"])
                if str(label) == name) for name in class_names
        ],
        "input_sha256": {
            str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in [dataset_path] + checkpoint_paths
        },
        "limitations": [
            "parameter-reset control also resets batch-normalization running states",
            "stability and parameter dependence do not prove attribution faithfulness or biology",
            "training seeds and model pairs are not independent biological samples",
            "test data reused descriptively; no model selection or retraining",
        ],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "per_sample_attribution.csv", index=False)
    pair_rows.to_csv(output / "map_comparisons.csv", index=False)
    structure_pairs.to_csv(output / "per_structure_comparisons.csv", index=False)
    structure_mass.to_csv(output / "per_structure_mass.csv")
    pd.DataFrame(group_summary).to_csv(output / "class_correctness_summary.csv", index=False)
    np.savez_compressed(output / "attribution_maps.npz", trained=trained_maps,
                        random=random_maps, contact_maps=matrices.mean(axis=1), **fields)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary
