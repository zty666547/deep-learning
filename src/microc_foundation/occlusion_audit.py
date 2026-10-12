"""Frozen symmetric, distance-matched perturbation controls for Task 1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from microc_foundation.model import MicroCCNN
from microc_foundation.saliency_audit import absolute_input_gradient

FRACTIONS = (0.05, 0.10, 0.20)
CONTROLS = ("distance_random", "distance_energy_random", "input_energy")


def distance_baseline(inputs: np.ndarray) -> np.ndarray:
    """Each channel's ORIGINAL offset median, mirrored; no label information."""
    values = np.asarray(inputs, dtype=np.float32)
    if (values.ndim != 3 or values.shape[-1] != values.shape[-2] or
            not np.isfinite(values).all() or
            not np.allclose(values, values.transpose(0, 2, 1), rtol=1e-5, atol=1e-5)):
        raise ValueError("expected finite symmetric channel matrices")
    size = values.shape[-1]
    baseline = np.empty_like(values)
    for offset in range(size):
        row = np.arange(size - offset)
        col = row + offset
        median = np.median(values[:, row, col], axis=1)
        baseline[:, row, col] = median[:, None]
        baseline[:, col, row] = median[:, None]
    return baseline


def top_pair_mask(
    attribution: np.ndarray, fraction: float, *, minimum_offset: int = 5,
) -> np.ndarray:
    """Fixed-index tie breaking; refuse uninformative constant/zero maps."""
    values = np.asarray(attribution, dtype=float)
    if (values.ndim != 2 or values.shape[0] != values.shape[1] or
            not np.isfinite(values).all() or (values < 0).any() or
            not 0 < fraction < 1 or not 0 < minimum_offset < len(values)):
        raise ValueError("invalid attribution, fraction or distance threshold")
    row, col = np.triu_indices(len(values), k=minimum_offset)
    eligible = values[row, col]
    if not len(eligible) or np.ptp(eligible) == 0:
        raise ValueError("constant off-diagonal attribution")
    order = np.argsort(-eligible, kind="stable")
    selected = order[:int(np.ceil(len(eligible) * fraction))]
    mask = np.zeros(values.shape, dtype=bool)
    mask[row[selected], col[selected]] = True
    return mask


def replace_symmetric_pairs(
    inputs: np.ndarray, baseline: np.ndarray, upper_mask: np.ndarray,
) -> np.ndarray:
    """Replace both replicates and mirrored positions, leaving other pixels exact."""
    values = np.asarray(inputs)
    mask = np.asarray(upper_mask)
    if (values.ndim != 3 or baseline.shape != values.shape or
            mask.shape != values.shape[-2:] or mask.dtype != bool or np.tril(mask).any()):
        raise ValueError("expected matching baseline and strict-upper boolean mask")
    result = values.copy()
    row, col = np.nonzero(mask)
    result[:, row, col] = baseline[:, row, col]
    result[:, col, row] = baseline[:, col, row]
    return result


def energy_strata(offsets: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Offset-specific rank quintiles; tied energy values always share a stratum."""
    offsets, energy = np.asarray(offsets), np.asarray(energy, dtype=float)
    if (offsets.ndim != 1 or offsets.shape != energy.shape or
            not np.isfinite(energy).all() or (energy < 0).any()):
        raise ValueError("expected matching offsets and finite nonnegative energy")
    strata = np.zeros(len(offsets), dtype=int)
    for offset in np.unique(offsets):
        selected = offsets == offset
        ranks = pd.Series(energy[selected]).rank(method="average", pct=True).to_numpy()
        bins = np.clip(np.ceil(ranks * 5).astype(int) - 1, 0, 4)
        strata[selected] = int(offset) * 5 + bins
    return strata


def matched_random_selection(
    selected: np.ndarray, strata: np.ndarray, rng: np.random.Generator,
    *, pools: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Exact stratum counts; full pool allows overlaps, not forced disjointness."""
    selected, strata = np.asarray(selected), np.asarray(strata)
    if (selected.ndim != 1 or selected.dtype != bool or selected.shape != strata.shape or
            not selected.any()):
        raise ValueError("expected nonempty boolean selection and matching strata")
    result = np.zeros_like(selected)
    for key in np.unique(strata[selected]):
        pool = np.flatnonzero(strata == key) if pools is None else pools[int(key)]
        count = int((selected & (strata == key)).sum())
        result[rng.choice(pool, size=count, replace=False)] = True
    return result


def energy_ranked_selection(
    selected: np.ndarray, offsets: np.ndarray, energy: np.ndarray,
) -> np.ndarray:
    """Highest perturbation-energy pairs, with the SAME offset count profile."""
    result = np.zeros_like(selected, dtype=bool)
    for offset in np.unique(offsets[selected]):
        pool = np.flatnonzero(offsets == offset)
        count = int((selected & (offsets == offset)).sum())
        order = np.argsort(-energy[pool], kind="stable")
        result[pool[order[:count]]] = True
    return result


def _probabilities(logits: np.ndarray) -> np.ndarray:
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def audit_occlusion(
    dataset_path: str, checkpoint_paths: list[str], output_dir: str,
    *, minimum_offset: int = 5, draws: int = 20, seed: int = 2026, batch_size: int = 8,
) -> dict[str, object]:
    """Evaluate every predeclared fraction/control; never fit or select a model."""
    if len(checkpoint_paths) != 3 or len(set(checkpoint_paths)) != 3 or draws < 1 or batch_size < 1:
        raise ValueError("three distinct checkpoints and positive draws/batch size required")
    with np.load(dataset_path, allow_pickle=False) as data:
        test = data["split"].astype(str) == "test"
        matrices = data["matrices"][test].astype(np.float32)
        fields = {name: data[name][test] for name in (
            "structure_id", "labels", "chrom", "window_start", "window_end"
        )}
        bin_size = int(data["pooled_bin_size"])
    if (matrices.ndim != 4 or not len(matrices) or not np.isfinite(matrices).all() or
            len(set(fields["structure_id"])) != len(matrices) or
            not 0 < minimum_offset < matrices.shape[-1]):
        raise ValueError("invalid test matrices, IDs or distance threshold")
    if not np.all(fields["window_end"] - fields["window_start"] == matrices.shape[-1] * bin_size):
        raise ValueError("window extents do not match bins")
    size = matrices.shape[-1]
    row, col = np.triu_indices(size, k=minimum_offset)
    offsets = col - row
    masks = np.zeros((3, len(matrices), len(FRACTIONS), size, size), dtype=bool)
    records, skipped, class_names = [], [], None
    for model_index, path in enumerate(checkpoint_paths):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        names = [str(name) for name in checkpoint["class_names"]]
        if class_names is not None and names != class_names:
            raise ValueError("checkpoint class orders must match")
        class_names = names
        mean = np.asarray(checkpoint["channel_mean"], dtype=np.float32).reshape(-1, 1, 1)
        std = np.asarray(checkpoint["channel_std"], dtype=np.float32).reshape(-1, 1, 1)
        if (len(mean) != matrices.shape[1] or len(std) != matrices.shape[1] or
                not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
            raise ValueError("invalid training scaler")
        model = MicroCCNN(matrices.shape[1], len(names))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        for index, matrix in enumerate(matrices):
            values = (matrix - mean) / std
            target = names.index(str(fields["labels"][index]))
            attribution, logits = absolute_input_gradient(
                model, torch.from_numpy(values[None]), torch.tensor([target], dtype=torch.long)
            )
            original = float(logits[0, target])
            original_probability = float(_probabilities(logits)[0, target])
            baseline = distance_baseline(values)
            energy = ((values[:, row, col] - baseline[:, row, col])**2).mean(axis=0)
            strata = energy_strata(offsets, energy)
            pools = [{int(key): np.flatnonzero(groups == key) for key in np.unique(groups)}
                     for groups in (offsets, strata)]
            try:
                top_masks = [top_pair_mask(attribution[0], fraction, minimum_offset=minimum_offset)
                             for fraction in FRACTIONS]
            except ValueError as error:
                if "constant off-diagonal" not in str(error):
                    raise
                skipped.append({"model_index": model_index,
                                "structure_id": str(fields["structure_id"][index]), "reason": str(error)})
                continue
            for fraction_index, (fraction, top) in enumerate(zip(FRACTIONS, top_masks)):
                masks[model_index, index, fraction_index] = top
                selected = top[row, col]
                selections = [("top_attribution", -1, selected),
                              ("input_energy", -1, energy_ranked_selection(selected, offsets, energy))]
                for control_index, (method, groups) in enumerate(zip(CONTROLS[:2], (offsets, strata))):
                    rng = np.random.default_rng(np.random.SeedSequence(
                        [seed, model_index, index, fraction_index, control_index]
                    ))
                    selections.extend((method, draw, matched_random_selection(
                        selected, groups, rng, pools=pools[control_index]
                    )) for draw in range(draws))
                perturbed = []
                for _, _, selection in selections:
                    mask = np.zeros((size, size), dtype=bool)
                    mask[row[selection], col[selection]] = True
                    perturbed.append(replace_symmetric_pairs(values, baseline, mask))
                changed_logits = []
                with torch.no_grad():
                    for start in range(0, len(perturbed), batch_size):
                        batch = torch.from_numpy(np.stack(perturbed[start:start + batch_size]))
                        changed_logits.append(model(batch).numpy())
                changed_logits = np.concatenate(changed_logits)
                changed_probabilities = _probabilities(changed_logits)[:, target]
                for position, (method, draw, selection) in enumerate(selections):
                    signed = original - float(changed_logits[position, target])
                    records.append({
                        "model_index": model_index, "structure_id": str(fields["structure_id"][index]),
                        "true_class": str(fields["labels"][index]), "fraction": fraction,
                        "method": method, "draw": draw,
                        "original_prediction": names[int(logits[0].argmax())],
                        "correct": bool(logits[0].argmax() == target),
                        "original_logit": original, "original_probability": original_probability,
                        "abs_logit_change": abs(signed), "signed_logit_drop": signed,
                        "probability_drop": original_probability - float(changed_probabilities[position]),
                        "selected_pairs": int(selection.sum()),
                        "mean_replacement_energy": float(energy[selection].mean()),
                        "overlap_with_top_fraction": float((selection & selected).sum() / selected.sum()),
                        "mask_sha256": hashlib.sha256(selection.tobytes()).hexdigest(),
                    })
    rows = pd.DataFrame(records)
    if rows.empty:
        raise ValueError("no nonconstant attribution maps could be evaluated")
    paired = []
    for keys, group in rows.groupby(
        ["model_index", "structure_id", "true_class", "correct", "fraction"], observed=True
    ):
        top = group.loc[group.method == "top_attribution"].iloc[0]
        for method in CONTROLS:
            control = group.loc[group.method == method]
            paired.append({
                **dict(zip(("model_index", "structure_id", "true_class", "correct", "fraction"), keys)),
                "control": method, "control_draws": len(control),
                "top_abs_change": float(top.abs_logit_change),
                "control_abs_change_median": float(control.abs_logit_change.median()),
                "abs_change_difference": float(top.abs_logit_change - control.abs_logit_change.median()),
                "top_signed_logit_drop": float(top.signed_logit_drop),
                "control_signed_logit_drop_median": float(control.signed_logit_drop.median()),
                "top_probability_drop": float(top.probability_drop),
                "control_probability_drop_median": float(control.probability_drop.median()),
                "top_replacement_energy": float(top.mean_replacement_energy),
                "control_replacement_energy_median": float(control.mean_replacement_energy.median()),
                "control_overlap_with_top_median": float(control.overlap_with_top_fraction.median()),
            })
    paired = pd.DataFrame(paired)
    numeric = [name for name in paired if name not in (
        "model_index", "structure_id", "true_class", "correct", "fraction", "control", "control_draws"
    )]
    per_structure = paired.groupby(
        ["structure_id", "true_class", "fraction", "control"], observed=True, as_index=False
    )[numeric].median()
    aggregate = []
    for label, source in [("all", per_structure)] + list(per_structure.groupby("true_class", observed=True)):
        for (fraction, control), group in source.groupby(["fraction", "control"], observed=True):
            aggregate.append({
                "class_group": str(label), "fraction": float(fraction), "control": str(control),
                "structure_n": len(group),
                **{name + "_median": float(group[name].median()) for name in numeric},
                "fraction_positive_difference": float((group.abs_change_difference > 0).mean()),
            })
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "all_perturbations.csv", index=False)
    paired.to_csv(output / "paired_model_results.csv", index=False)
    per_structure.to_csv(output / "per_structure_results.csv", index=False)
    pd.DataFrame(aggregate).to_csv(output / "group_summary.csv", index=False)
    pd.DataFrame(skipped, columns=["model_index", "structure_id", "reason"]).to_csv(
        output / "skipped_maps.csv", index=False
    )
    np.savez_compressed(output / "top_masks.npz", masks=masks, fractions=np.array(FRACTIONS), **fields)
    summary = {
        "protocol": "task1-distance-occlusion-v1", "test_structures": len(matrices),
        "minimum_offset_bins": minimum_offset, "minimum_distance_bp": minimum_offset * bin_size,
        "fractions": list(FRACTIONS), "draws_per_random_control": draws,
        "seed": seed, "rng_components": ["seed", "model_index", "test_array_index", "fraction_index", "control_index"],
        "checkpoint_paths": checkpoint_paths, "class_names": class_names,
        "replacement": "original sample/channel exact-offset median in standardized input",
        "primary_metric": "absolute true-class logit change; signed drops retained",
        "perturbation_rows": len(rows), "skipped_maps": skipped, "aggregates": aggregate,
        "input_sha256": {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                         for path in [dataset_path] + checkpoint_paths},
        "limitations": ["reused test regions; no retraining/model selection", "replacement may be out of distribution",
                        "distance/energy strata do not fully match spatial texture", "seeds/draws are not biological samples",
                        "absolute change is not directional support for the target class"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary
