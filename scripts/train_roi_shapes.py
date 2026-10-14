"""Fixed six-model Task 1 ROI experiment, selected without reading test performance."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from microc_foundation.discovery_benchmark import fit_logistic, predict_logistic
from microc_foundation.metrics import classification_metrics
from microc_foundation.roi_classification import (
    density_preserving_shuffle,
    input_gradient_map,
    paired_logits,
    region_grid,
    roi_shape_input,
    train_roi_classifier,
)

CLASSES = ["CHID", "CHIN", "OPCID"]
SEEDS = (2026, 2027, 2028)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def region_enrichment(attribution, mask):
    upper = np.triu(np.ones((64, 64), dtype=bool), k=2)
    area = float(mask[upper].mean())
    total = float(attribution[upper].sum())
    mass = float(attribution[upper & mask].sum()/total) if total > 1e-15 else None
    return {"area_fraction": area, "mass_fraction": mass,
            "enrichment": mass/area if mass is not None and area > 0 else None}


def plot_confusion(metrics, path):
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(3), xticklabels=CLASSES, yticks=range(3), yticklabels=CLASSES,
           xlabel="Predicted", ylabel="True")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        parser.error("use a fresh directory, never overwrite old evidence")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    with np.load(args.dataset) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    with np.load(args.reference) as ref:
        for key in ["structure_id", "labels", "chrom", "annotation_start", "annotation_end",
                    "center", "window_start", "window_end", "replicate", "split"]:
            np.testing.assert_array_equal(data[key], ref[key], err_msg=f"identity changed: {key}")
    if (data["matrices"].shape != (344, 2, 256, 256) or int(data["pooled_bin_size"]) != 80 or
            str(data["normalization"]) != "log1p" or not np.isfinite(data["matrices"]).all()):
        raise ValueError("requires the frozen 344-sample 80 bp log1p dataset")
    labels = np.array([CLASSES.index(x) for x in data["labels"]])
    masks = {name: data["split"] == name for name in ["train", "validation", "test"]}
    if [int(masks[k].sum()) for k in masks] != [248, 55, 41]:
        raise ValueError("original spatial split changed")
    # The adaptive ROI never reads beyond original 20,480 bp windows.
    a, b = np.flatnonzero(masks["train"]), np.flatnonzero(~masks["train"])
    if ((data["window_start"][a, None] < data["window_end"][None, b]) &
            (data["window_end"][a, None] > data["window_start"][None, b])).any():
        raise ValueError("training and held-out windows overlap")
    maps, region_masks, extents, controls = [], [], [], []
    for i in range(344):
        grid, mask, extent = region_grid(int(data["window_start"][i]), int(data["window_end"][i]),
            int(data["annotation_start"][i]), int(data["annotation_end"][i]))
        maps.append(roi_shape_input(data["matrices"][i], grid))
        region_masks.append(mask)
        extents.append(extent)
        centers = data["window_start"][i]+(np.arange(256)+.5)*80
        inside = (centers >= data["annotation_start"][i]) & (centers < data["annotation_end"][i])
        roi = data["matrices"][i][:, inside][:, :, inside]
        controls.append([np.log(data["annotation_end"][i]-data["annotation_start"][i]),
                         float(roi.mean()), float(roi.std()), float((roi > 0).mean())])
    maps = np.asarray(maps, dtype="float32")
    region_masks = np.asarray(region_masks, dtype=bool)
    controls = np.asarray(controls)
    output.mkdir(parents=True)
    np.savez_compressed(output/"roi_inputs.npz", maps=maps, masks=region_masks,
        extents=extents, structure_id=data["structure_id"], labels=data["labels"], split=data["split"],
        length_density_features=controls)
    trained, initial, summaries, completed = {}, {}, {}, []
    try:
        for strategy, penalty in [("roi_ce", 0.), ("roi_rrr", .1)]:
            summaries[strategy] = {}
            for seed in SEEDS:
                print(f"train {strategy} seed={seed}; test performance not read", flush=True)
                model, random, history, meta = train_roi_classifier(
                    maps[masks["train"]], labels[masks["train"]], region_masks[masks["train"]],
                    maps[masks["validation"]], labels[masks["validation"]],
                    seed=seed, gradient_penalty=penalty, epochs=100, patience=20)
                directory = output/strategy/f"seed_{seed}"
                directory.mkdir(parents=True)
                pd.DataFrame(history).to_csv(directory/"history.csv", index=False)
                torch.save({"model_state": model.state_dict(), "initial_model_state": random.state_dict(),
                    "class_names": CLASSES, **meta, "protocol": "task1-roi-shape-v1",
                    "train_ids": data["structure_id"][masks["train"]].tolist()}, directory/"model.pt")
                validation = classification_metrics(labels[masks["validation"]],
                    paired_logits(model, maps[masks["validation"]]).mean(1).argmax(1), CLASSES)
                trained[(strategy, seed)], initial[(strategy, seed)] = model, random
                summaries[strategy][str(seed)] = {**meta, "epochs_run": len(history),
                                                 "validation": validation}
                completed.append({"strategy": strategy, "seed": seed})
                print(f"  best={meta['best_epoch']} validation min recall={meta['selection_key'][0]:.4f}"
                      f" macro F1={validation['macro_f1']:.4f}", flush=True)
        validation_keys = {name: [min(min(m["validation"]["per_class"][c]["recall"]
            for c in CLASSES) for m in values.values()),
            float(np.mean([m["validation"]["macro_f1"] for m in values.values()]))]
            for name, values in summaries.items()}
        selected = max(validation_keys, key=lambda name: tuple(validation_keys[name]))
        save_json(output/"validation_selection.json", {"selected_strategy": selected,
            "selection_keys": validation_keys, "test_not_used_for_selection": True})
        print(f"selected using validation only: {selected}; begin full test/controls", flush=True)
        test = masks["test"]
        test_maps, test_masks, test_labels = maps[test], region_masks[test], labels[test]
        ids = data["structure_id"][test]
        prediction_rows, attribution_rows, shuffle_rows = [], [], []
        all_attributions, all_random = {}, {}
        controls_summary = {}
        for name, indices in {"length_only": [0], "density_only": [1, 2, 3],
                              "length_and_density": [0, 1, 2, 3]}.items():
            fits, probability_columns = [], []
            for label in range(3):
                _, fit = fit_logistic(controls[masks["train"]][:, indices],
                                     (labels[masks["train"]] == label).astype(int), penalty=.01)
                fits.append(fit)
                probability_columns.append(predict_logistic(controls[:, indices], fit))
            probabilities = np.stack(probability_columns, axis=1)
            controls_summary[name] = {"features": indices, "fits": fits,
                "validation": classification_metrics(labels[masks["validation"]],
                    probabilities[masks["validation"]].argmax(1), CLASSES),
                "test": classification_metrics(test_labels, probabilities[test].argmax(1), CLASSES)}
        for (strategy, seed), model in trained.items():
            logits = paired_logits(model, test_maps)
            mean_logits = logits.mean(1)
            predictions = mean_logits.argmax(1)
            m = summaries[strategy][str(seed)]
            m["test"] = classification_metrics(test_labels, predictions, CLASSES)
            m["rep1_test"] = classification_metrics(test_labels, logits[:, 0].argmax(1), CLASSES)
            m["rep2_test"] = classification_metrics(test_labels, logits[:, 1].argmax(1), CLASSES)
            m["replicate_prediction_agreement"] = float((logits[:, 0].argmax(1) == logits[:, 1].argmax(1)).mean())
            plot_confusion(m["test"], output/strategy/f"seed_{seed}"/"confusion.png")
            attributions, random_maps = [], []
            for i, identity in enumerate(ids):
                prediction_rows.append({"strategy": strategy, "seed": seed, "structure_id": identity,
                    "true_class": CLASSES[test_labels[i]], "predicted_class": CLASSES[predictions[i]],
                    "rep1_prediction": CLASSES[logits[i, 0].argmax()],
                    "rep2_prediction": CLASSES[logits[i, 1].argmax()]})
                attribution = input_gradient_map(model, test_maps[i], int(test_labels[i]))
                random = input_gradient_map(initial[(strategy, seed)], test_maps[i], int(test_labels[i]))
                attributions.append(attribution)
                random_maps.append(random)
                for kind, values in [("trained", attribution), ("random", random)]:
                    attribution_rows.append({"strategy": strategy, "seed": seed, "structure_id": identity,
                        "true_class": CLASSES[test_labels[i]], "network": kind,
                        **region_enrichment(values, test_masks[i])})
            all_attributions[(strategy, seed)] = np.asarray(attributions)
            all_random[(strategy, seed)] = np.asarray(random_maps)
            np.savez_compressed(output/strategy/f"seed_{seed}"/"attributions.npz",
                trained=attributions, random=random_maps, structure_id=ids)
            # All models/draws/regions retained, including negative changes.
            for region in ("all", "structure", "background"):
                for draw in range(20):
                    shuffled = np.stack([density_preserving_shuffle(test_maps[i],
                        np.ones((64, 64), bool) if region == "all" else
                        (test_masks[i] if region == "structure" else ~test_masks[i]),
                        seed=9100+draw) for i in range(41)])
                    altered = paired_logits(model, shuffled).mean(1)
                    for i, identity in enumerate(ids):
                        shuffle_rows.append({"strategy": strategy, "seed": seed, "region": region,
                            "draw": draw, "structure_id": identity, "true_class": CLASSES[test_labels[i]],
                            "correct": bool(altered[i].argmax() == test_labels[i]),
                            "true_logit_drop": float(mean_logits[i, test_labels[i]]-altered[i, test_labels[i]])})
            print(f"evaluated {strategy} {seed}: accuracy={m['test']['accuracy']:.4f}"
                  f" macro F1={m['test']['macro_f1']:.4f}", flush=True)
        pd.DataFrame(prediction_rows).to_csv(output/"test_predictions.csv", index=False)
        attribution_table = pd.DataFrame(attribution_rows)
        attribution_table.to_csv(output/"attribution_quality.csv", index=False)
        shuffles = pd.DataFrame(shuffle_rows)
        shuffles.to_csv(output/"shape_shuffle_controls.csv", index=False)
        group = shuffles.groupby(["strategy", "seed", "region", "draw"], sort=False)
        shuffle_summary = group.agg(accuracy=("correct", "mean"),
            median_true_logit_drop=("true_logit_drop", "median")).reset_index()
        shuffle_summary.to_csv(output/"shuffle_summary.csv", index=False)
        # Fixed first sample per class, not the best looking/correct predictions.
        chosen = [min(np.flatnonzero(test_labels == c), key=lambda i: str(ids[i])) for c in range(3)]
        fig, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
        for row, i in enumerate(chosen):
            for col, values in enumerate([test_maps[i].mean(0), all_attributions[(selected, 2026)][i],
                                          all_random[(selected, 2026)][i]]):
                vmax = max(float(np.percentile(np.abs(values), 99)), 1e-9)
                axes[row, col].imshow(values, origin="lower", cmap="RdBu_r" if col == 0 else "magma",
                    vmin=-vmax if col == 0 else 0, vmax=vmax)
                axes[row, col].contour(test_masks[i], levels=[.5], colors="cyan", linewidths=.7)
                axes[row, col].set_title(f"{ids[i]} | {['ROI shape','trained attribution','random attribution'][col]}")
        fig.savefig(output/"fixed_saliency_examples.png", dpi=180)
        plt.close(fig)
        acceptance = {}
        for strategy, models in summaries.items():
            acceptance[strategy] = {"every_seed_class_recall_ge_0_75": all(
                v["recall"] >= .75 for m in models.values() for v in m["test"]["per_class"].values()),
                "every_seed_class_f1_ge_0_65": all(v["f1"] >= .65 for m in models.values()
                    for v in m["test"]["per_class"].values()),
                "every_seed_replicate_agreement_ge_0_85": all(
                    m["replicate_prediction_agreement"] >= .85 for m in models.values())}
        summary = {"status": "completed_exploratory_task1_roi_experiment",
            "protocol": "task1-roi-shape-v1", "selected_strategy": selected,
            "validation_selection_keys": validation_keys, "models": summaries,
            "metadata_controls": controls_summary, "classification_targets": acceptance,
            "input_sha256": {args.dataset: file_hash(args.dataset), args.reference: file_hash(args.reference)},
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "threads": 1}, "elapsed_seconds": time.perf_counter()-started,
            "limitations": ["Known annotation boundaries used for class-blind ROI; not annotation-free detection",
                "Original labels/splits all retained; older test areas have already been seen in development",
                "Same test set for exploratory comparison, not a fresh blind generalization claim",
                "Gradient ratio is an adaptation, not a reproduction of the RRR paper",
                "Shape controls preserve transformed offset histograms, not native read totals or biology",
                "Classification goals and attribution evidence evaluated separately; no automatic scientific completion",
                "CHID test n=4; three seeds do not create new independent biological samples",
                "All variants/seeds/incorrect samples preserved; chosen strategy uses validation only"]}
        save_json(output/"summary.json", summary)
        print(json.dumps({"selected": selected, "goals": acceptance,
            "test": {k: {seed: m['test'] for seed, m in v.items()} for k, v in summaries.items()}},
                         indent=2), flush=True)
    except Exception as error:
        save_json(output/"failure.json", {"status": "failed", "error": str(error),
                                          "completed_models": completed})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
