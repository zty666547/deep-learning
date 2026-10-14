"""One frozen structure-centred benchmark; no result-based tuning or candidate reselection."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from microc_foundation.annotations import read_structures_csv
from microc_foundation.centered_detection import (
    centered_registry,
    component_binary_metrics,
    component_weights,
    detector_outputs,
    purged_centered_splits,
    train_centered_detector,
)
from microc_foundation.contrastive_texture import texture_input
from microc_foundation.discovery import extract_contact_features
from microc_foundation.discovery_benchmark import (
    binary_metrics,
    fit_logistic,
    predict_logistic,
    select_nonoverlapping_windows,
)
from microc_foundation.io import open_cooler

SEEDS = (2026, 2027, 2028)
LENGTH = 4_641_652


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def quality(features):
    reasons = []
    for rep, item in enumerate(features, 1):
        for name, failed in [
            ("low_total", item["total_contacts"] < 1),
            ("sparse", item["nonzero_fraction"] < .01),
            ("missing", item["finite_fraction"] < .99),
            ("zero_bins", item["zero_bin_fraction"] > .03),
            ("zero_run", item["largest_zero_run_fraction"] > .03),
        ]:
            if failed:
                reasons.append(f"rep{rep}_{name}")
    return ";".join(reasons)


def check_purging(registry, train, intervals):
    # Check actual train/query intervals, not only nominal fold IDs.
    a = registry.loc[train]
    b = intervals
    separated = ((a.end.to_numpy()[:, None]+20_480 <= b.start.to_numpy()[None, :]) |
                 (a.start.to_numpy()[:, None] >= b.end.to_numpy()[None, :]+20_480))
    if not separated.all():
        raise ValueError("actual training/query intervals violate spatial buffer")


def fold_metrics(registry, scores):
    return {"window": binary_metrics(registry.label.to_numpy(int), scores),
            "component": component_binary_metrics(registry, scores)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cool", action="append", required=True)
    parser.add_argument("--structures", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        parser.error("requires a fresh output directory")
    if len(args.cool) != 2 or len({Path(p).resolve() for p in args.cool}) != 2:
        parser.error("need distinct WT1 and WT2 matrices in that order")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    structures = read_structures_csv(args.structures)
    scan = pd.read_csv(args.windows)
    if len(scan) != 1807 or len(structures) != 344:
        raise ValueError("frozen input identities require 1807 windows / 344 annotations")
    registry = centered_registry(structures, scan)
    old_grid = select_nonoverlapping_windows(scan, structures, chrom="NC_000913.3",
        chrom_length=LENGTH, window_bp=20_480, folds=5)
    if len(registry) != 389 or len(old_grid) != 187:
        raise ValueError("pre-QC registry changed: expect 343 unique centres + 46 backgrounds")
    output.mkdir(parents=True)
    completed = []
    try:
        reasons, maps, density = [], [], []
        scan_pass = scan.loc[scan.quality_pass].copy().reset_index(drop=True)
        scan_pass["fold"] = np.floor(scan_pass.center/LENGTH*5).astype(int)
        scan_maps = []
        with ExitStack() as stack:
            coolers = [stack.enter_context(open_cooler(path)) for path in args.cool]
            if any(c.binsize != 160 or list(c.chromnames) != ["NC_000913.3"] or
                   int(c.chromsizes["NC_000913.3"]) != LENGTH for c in coolers):
                raise ValueError("incorrect grid or reference")
            for row in registry.itertuples(index=False):
                counts = [c.matrix(balance=False).fetch(f"{row.chrom}:{row.start}-{row.end}")
                          for c in coolers]
                features = [extract_contact_features(x) for x in counts]
                reason = quality(features)
                reasons.append(reason)
                if row.label == 0 and reason:
                    raise ValueError("old background QC differs from recomputation")
                maps.append([texture_input(x) for x in counts])
                density.append([[np.log1p(f["total_contacts"]), f["nonzero_fraction"]]
                                for f in features])
            for row in scan_pass.itertuples(index=False):
                scan_maps.append([texture_input(c.matrix(balance=False).fetch(
                    f"{row.chrom}:{row.start}-{row.end}")) for c in coolers])
        registry["quality_reasons"] = reasons
        registry["quality_pass"] = registry.quality_reasons == ""
        registry.to_csv(output/"all_centered_windows.csv", index=False)
        keep = registry.quality_pass.to_numpy()
        maps = np.asarray(maps, dtype="float32")[keep]
        density = np.asarray(density, dtype=float)[keep]
        registry = registry.loc[keep].reset_index(drop=True)
        scan_maps = np.asarray(scan_maps, dtype="float32")
        registry.to_csv(output/"evaluation_registry.csv", index=False)
        np.savez_compressed(output/"inputs.npz", maps=maps, density=density,
            sample_id=registry.sample_id.to_numpy(str), scan_maps=scan_maps,
            scan_window_id=scan_pass.window_id.to_numpy(str))
        splits = list(purged_centered_splits(registry))
        predictions = registry.copy()
        models = {"density_rep2": []}
        for seed in SEEDS:
            for name in ("cnn_rep2", "cnn_rep1", "random_rep2"):
                models[f"{name}_seed{seed}"] = []
                predictions[f"{name}_seed{seed}"] = np.nan
            scan_pass[f"cnn_rep2_seed{seed}"] = np.nan
        predictions["density_rep2"] = np.nan
        labels = registry.label.to_numpy(int)
        for fold, train, test in splits:
            scan_test = scan_pass.fold.to_numpy() == fold
            check_purging(registry, train, registry.loc[test])
            check_purging(registry, train, scan_pass.loc[scan_test])
            print(f"fold={fold} train={train.sum()} test={test.sum()} scan={scan_test.sum()}",
                  flush=True)
            _, fit = fit_logistic(density[train, 0], labels[train], penalty=.01)
            scores = predict_logistic(density[test, 1], fit)
            predictions.loc[test, "density_rep2"] = scores
            models["density_rep2"].append({"fold": fold, "fit": fit,
                                           **fold_metrics(registry.loc[test], scores)})
            for seed in SEEDS:
                directory = output/f"fold_{fold}"/f"seed_{seed}"
                directory.mkdir(parents=True)
                trained, initial, history = train_centered_detector(maps[train, 0],
                    registry.loc[train], seed=seed, epochs=20)
                train_weights, groups = component_weights(registry.loc[train])
                pd.DataFrame({"sample_id": registry.loc[train, "sample_id"].to_numpy(),
                    "label": labels[train], "weight": train_weights,
                    "component": groups}).to_csv(directory/"training_weights.csv", index=False)
                pd.DataFrame(history).to_csv(directory/"history.csv", index=False)
                torch.save({"model_state": trained.state_dict(),
                    "initial_model_state": initial.state_dict(), "seed": seed,
                    "train_sample_ids": registry.loc[train, "sample_id"].tolist(),
                    "test_sample_ids": registry.loc[test, "sample_id"].tolist(),
                    "fold": fold, "epochs": 20, "trained_on": "WT1 only"},
                    directory/"detector.pt")
                rep1, embedding1 = detector_outputs(trained, maps[:, 0])
                rep2, embedding2 = detector_outputs(trained, maps[:, 1])
                _, random1 = detector_outputs(initial, maps[:, 0])
                _, random2 = detector_outputs(initial, maps[:, 1])
                _, fit = fit_logistic(random1[train], labels[train], penalty=.01)
                random_scores = predict_logistic(random2[test], fit)
                for name, scores in [("cnn_rep1", rep1[test]), ("cnn_rep2", rep2[test]),
                                     ("random_rep2", random_scores)]:
                    key = f"{name}_seed{seed}"
                    predictions.loc[test, key] = scores
                    models[key].append({"fold": fold,
                        **fold_metrics(registry.loc[test], scores),
                        **({"fit": fit} if name == "random_rep2" else {})})
                scan_scores, scan_embeddings = detector_outputs(trained, scan_maps[scan_test, 1])
                scan_pass.loc[scan_test, f"cnn_rep2_seed{seed}"] = scan_scores
                np.savez_compressed(directory/"features.npz", learned_rep1=embedding1,
                    learned_rep2=embedding2, random_rep1=random1, random_rep2=random2,
                    sample_id=registry.sample_id.to_numpy(str), train=train, test=test,
                    scan_embeddings=scan_embeddings,
                    scan_window_id=scan_pass.loc[scan_test, "window_id"].to_numpy(str))
                completed.append({"fold": fold, "seed": seed})
                print(f"  seed={seed} WT2 component AUC="
                      f"{models[f'cnn_rep2_seed{seed}'][-1]['component']['roc_auc']:.4f}",
                      flush=True)
        for key, folds in models.items():
            if not np.isfinite(predictions[key]).all():
                raise ValueError("incomplete out-of-fold probabilities")
            models[key] = {"folds": folds,
                "mean_component_auc": float(np.mean([f["component"]["roc_auc"] for f in folds])),
                "sd_component_auc": float(np.std([f["component"]["roc_auc"] for f in folds], ddof=1)),
                "mean_window_auc": float(np.mean([f["window"]["roc_auc"] for f in folds])),
                "pooled_window": binary_metrics(labels, predictions[key].to_numpy())}
        cnn_columns = [f"cnn_rep2_seed{s}" for s in SEEDS]
        predictions["cnn_rep2_mean_probability"] = predictions[cnn_columns].mean(axis=1)
        scan_pass["cnn_rep2_mean_probability"] = scan_pass[cnn_columns].mean(axis=1)
        full_scan = scan.merge(scan_pass[["window_id", "fold", *cnn_columns,
            "cnn_rep2_mean_probability"]], on="window_id", how="left", validate="one_to_one")
        predictions.to_csv(output/"centered_oof_predictions.csv", index=False)
        full_scan.to_csv(output/"genome_oof_scores.csv", index=False)
        old_predictions = old_grid.merge(scan_pass[["window_id", *cnn_columns,
            "cnn_rep2_mean_probability"]], on="window_id", validate="one_to_one")
        old_predictions.to_csv(output/"old_grid_oof_predictions.csv", index=False)
        old_fold_metrics = [binary_metrics(g.label.to_numpy(int),
            g.cnn_rep2_mean_probability.to_numpy()) for _, g in old_predictions.groupby("fold")]
        all_registry = pd.read_csv(output/"all_centered_windows.csv").fillna("")
        annotation_rows = []
        for row in structures.itertuples(index=False):
            match = all_registry.loc[all_registry.structure_ids.str.split(";").map(
                lambda identities, identity=row.structure_id: identity in identities)].iloc[0]
            center = predictions.loc[predictions.sample_id == match.sample_id]
            nearby = scan_pass.loc[(scan_pass.center-int(row.center)).abs() <= 1280]
            probability = float(center.cnn_rep2_mean_probability.iloc[0]) if len(center) else None
            score = float(nearby.cnn_rep2_mean_probability.max()) if len(nearby) else None
            annotation_rows.append({"structure_id": row.structure_id,
                "structure_type": row.structure_type, "chrom": row.chrom,
                "annotation_start": row.start, "annotation_end": row.end,
                "annotation_center": row.center, "sample_id": match.sample_id,
                "center_qc_pass": bool(match.quality_pass),
                "quality_reasons": match.quality_reasons,
                "center_probability": probability,
                "center_recalled": probability is not None and probability >= .5,
                "scan_scored_windows": len(nearby), "scan_max_probability": score,
                "scan_recalled": score is not None and score >= .5})
        coverage = pd.DataFrame(annotation_rows)
        coverage.to_csv(output/"annotation_recall.csv", index=False)
        per_class = {name: {"total": len(g), "center_qc_pass": int(g.center_qc_pass.sum()),
            "center_recalled": int(g.center_recalled.sum()),
            "scan_scorable": int((g.scan_scored_windows > 0).sum()),
            "scan_recalled": int(g.scan_recalled.sum()),
            "scan_recall_all_annotations": float(g.scan_recalled.mean())}
            for name, g in coverage.groupby("structure_type")}
        summary = {"status": "exploratory_centered_detection_completed",
            "protocol": "task2-centered-detection-v1", "seeds": list(SEEDS),
            "pre_qc_unique_positive": 343, "pre_qc_background": 46,
            "retained_unique_positive": int(labels.sum()),
            "retained_background": int((labels == 0).sum()), "n": len(registry),
            "threshold": .5, "epochs": 20, "purge_bp": 20480,
            "training_replicate": "WT1", "primary_evaluation_replicate": "WT2",
            "models": models, "per_class": per_class,
            "old_grid_mean_fold_auc": float(np.mean([f["roc_auc"] for f in old_fold_metrics])),
            "old_grid_folds": old_fold_metrics,
            "scan": {"all_windows": len(full_scan), "qc_scored_windows": len(scan_pass),
                "recalled_annotations": int(coverage.scan_recalled.sum()),
                "all_annotations": len(coverage),
                "scorable_annotations": int((coverage.scan_scored_windows > 0).sum()),
                "all_annotation_recall": float(coverage.scan_recalled.mean()),
                "unannotated_background_ensemble_specificity": float((predictions.loc[
                    predictions.label == 0, "cnn_rep2_mean_probability"] < .5).mean())},
            "splits": [{"fold": f, "train_n": int(a.sum()), "test_n": int(b.sum()),
                "train_sample_ids": registry.loc[a, "sample_id"].tolist(),
                "test_sample_ids": registry.loc[b, "sample_id"].tolist()} for f, a, b in splits],
            "input_sha256": {str(p): sha256(p) for p in [*args.cool, args.structures, args.windows]},
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                "torch": torch.__version__, "numpy": np.__version__, "threads": 1},
            "elapsed_seconds": time.perf_counter()-started,
            "limitations": ["Existing regions have been inspected; not a new blind holdout",
                "Centred labels change the population; do not replace any-overlap results",
                "Unannotated backgrounds are not verified negatives",
                "Overlapping component weights do not yield independent biological samples",
                "Replicate transfer is not external condition or species validation",
                "Component-weighted CNN and class-balanced logistic controls use different objectives",
                "Fixed final epoch, all seeds/folds retained, no performance-driven tuning",
                "Cross-fold embeddings are different coordinate systems; do not cluster pooled embeddings",
                "Known-centre scores are not novelty scores or confirmation of five candidates"]}
        write_json(output/"summary.json", summary)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
        names = ["density_rep2", *[f"random_rep2_seed{s}" for s in SEEDS],
                 *[f"cnn_rep2_seed{s}" for s in SEEDS]]
        for i, key in enumerate(names):
            values = [f["component"]["roc_auc"] for f in models[key]["folds"]]
            axes[0].scatter(np.full(5, i), values)
            axes[0].plot(i, np.mean(values), "_", color="black", markersize=15)
        axes[0].axhline(.5, linestyle="--", color="gray")
        axes[0].set(xticks=range(7), xticklabels=["density", "random\n2026", "random\n2027",
            "random\n2028", "CNN\n2026", "CNN\n2027", "CNN\n2028"],
            ylim=(0, 1), ylabel="Component-weighted spatial WT2 AUC")
        axes[1].bar(list(per_class), [v["scan_recall_all_annotations"] for v in per_class.values()],
                    color="#407c9e")
        axes[1].set(ylim=(0, 1), ylabel="Genome-scan recall / all annotations",
                    title="Fixed 0.5 threshold; centre +/- 1,280 bp")
        fig.savefig(output/"centered_detection_controls.png", dpi=180)
        plt.close(fig)
        print(json.dumps({"models": {k: v["mean_component_auc"] for k, v in models.items()},
                          "scan": summary["scan"]}, indent=2), flush=True)
    except Exception as error:
        write_json(output/"failure.json", {"status": "failed", "error": str(error),
                                          "completed_fold_seeds": completed})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
