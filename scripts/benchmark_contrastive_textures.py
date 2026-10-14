"""Fixed exploratory replicate-paired deep texture benchmark with spatial purging."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from microc_foundation.contrastive_texture import (
    encode_replicates,
    representation_diagnostics,
    texture_input,
    train_pair_encoder,
)
from microc_foundation.discovery_benchmark import (
    binary_metrics,
    fit_logistic,
    predict_logistic,
    select_nonoverlapping_windows,
    spatial_splits,
)
from microc_foundation.io import open_cooler


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cool", action="append", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--structures", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        parser.error("use a fresh output directory; do not overwrite prior evidence")
    if len(args.cool) != 2 or len({Path(p).resolve() for p in args.cool}) != 2:
        parser.error("requires two distinct WT replicates")
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    windows = select_nonoverlapping_windows(
        pd.read_csv(args.windows), read_structures_csv(args.structures),
        chrom="NC_000913.3", chrom_length=4_641_652, window_bp=20_480, folds=5,
    )
    if len(windows) != 187 or windows.label.sum() != 141:
        raise ValueError("frozen benchmark identity changed; expect 187/141/46")
    splits = list(spatial_splits(windows, purge_bp=20_480))
    maps = np.empty((187, 2, 64, 64), dtype="float32")
    with ExitStack() as stack:
        coolers = [stack.enter_context(open_cooler(path)) for path in args.cool]
        if any(cool.binsize != 160 or int(cool.chromsizes["NC_000913.3"]) != 4_641_652 for cool in coolers):
            raise ValueError("incorrect resolution/reference")
        for index, row in enumerate(windows.itertuples(index=False)):
            for rep, cool in enumerate(coolers):
                counts = cool.matrix(balance=False).fetch(f"{row.chrom}:{row.start}-{row.end}")
                maps[index, rep] = texture_input(counts)
    output.mkdir(parents=True)
    np.savez_compressed(output / "texture_inputs.npz", maps=maps,
                        window_id=windows.window_id.to_numpy(str), label=windows.label.to_numpy(),
                        fold=windows.fold.to_numpy())
    labels = windows.label.to_numpy(int)
    model_names = ("learned_joint", "random_joint", "learned_rep1_to_rep2", "learned_rep2_to_rep1")
    scores = {name: np.full(187, np.nan) for name in model_names}
    folds = {name: [] for name in model_names}
    diagnostics, histories, shuffle_fits = [], [], []
    try:
        for fold, train, test in splits:
            print(f"fold={fold} training_windows={int(train.sum())} test_windows={int(test.sum())}", flush=True)
            trained, random, history = train_pair_encoder(maps[train], seed=2026+fold, epochs=20, batch_size=32)
            learned_features = encode_replicates(trained, maps)
            random_features = encode_replicates(random, maps)
            directory = output / f"fold_{fold}"
            directory.mkdir()
            torch.save({"model_state": trained.state_dict(), "initial_model_state": random.state_dict(),
                        "seed": 2026+fold, "trained_window_ids": windows.loc[train, "window_id"].tolist(),
                        "heldout_window_ids": windows.loc[test, "window_id"].tolist(),
                        "epochs": 20, "objective": "replicate-paired mean NT-Xent; temperature .2"}, directory / "encoder.pt")
            np.savez_compressed(directory / "embeddings.npz", learned=learned_features, random=random_features,
                                window_id=windows.window_id.to_numpy(str), train=train, test=test)
            pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
            histories.append({"fold": fold, "history": history})
            diagnostics.append({"fold": fold, "learned_train": representation_diagnostics(learned_features[train]),
                                "learned_test": representation_diagnostics(learned_features[test]),
                                "random_test": representation_diagnostics(random_features[test])})
            joint = learned_features.mean(axis=1)
            variants = {"learned_joint": (joint, joint),
                        "random_joint": (random_features.mean(axis=1), random_features.mean(axis=1)),
                        "learned_rep1_to_rep2": (learned_features[:, 0], learned_features[:, 1]),
                        "learned_rep2_to_rep1": (learned_features[:, 1], learned_features[:, 0])}
            for name, (fit_features, query_features) in variants.items():
                _, fit = fit_logistic(fit_features[train], labels[train], penalty=.01)
                predicted = predict_logistic(query_features[test], fit)
                scores[name][test] = predicted
                folds[name].append({"fold": fold, "train_window_ids": windows.loc[train, "window_id"].tolist(),
                                    "test_window_ids": windows.loc[test, "window_id"].tolist(),
                                    "metrics": binary_metrics(labels[test], predicted), "fit": fit})
            shuffle_fits.append((train, test, joint))
        rng = np.random.default_rng(2026)
        null = []
        for permutation in range(20):
            shuffled = labels.copy()
            for _, _, test in splits:
                shuffled[test] = rng.permutation(labels[test])
            aucs = []
            for train, test, joint in shuffle_fits:
                _, fit = fit_logistic(joint[train], shuffled[train], penalty=.01)
                aucs.append(binary_metrics(shuffled[test], predict_logistic(joint[test], fit))["roc_auc"])
            null.append(float(np.mean(aucs)))
            if (permutation+1) % 5 == 0:
                print(f"label control {permutation+1}/20", flush=True)
        models = {}
        for name in model_names:
            windows[f"{name}_probability"] = scores[name]
            aucs = [row["metrics"]["roc_auc"] for row in folds[name]]
            models[name] = {"folds": folds[name], "mean_fold_roc_auc": float(np.mean(aucs)),
                            "sd_fold_roc_auc": float(np.std(aucs, ddof=1)),
                            "pooled_oof_metrics": binary_metrics(labels, scores[name])}
        summary = {"status": "exploratory_deep_texture_benchmark_completed",
                   "protocol": "task2-purged-replicate-contrastive-v1", "n": 187,
                   "positive": 141, "unannotated_background": 46, "embedding_dimensions": 128,
                   "epochs": 20, "batch_size_max": 32, "base_seed": 2026,
                   "temperature": .2, "optimizer": "AdamW lr=.001 weight_decay=.0001",
                   "models": models, "diagnostics": diagnostics, "histories": histories,
                   "shuffle_control": {"count": 20, "seed": 2026, "mean_fold_roc_auc": null,
                                       "quantiles_2_5_97_5": np.quantile(null, [.025, .975]).tolist(),
                                       "is_descriptive_not_p_value": True},
                   "elapsed_seconds": time.perf_counter()-started,
                   "input_sha256": {path: hash_file(path) for path in [args.windows, args.structures, *args.cool]},
                   "limitations": ["Existing spatial benchmark regions have already been inspected; not a new blind holdout",
                                   "Encoder sees only purged training maps, without structure labels; classifier sees train labels only",
                                   "Fixed one architecture/base seed and 20 epochs; no test-based checkpoint or parameter selection",
                                   "Random same-initialization network and all transfer directions reported, not cherry-picked",
                                   "Unannotated background is not a verified negative; any-overlap labels can be ambiguous",
                                   "Natural biological-replicate pairs, not a reproduction of ImageNet SimCLR",
                                   "20 shuffled-label controls are sparse/descriptive and do not preserve all spatial dependence",
                                   "Benchmark does not reselect five candidates, create new clusters, or prove biological novelty"]}
        windows.to_csv(output / "out_of_fold_windows.csv", index=False)
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for index, name in enumerate(model_names):
            aucs = [row["metrics"]["roc_auc"] for row in folds[name]]
            axes[0].scatter(np.full(5, index), aucs)
            axes[0].plot(index, np.mean(aucs), "_", color="black", markersize=20)
        axes[0].axhline(.5, linestyle="--", color="gray")
        axes[0].set(xticks=range(4), xticklabels=[name.replace("_", "\n") for name in model_names],
                    ylabel="Spatial held-out ROC AUC", ylim=(0, 1), title="Purged train-only deep textures")
        axes[1].hist(null, bins=8, color="#809bbb")
        axes[1].axvline(models["learned_joint"]["mean_fold_roc_auc"], color="#c64a38", label="learned joint")
        axes[1].set(xlim=(0, 1), xlabel="Mean fold AUC", ylabel="Label controls", title="20 descriptive label controls")
        axes[1].legend(frameon=False)
        fig.savefig(output / "deep_texture_controls.png", dpi=180)
        plt.close(fig)
        print(json.dumps({name: item["mean_fold_roc_auc"] for name, item in models.items()}, indent=2), flush=True)
    except Exception as error:
        (output / "failure.json").write_text(json.dumps({"status": "failed", "error": str(error),
                                                       "completed_folds": len(histories)}, indent=2), encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
