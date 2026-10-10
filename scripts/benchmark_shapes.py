"""Exploratory pooled spatial and distance-centered texture benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from microc_foundation.annotations import read_structures_csv
from microc_foundation.discovery_benchmark import select_nonoverlapping_windows
from microc_foundation.shape_benchmark import (
    evaluate_spatial_representation,
    extract_replicate_shapes,
)


def file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cool", action="append", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--structures", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chrom", default="NC_000913.3")
    parser.add_argument("--chrom-length", type=int, default=4_641_652)
    parser.add_argument("--window-bp", type=int, default=20_480)
    parser.add_argument("--purge-bp", type=int, default=20_480)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--minimum-offset", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    selected = select_nonoverlapping_windows(
        pd.read_csv(args.windows), read_structures_csv(args.structures), chrom=args.chrom,
        chrom_length=args.chrom_length, window_bp=args.window_bp, folds=args.folds,
    )
    features, source = extract_replicate_shapes(
        args.cool, selected, blocks=args.blocks, minimum_offset=args.minimum_offset
    )
    raw_joint = features["raw_log"].mean(axis=1)
    texture_joint = features["distance_texture"].mean(axis=1)
    models = {
        "raw_spatial_joint": (raw_joint, raw_joint),
        "texture_joint": (texture_joint, texture_joint),
        "texture_rep1_to_rep2": (features["distance_texture"][:, 0],
                                  features["distance_texture"][:, 1]),
        "texture_rep2_to_rep1": (features["distance_texture"][:, 1],
                                  features["distance_texture"][:, 0]),
    }
    summary = {
        "protocol": "task2-shape-exploration-v1", "configuration": vars(args),
        "interpretation": "exploratory after previous benchmark; not a new untouched test set",
        "n": len(selected), "positive": int(selected.label.sum()), "source": source,
        "input_sha256": {path: file_hash(path) for path in
                         [args.windows, args.structures, *args.cool]}, "models": {},
    }
    for name, (train, test) in models.items():
        scores, result = evaluate_spatial_representation(
            train, test, selected, purge_bp=args.purge_bp,
            permutations=args.permutations if name == "texture_joint" else 0, seed=args.seed,
        )
        selected[f"{name}_probability"] = scores
        summary["models"][name] = result
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "out_of_fold_windows.csv", index=False)
    np.savez_compressed(output / "spatial_features.npz",
                        window_ids=selected.window_id.to_numpy(str), **features)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    names = list(models)
    for index, name in enumerate(names):
        result = summary["models"][name]
        aucs = [item["metrics"]["roc_auc"] for item in result["folds"]]
        axis.scatter(np.full(len(aucs), index), aucs, alpha=0.7)
        axis.plot(index, np.mean(aucs), marker="_", color="black", markersize=20)
    axis.axhline(0.5, color="gray", linestyle="--")
    axis.set(xticks=range(len(names)), xticklabels=[name.replace("_", "\n") for name in names],
             ylabel="Spatial held-out ROC AUC", ylim=(0, 1),
             title="Exploratory texture benchmark; existing test regions reused")
    figure.savefig(output / "shape_controls.png", dpi=170)
    plt.close(figure)
    print(json.dumps({name: result["mean_fold_roc_auc"]
                      for name, result in summary["models"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
