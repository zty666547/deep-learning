"""Check Task 2 known-annotation coverage and spatial feature generalization."""

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
from microc_foundation.discovery_benchmark import (
    benchmark_features,
    select_nonoverlapping_windows,
    structure_coverage,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True, help="full scan table, not novel candidates")
    parser.add_argument("--structures", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chrom", default="NC_000913.3")
    parser.add_argument("--chrom-length", type=int, default=4_641_652)
    parser.add_argument("--window-bp", type=int, default=20_480)
    parser.add_argument("--purge-bp", type=int, default=20_480)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    windows = pd.read_csv(args.windows)
    structures = read_structures_csv(args.structures)
    selected = select_nonoverlapping_windows(
        windows, structures, chrom=args.chrom, chrom_length=args.chrom_length,
        window_bp=args.window_bp, folds=args.folds,
    )
    predictions, summary = benchmark_features(
        selected, purge_bp=args.purge_bp, permutations=args.permutations, seed=args.seed
    )
    summary["input_sha256"] = {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in [("windows", args.windows), ("structures", args.structures)]
    }
    summary["configuration"] = vars(args)
    stages = {"all_scan": windows, "quality_scan": windows.loc[windows.quality_pass],
              "nonoverlap_grid": predictions}
    for model in summary["models"]:
        stages[f"oof_{model}"] = predictions.loc[predictions[f"{model}_probability"] >= 0.5]
    coverage_tables = []
    coverage_summary = {}
    for name, stage in stages.items():
        coverage = structure_coverage(structures, stage)
        coverage["stage"] = name
        coverage_tables.append(coverage)
        coverage_summary[name] = {
            str(label): {
                "n": len(group), "mean_coverage_fraction": float(group.coverage_fraction.mean()),
                "fraction_at_least_50_percent": float((group.coverage_fraction >= 0.5).mean()),
                "fraction_at_least_90_percent": float((group.coverage_fraction >= 0.9).mean()),
            }
            for label, group in coverage.groupby("structure_type", observed=True)
        }
    summary["annotation_coverage"] = coverage_summary
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "out_of_fold_windows.csv", index=False)
    pd.concat(coverage_tables, ignore_index=True).to_csv(output / "annotation_coverage.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    names = list(summary["models"])
    for index, name in enumerate(names):
        values = [item["metrics"]["roc_auc"] for item in summary["models"][name]["folds"]]
        axes[0].scatter(np.full(len(values), index), values, alpha=0.7)
        axes[0].plot(index, np.mean(values), marker="_", markersize=20, color="black")
    axes[0].axhline(0.5, linestyle="--", color="gray")
    axes[0].set(xticks=range(len(names)), xticklabels=names, ylim=(0, 1),
                ylabel="Held-out block ROC AUC", title="Five genomic folds; one-window purge")
    null = summary["shuffle_control"]["mean_fold_roc_auc"]
    if null:
        axes[1].hist(null, bins=15, color="#809bbb", alpha=0.8)
    axes[1].axvline(summary["models"]["all_features"]["mean_fold_roc_auc"],
                    color="#c64a38", label="Observed all-feature mean")
    axes[1].set(xlim=(0, 1), xlabel="Mean fold ROC AUC", ylabel="Label shuffles",
                title="Descriptive control, not biological significance")
    axes[1].legend(fontsize=8)
    figure.savefig(output / "feature_controls.png", dpi=170)
    plt.close(figure)
    print(json.dumps({
        "selected_windows": len(predictions),
        "mean_fold_auc": {name: item["mean_fold_roc_auc"]
                          for name, item in summary["models"].items()},
        "saved": str(output / "summary.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
