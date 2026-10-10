"""Run all frozen Task 2 overlap-label controls without changing candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from microc_foundation.annotations import read_structures_csv
from microc_foundation.discovery_benchmark import select_nonoverlapping_windows
from microc_foundation.label_sensitivity import audit_label_sensitivity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True, help="complete scan, not candidate subset")
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
    structures = read_structures_csv(args.structures)
    selected = select_nonoverlapping_windows(
        pd.read_csv(args.windows), structures, chrom=args.chrom, chrom_length=args.chrom_length,
        window_bp=args.window_bp, folds=args.folds,
    )
    predictions, counts, summary = audit_label_sensitivity(
        selected, structures, purge_bp=args.purge_bp,
        permutations=args.permutations, seed=args.seed,
    )
    summary["configuration"] = vars(args)
    summary["input_sha256"] = {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in [("windows", args.windows), ("structures", args.structures)]
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "window_roles_predictions.csv", index=False)
    counts.to_csv(output / "fold_counts.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for index, (rule, result) in enumerate(summary["rules"].items()):
        axes[0].bar(index, result["background"], color="#809bbb")
        axes[0].bar(index, result["positive"], bottom=result["background"], color="#c64a38")
        axes[0].bar(index, result["excluded"],
                    bottom=result["background"] + result["positive"], color="#d8d8d8")
        if result["status"] == "evaluated":
            null = result["shuffle_control"]
            if null["permutations"]:
                axes[1].plot([index, index], [null["null_2_5_percentile"],
                                            null["null_97_5_percentile"]],
                             color="#bbb", linewidth=6, zorder=0)
            for name, marker, color in [("density_only", "s", "#809bbb"),
                                        ("shape_repeat", "^", "#57936e"),
                                        ("all_features", "o", "#c64a38")]:
                axes[1].plot(index, result["models"][name]["mean_fold_roc_auc"],
                             marker=marker, color=color, label=name if index == 0 else None)
        else:
            axes[1].text(index, 0.12, "unsupported", rotation=90, ha="center", fontsize=8)
    rules = list(summary["rules"])
    for axis in axes:
        axis.set_xticks(range(len(rules)), rules, rotation=30, ha="right")
    axes[0].set(ylabel="Reference windows", title="Blue: background / red: positive / gray: excluded")
    axes[1].axhline(0.5, linestyle="--", color="gray")
    axes[1].set(ylim=(0, 1), ylabel="Mean held-out fold ROC AUC",
                title="Gray: all-feature shuffled-label 95% range")
    axes[1].legend(fontsize=8)
    figure.savefig(output / "label_sensitivity.png", dpi=170)
    plt.close(figure)
    print(json.dumps({rule: {
        "status": result["status"], "positive": result["positive"],
        "background": result["background"], "excluded": result["excluded"],
        "auc": {name: model["mean_fold_roc_auc"] for name, model in result["models"].items()},
    } for rule, result in summary["rules"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
