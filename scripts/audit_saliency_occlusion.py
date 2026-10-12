"""Run every fixed distance/energy-matched occlusion control on Task 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from microc_foundation.occlusion_audit import CONTROLS, audit_occlusion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-offset", type=int, default=5)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = audit_occlusion(args.dataset, args.checkpoint, args.output_dir,
                              minimum_offset=args.minimum_offset, draws=args.draws,
                              seed=args.seed, batch_size=args.batch_size)
    output = Path(args.output_dir)
    structures = pd.read_csv(output / "per_structure_results.csv")
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    colors = ["#809bbb", "#c64a38", "#57936e"]
    for index, (control, color) in enumerate(zip(CONTROLS, colors)):
        for fraction, group in structures.loc[structures.control == control].groupby("fraction", observed=True):
            x = float(fraction) * 100
            values = group.abs_change_difference.to_numpy()
            axes[index].scatter([x] * len(values), values, color=color, alpha=0.25, s=18)
            axes[index].plot(x, group.abs_change_difference.median(), "_", color="black", markersize=22)
        axes[index].axhline(0, color="gray", linestyle="--")
        axes[index].set(xticks=[5, 10, 20], xlabel="Occluded eligible pairs (%)",
                        ylabel="Top attribution minus control |logit change|",
                        title=control)
    figure.suptitle("One dot per structure after within-structure seed median; not biological significance")
    figure.savefig(output / "occlusion_controls.png", dpi=170)
    plt.close(figure)
    print(json.dumps({"test_structures": summary["test_structures"],
                      "perturbation_rows": summary["perturbation_rows"],
                      "skipped_maps": len(summary["skipped_maps"]),
                      "all_structure_results": [item for item in summary["aggregates"]
                                                if item["class_group"] == "all"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
