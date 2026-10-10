"""Audit all test structures across trained seeds and random-parameter controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

from microc_foundation.explainability import normalize_saliency
from microc_foundation.saliency_audit import audit_checkpoint_saliency


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-offset", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = audit_checkpoint_saliency(
        args.dataset, args.checkpoint, args.output_dir,
        minimum_offset=args.minimum_offset, batch_size=args.batch_size,
    )
    output = Path(args.output_dir)
    rows = pd.read_csv(output / "per_sample_attribution.csv")
    with np.load(output / "attribution_maps.npz", allow_pickle=False) as data:
        ids = data["structure_id"].astype(str)
        examples = summary["example_structure_ids"]
        figure, axes = plt.subplots(len(examples), 5, figsize=(17, 3.6 * len(examples)),
                                    squeeze=False, constrained_layout=True)
        for row, structure_id in enumerate(examples):
            index = int(np.flatnonzero(ids == structure_id)[0])
            start, end = int(data["window_start"][index]), int(data["window_end"][index])
            extent = (start / 1e6, end / 1e6, start / 1e6, end / 1e6)
            maps = [data["contact_maps"][index]] + list(data["trained"][:, index]) + [
                data["random"][0, index]
            ]
            for column, values in enumerate(maps):
                if column == 0:
                    axes[row, column].imshow(values, origin="lower", extent=extent, cmap="Reds",
                                             vmin=0, vmax=float(np.percentile(values, 99)))
                    title = "Mean log1p contacts"
                else:
                    axes[row, column].imshow(normalize_saliency(values), origin="lower",
                                             extent=extent, cmap="magma", vmin=0, vmax=1)
                    if column == 4:
                        title = "Random network / control 9101"
                    else:
                        record = rows.loc[(rows.structure_id == structure_id) &
                                          (rows.model_index == column - 1)].iloc[0]
                        title = f"Model {column - 1}: pred={record.predicted_class}"
                left, right = data["annotation_start"][index] / 1e6, data["annotation_end"][index] / 1e6
                axes[row, column].add_patch(Rectangle((left, left), right - left, right - left,
                                                       fill=False, edgecolor="#43c7dd", linewidth=1.2))
                axes[row, column].set_title(f"{structure_id}\n{title}", fontsize=9)
                axes[row, column].set(xlabel="Position (Mb)", ylabel="Position (Mb)")
        figure.suptitle("True-class attribution; display scales per map; cyan = annotation interval")
        figure.savefig(output / "fixed_examples.png", dpi=150)
        plt.close(figure)
    print(json.dumps({"test_structures": summary["test_structures"],
                      "comparisons": summary["comparisons"],
                      "examples": summary["example_structure_ids"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
