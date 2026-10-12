"""Export frozen Task 2 members/centers and nearest-window contact examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from microc_foundation.cluster_evidence import build_cluster_evidence
from microc_foundation.io import open_cooler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--cluster-summary", required=True)
    parser.add_argument("--replicate", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if len(args.replicate) != 2:
        parser.error("exactly two replicates required")
    summary = json.loads(Path(args.cluster_summary).read_text(encoding="utf-8"))
    members, profiles, centers, supports, audit = build_cluster_evidence(
        pd.read_csv(args.windows), pd.read_csv(args.candidates), summary,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in [("cluster_members", members), ("cluster_feature_profiles", profiles),
                         ("cluster_centers", centers), ("candidate_support_clusters", supports)]:
        table.to_csv(output / f"{name}.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    labels = centers.cluster.to_numpy(int)
    axes[0].bar(labels, centers.known_overlap_members, color="#809bbb", label="Known overlap")
    axes[0].bar(labels, centers.unannotated_members, bottom=centers.known_overlap_members,
                color="#d8d8d8", label="Unannotated")
    axes[0].set(xticks=labels, xlabel="Frozen global cluster", ylabel="QC-passed overlapping windows",
                title="Both clusters contain known annotations")
    axes[0].legend(fontsize=9)
    feature_order = summary["feature_names"]
    values = profiles.pivot(index="cluster", columns="feature", values="scaled_median")[feature_order]
    limit = max(float(np.abs(values.to_numpy()).max()), 1e-6)
    plotted = axes[1].imshow(values.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axes[1].set_xticks(range(len(feature_order)), feature_order, rotation=45, ha="right", fontsize=8)
    axes[1].set(yticks=range(len(centers)), yticklabels=labels, ylabel="Frozen global cluster",
                title="Feature medians after original global median/IQR scale")
    figure.colorbar(plotted, ax=axes[1], shrink=0.8)
    figure.savefig(output / "cluster_composition_profiles.png", dpi=170)
    plt.close(figure)
    contact_maps, bin_size = [], None
    with ExitStack() as stack:
        cools = [stack.enter_context(open_cooler(path)) for path in args.replicate]
        if cools[0].binsize is None or cools[0].binsize != cools[1].binsize or not cools[0].chromsizes.equals(cools[1].chromsizes):
            raise ValueError("replicates must match in bin size and chromosome coordinates")
        bin_size = int(cools[0].binsize)
        for center in centers.itertuples(index=False):
            if center.nearest_start % bin_size or center.nearest_end % bin_size:
                raise ValueError("nearest windows must align to matrix bins")
            region = (str(center.nearest_chrom), int(center.nearest_start), int(center.nearest_end))
            maps = [np.asarray(cool.matrix(balance=False).fetch(region), dtype=float) for cool in cools]
            size = (region[2] - region[1]) // bin_size
            if any(matrix.shape != (size, size) or not np.isfinite(matrix).all() or (matrix < 0).any()
                   for matrix in maps):
                raise ValueError("invalid extracted nearest-window matrices")
            contact_maps.append(np.stack(maps))
    contact_maps = np.stack(contact_maps)
    np.savez_compressed(output / "nearest_centroid_windows.npz", counts=contact_maps,
                        window_id=centers.nearest_window_id.to_numpy(str),
                        cluster=centers.cluster.to_numpy(int), bin_size=bin_size)
    figure, axes = plt.subplots(len(centers), 2, figsize=(9, 4 * len(centers)),
                                squeeze=False, constrained_layout=True)
    for index, center in enumerate(centers.itertuples(index=False)):
        maps = np.log1p(contact_maps[index])
        vmax = float(np.percentile(maps, 99))
        extent = (center.nearest_start / 1e6, center.nearest_end / 1e6,
                  center.nearest_start / 1e6, center.nearest_end / 1e6)
        for repeat in range(2):
            plotted = axes[index, repeat].imshow(maps[repeat], origin="lower", extent=extent,
                                                 cmap="Reds", vmin=0, vmax=vmax)
            axes[index, repeat].set_title(f"Cluster {center.cluster} nearest actual window\n"
                                          f"{center.nearest_window_id}; replicate {repeat + 1}")
            axes[index, repeat].set(xlabel="Position (Mb)", ylabel="Position (Mb)")
            figure.colorbar(plotted, ax=axes[index, repeat], shrink=0.8)
    figure.suptitle("Fixed nearest-centroid examples, not mean structures or biological classes")
    figure.savefig(output / "nearest_centroid_heatmaps.png", dpi=160)
    plt.close(figure)
    audit["input_sha256"] = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in [args.windows, args.candidates, args.cluster_summary] + args.replicate
    }
    audit["matrix_bin_size"] = bin_size
    audit["nearest_windows"] = centers.nearest_window_id.tolist()
    (output / "summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
