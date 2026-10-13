"""Freeze a WT-rep1 diagnostic partition away from five existing candidates."""

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

from microc_foundation.cluster_transfer import (
    SINGLE_FEATURES,
    assign_partition,
    candidate_guard_mask,
    circular_shift_control,
    fit_partition,
    library_cpm_matrix,
    nonoverlapping_grid,
    single_replicate_features,
)
from microc_foundation.discovery import extract_contact_features
from microc_foundation.io import open_cooler


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--cool", action="append", required=True, help="name=160bp.cool; WT1 first")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--guard-bp", type=int, default=20_480)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--feature-normalization", choices=("raw_counts", "library_cpm"), default="raw_counts")
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        parser.error("use a fresh output directory; never overwrite prior evidence")
    paths = dict(item.split("=", 1) for item in args.cool)
    if len(paths) != len(args.cool) or len(paths) != 6 or list(paths) != [
            "WT1", "WT2", "dstpa1", "dstpa2", "double1", "double2"]:
        parser.error("provide six ordered unique sources: WT1 WT2 dstpa1 dstpa2 double1 double2")
    if len({Path(p).resolve() for p in paths.values()}) != 6:
        parser.error("source files must be distinct")
    windows = pd.read_csv(args.windows)
    candidates = pd.read_csv(args.candidates)
    if candidates.candidate_region_id.duplicated().any():
        raise ValueError("duplicate candidate IDs")
    grid = nonoverlapping_grid(windows.loc[windows.quality_pass.eq(True)])
    reps = windows.loc[windows.window_id.isin(candidates.representative_window_id)]
    if len(reps) != len(candidates) or not reps.quality_pass.eq(True).all():
        raise ValueError("candidate representatives must be unique and QC-passed")
    selected = pd.concat([grid, reps]).drop_duplicates("window_id").sort_values(
        ["chrom", "start", "window_id"], kind="stable").reset_index(drop=True)
    features, valid, library_sums = {}, {}, {}
    with ExitStack() as stack:
        coolers = {name: stack.enter_context(open_cooler(path)) for name, path in paths.items()}
        for name, cool in coolers.items():
            if cool.binsize != 160:
                raise ValueError("protocol requires 160bp bins")
            library_sum = float(cool.info["sum"])
            if not np.isfinite(library_sum) or library_sum <= 0:
                raise ValueError("invalid full upper-triangle library sum")
            library_sums[name] = library_sum
            records, qc = [], []
            for row in selected.itertuples(index=False):
                if row.start % 160 or row.end % 160 or row.end - row.start != 20_480:
                    raise ValueError("requires fixed 20480bp aligned windows")
                matrix = cool.matrix(balance=False).fetch(f"{row.chrom}:{row.start}-{row.end}")
                if matrix.shape != (128, 128):
                    raise ValueError("truncated window")
                f = extract_contact_features(matrix)
                records.append(f if args.feature_normalization == "raw_counts" else
                               extract_contact_features(library_cpm_matrix(
                                   np.where(np.isfinite(matrix), matrix, 0.), library_sum)))
                qc.append(f["finite_fraction"] >= .99 and f["total_contacts"] >= 1
                          and f["nonzero_fraction"] >= .01 and f["zero_bin_fraction"] <= .03
                          and f["largest_zero_run_fraction"] <= .03)
            features[name] = single_replicate_features(pd.DataFrame(records))
            valid[name] = np.array(qc, dtype=bool)
    train_mask = (selected.window_id.isin(grid.window_id).to_numpy() & valid["WT1"]
                  & ~candidate_guard_mask(selected, candidates, args.guard_bp))
    model = fit_partition(features["WT1"][train_mask], seed=args.seed)
    table = selected[["window_id", "chrom", "start", "end", "known_overlap_bp"]].copy()
    table["fit_WT1"] = train_mask
    table["nonoverlapping_grid"] = table.window_id.isin(grid.window_id)
    table["candidate_region_id"] = table.window_id.map(
        candidates.set_index("representative_window_id").candidate_region_id).fillna("")
    for name in paths:
        table[f"{name}_cluster"], table[f"{name}_margin"] = assign_partition(features[name], model)
        table[f"{name}_quality_pass"] = valid[name]
    agreements = {}
    for first, second in (("WT1", "WT2"), ("dstpa1", "dstpa2"), ("double1", "double2")):
        mask = table.nonoverlapping_grid.to_numpy() & valid[first] & valid[second]
        agreements[f"{first}:{second}"] = circular_shift_control(
            table.loc[mask, f"{first}_cluster"].to_numpy(),
            table.loc[mask, f"{second}_cluster"].to_numpy())
    candidates_out = table.loc[table.candidate_region_id.ne("")].copy()
    for first, second in (("WT1", "WT2"), ("dstpa1", "dstpa2"), ("double1", "double2")):
        candidates_out[f"{first}:{second}_eligible"] = (candidates_out[f"{first}_quality_pass"]
                                                       & candidates_out[f"{second}_quality_pass"])
        candidates_out[f"{first}:{second}_same_cluster"] = (candidates_out[f"{first}_cluster"]
                                                          == candidates_out[f"{second}_cluster"])
    summary = {"protocol": "task2-candidate-excluded-cluster-transfer-v1", "seed": args.seed,
               "fit_source": "WT1", "fit_windows": int(train_mask.sum()),
               "nonoverlapping_grid_windows": len(grid), "evaluated_windows": len(table),
               "guard_bp": args.guard_bp, "features": list(SINGLE_FEATURES),
               "components": 3, "clusters": 2, "variance_ratios": model["variance_ratios"].tolist(),
               "agreements": agreements,
               "input_sha256": {name: sha256_file(path)
                                  for name, path in {"windows": args.windows, "candidates": args.candidates,
                                                     **paths}.items()},
               "limitations": ["Separate seven-feature diagnostic partition, not original pooled eight-feature cluster labels",
                               "WT2 contributed to original candidate selection; not an independent novelty holdout",
                               "Mutants never fit the partition; their reproducibility cannot prove new biological classes",
                               "Candidate windows plus 20480bp guard excluded from fit; k=2/PC=3 fixed before evaluation",
                               "Nonoverlapping windows still spatially correlated; cyclic controls are descriptive, not p-values",
                               "Features include intensity and decay; stable assignment is not morphology or mechanism proof",
                               "Quality eligibility fixed independently per query; failed candidate QC is never hidden"]}
    if args.feature_normalization == "library_cpm":
        summary.update(protocol="task2-candidate-excluded-cluster-transfer-library-cpm-v1",
                       feature_normalization="count * 1e6 / full upper-triangle library sum",
                       library_sums=library_sums,
                       sensitivity_origin="precommitted extension after observing raw-count transfer weakness; not blind model selection")
        summary["limitations"].append("Library scaling does not correct local coverage, distance bias or establish biological novelty")
    out.mkdir(parents=True)
    table.to_csv(out / "assignments.csv", index=False)
    candidates_out.to_csv(out / "candidate_assignments.csv", index=False)
    np.savez(out / "frozen_partition.npz", **model, training_window_ids=table.loc[train_mask, "window_id"].to_numpy(str))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for index, (name, result) in enumerate(agreements.items()):
        low, median, high = result["shift_ari_quantiles"]
        ax.plot([index, index], [low, high], color="gray", linewidth=5, alpha=.5)
        ax.plot(index, median, "_", color="gray", markersize=14)
        ax.plot(index, result["adjusted_rand_index"], "o", color="#1769aa")
    ax.set(xticks=range(3), xticklabels=list(agreements), ylabel="Adjusted Rand index",
           title="Frozen WT1 diagnostic partition; gray = cyclic-shift 95% range", ylim=(-.1, 1.05))
    if args.feature_normalization == "library_cpm":
        ax.set_title("Frozen WT1 partition (library CPM)\nGray = cyclic-shift ARI 95% range")
    fig.savefig(out / "cluster_transfer.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"fit_windows": int(train_mask.sum()), "agreements": agreements}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
