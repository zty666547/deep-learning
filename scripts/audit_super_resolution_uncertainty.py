"""Fixed-test paired uncertainty and off-diagonal residual fidelity; no retraining."""

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

from microc_foundation.paired_uncertainty import (
    genomic_groups,
    log_distance_residual_r,
    paired_group_bootstrap,
)
from microc_foundation.super_resolution import local_ssim, psnr
from microc_foundation.super_resolution_evaluation import _load_reconstruction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        parser.error("use a fresh output directory; never overwrite prior evidence")
    files = sorted(Path(args.reconstruction_dir).glob("seed_*/test_reconstruction.npz"))
    if {path.parent.name for path in files} != {"seed_2026", "seed_2027", "seed_2028"}:
        parser.error("frozen protocol requires three seed files")
    loaded = [_load_reconstruction(path) for path in files]
    ref = loaded[0]
    ids = ref["structure_id"].astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate structure IDs")
    for data in loaded:
        for key in ("structure_id", "label", "chrom", "window_start", "window_end",
                    "annotation_start", "annotation_end", "replicate", "target", "bicubic"):
            if not np.array_equal(ref[key], data[key]):
                raise ValueError(f"seed files disagree: {key}")
        if data["target"].shape[2:] != (256, 256):
            raise ValueError("frozen protocol requires 80bp 256x256 maps")
        if not np.all(data["window_end"] - data["window_start"] == 20_480):
            raise ValueError("invalid window span")
    meta = pd.DataFrame({k: ref[k] for k in ("structure_id", "label", "chrom", "window_start", "window_end")})
    all_metrics, hashes = [], {}
    for path, data in zip(files, loaded):
        metric_path = path.parent / "per_sample_metrics.csv"
        metrics = pd.read_csv(metric_path)
        expected_keys = {(sid, r, m) for sid in ids for r in (1, 2) for m in ("bicubic", "cnn")}
        actual_keys = list(zip(metrics.structure_id, metrics.replicate_index, metrics.method))
        if len(actual_keys) != len(expected_keys) or set(actual_keys) != expected_keys:
            raise ValueError("metric rows must match ordered reconstruction IDs/replicates/methods")
        if not np.isfinite(metrics[["psnr", "ssim", "mse"]].to_numpy()).all():
            raise ValueError("nonfinite saved metrics")
        for index, sid in enumerate(ids):
            for rep in range(2):
                for method in ("bicubic", "cnn"):
                    row = metrics.loc[(metrics.structure_id == sid) & (metrics.replicate_index == rep + 1)
                                      & (metrics.method == method)].iloc[0]
                    truth, prediction = data["target"][index, rep], data[method][index, rep]
                    recomputed = {"mse": float(np.mean((truth - prediction) ** 2)),
                                  "psnr": psnr(truth, prediction), "ssim": local_ssim(truth, prediction)}
                    for metric, value in recomputed.items():
                        if not np.isclose(value, row[metric], rtol=1e-12, atol=1e-12):
                            raise ValueError(f"saved metric {metric} disagrees with reconstruction")
                    r = log_distance_residual_r(truth, prediction)
                    all_metrics.append({"structure_id": sid, "label": meta.label.iloc[index],
                                        "seed": path.parent.name, "replicate": rep + 1, "method": method,
                                        "psnr": row.psnr, "ssim": row.ssim, "mse": row.mse,
                                        "distance_residual_r": r})
        for p in (path, metric_path):
            hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    full = pd.DataFrame(all_metrics)
    means = full.groupby(["structure_id", "method"], sort=False)[["psnr", "ssim", "mse", "distance_residual_r"]].mean()
    valid_counts = full.groupby(["structure_id", "method"], sort=False)["distance_residual_r"].count()
    paired = meta.copy()
    for metric, direction in (("psnr", 1), ("ssim", 1), ("mse", -1), ("distance_residual_r", 1)):
        for method in ("bicubic", "cnn"):
            paired[f"{method}_{metric}"] = means.xs(method, level="method").loc[ids, metric].to_numpy()
            if metric == "distance_residual_r":
                paired[f"{method}_residual_valid_observations"] = valid_counts.xs(method, level="method").loc[ids].to_numpy()
        paired[f"delta_{metric}"] = direction * (paired[f"cnn_{metric}"] - paired[f"bicubic_{metric}"])
    schemes, group_rows = {}, []
    for block_bp, guard_bp in ((1, 0), (1, 20_480), (100_000, 0)):
        name = f"block{block_bp}_guard{guard_bp}"
        groups = genomic_groups(meta, block_bp=block_bp, guard_bp=guard_bp)
        paired[name] = groups
        schemes[name] = {}
        for metric in ("psnr", "ssim", "mse", "distance_residual_r"):
            values = paired[f"delta_{metric}"].to_numpy(float)
            finite = np.isfinite(values)
            if not finite.any():
                schemes[name][metric] = {"status": "undefined_constant_residuals", "structure_count": 0}
            else:
                schemes[name][metric] = paired_group_bootstrap(
                    values[finite], groups[finite], draws=args.draws, seed=args.seed)
        for group in np.unique(groups):
            subset = paired.loc[groups == group]
            group_rows.append({"scheme": name, "group": int(group), "n": len(subset),
                               "chrom": subset.chrom.iloc[0], "start": int(subset.window_start.min()),
                               "end": int(subset.window_end.max()), "structure_ids": ";".join(subset.structure_id)})
    summary = {"protocol": "task5-paired-genomic-uncertainty-v1", "test_structures": len(ids),
               "seed_count": len(files), "replicate_count": 2, "schemes": schemes,
               "residual_metric": "per-channel distance-centered Pearson r on log1p maps, exclude <800bp and outer diagonals <3 pixels",
               "aggregation": "mean per-channel per-seed metric within structure ID; positive delta always favors CNN",
               "input_sha256": hashes,
               "limitations": ["Fixed spatial test split; group bootstrap is conditional and not independent biological validation",
                               "100kb blocks merge transitively when any interval crosses their boundary",
                               "Overlap-connected/20480bp-expanded components are prespecified grouping sensitivities, not tuned for favorable CI",
                               "Few genomic groups; percentile intervals can be unstable and long-range dependence remains",
                               "No p-value, no test-set tuning, no retraining, no replacement of strict-history diverged status",
                               "Low resolution is derived from target; residual r is image fidelity, not novel structure or 3D proof"]}
    out.mkdir(parents=True)
    full.to_csv(out / "per_channel_metrics.csv", index=False)
    paired.to_csv(out / "paired_structure_metrics.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out / "genomic_groups.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric, title in zip(axes, ("psnr", "distance_residual_r"), ("Paired PSNR gain (dB)", "Paired distance-residual r gain")):
        for i, (name, result) in enumerate(schemes.items()):
            item = result[metric]
            ci = item.get("interval_95")
            if ci is not None:
                ax.plot([i, i], ci, color="#1769aa", linewidth=3)
                ax.plot(i, item["mean_delta"], "o", color="#1769aa")
        ax.axhline(0, linestyle="--", color="gray")
        labels = [f"{label}\ngroups={schemes[name][metric].get('group_count', 0)}"
                  for label, name in zip(("Overlap", "Overlap + guard", "Merged 100kb"), schemes)]
        ax.set(xticks=range(3), xticklabels=labels, title=title)
        ax.tick_params(axis="x", rotation=15)
    fig.savefig(out / "paired_uncertainty.png", dpi=180)
    plt.close(fig)
    print(json.dumps(schemes, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
