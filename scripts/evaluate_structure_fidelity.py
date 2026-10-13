"""Evaluate saved reconstructions' fixed morphology and position surrogates."""

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
from matplotlib.patches import Rectangle

from microc_foundation.structure_fidelity import (
    annotation_shape_r,
    compare_profiles,
    distance_residual,
    excess_profile,
)
from microc_foundation.super_resolution_evaluation import _load_reconstruction

METRICS = ("annotation_shape_r", "profile_correlation", "peak_displacement_bp", "top_profile_jaccard")


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        parser.error("use a fresh output directory; prior evidence cannot be overwritten")
    files = [Path(args.reconstruction_dir) / f"seed_{seed}/test_reconstruction.npz" for seed in (2026, 2027, 2028)]
    data = [_load_reconstruction(path) for path in files]
    ref = data[0]
    if ref["target"].shape != (41, 2, 256, 256) or len(set(ref["structure_id"].astype(str))) != 41:
        raise ValueError("frozen protocol requires 41 unique two-channel 256x256 test structures")
    for item in data:
        for field in ("structure_id", "label", "chrom", "window_start", "window_end",
                      "annotation_start", "annotation_end", "replicate", "target", "bicubic"):
            if not np.array_equal(item[field], ref[field]):
                raise ValueError(f"seed inputs disagree: {field}")
        if not np.all(item["window_end"] - item["window_start"] == 20_480):
            raise ValueError("unexpected window span")
    cases = [int(np.flatnonzero(ref["label"].astype(str) == label)[0]) for label in ("CHIN", "OPCID", "CHID")]
    if ref["structure_id"][cases].astype(str).tolist() != ["CHIN_221", "OPCID_62", "CHID_23"]:
        raise ValueError("fixed report cases changed")
    rows, profiles = [], []
    for seed, item in zip((2026, 2027, 2028), data):
        seed_profiles = {name: np.empty((41, 2, 256)) for name in ("target", "bicubic", "cnn")}
        for index, sid in enumerate(ref["structure_id"].astype(str)):
            start, ann_start, ann_end = (int(ref[name][index]) for name in
                                         ("window_start", "annotation_start", "annotation_end"))
            for rep in range(2):
                target = distance_residual(item["target"][index, rep])
                profile = excess_profile(target)
                seed_profiles["target"][index, rep] = profile
                target_peak = start + (int(profile.argmax()) + .5) * 80 if profile.std() >= 1e-12 else float("nan")
                for method in ("bicubic", "cnn"):
                    residual = distance_residual(item[method][index, rep])
                    predicted_profile = excess_profile(residual)
                    seed_profiles[method][index, rep] = predicted_profile
                    shape_r, n = annotation_shape_r(target, residual, start_bp=start,
                                                   annotation_start=ann_start, annotation_end=ann_end)
                    metrics = compare_profiles(profile, predicted_profile)
                    pred_peak = (start + (int(predicted_profile.argmax()) + .5) * 80
                                 if np.isfinite(metrics["profile_correlation"]) else float("nan"))
                    rows.append({"seed": seed, "structure_id": sid, "label": str(ref["label"][index]),
                                 "replicate": rep + 1, "method": method, "annotation_shape_r": shape_r,
                                 "annotation_pixel_pairs": n, **metrics, "target_peak_bp": target_peak,
                                 "predicted_peak_bp": pred_peak,
                                 "target_peak_inside_annotation": (ann_start <= target_peak < ann_end) if np.isfinite(target_peak) else float("nan"),
                                 "prediction_peak_inside_annotation": (ann_start <= pred_peak < ann_end) if np.isfinite(pred_peak) else float("nan")})
        profiles.append(seed_profiles)
    frame = pd.DataFrame(rows)
    per_structure = frame.groupby(["structure_id", "label", "method"], sort=False)[list(METRICS)].mean().reset_index()
    summaries = []
    for (label, method), group in per_structure.groupby(["label", "method"], sort=False):
        record = {"label": label, "method": method, "structures": len(group)}
        for metric in METRICS:
            valid = group[metric].dropna()
            record.update({f"{metric}_n": len(valid), f"{metric}_mean": float(valid.mean()),
                           f"{metric}_median": float(valid.median())})
        summaries.append(record)
    paired = per_structure.pivot(index=["structure_id", "label"], columns="method", values=list(METRICS))
    improvements = []
    for metric in METRICS:
        sign = -1 if metric == "peak_displacement_bp" else 1
        delta = sign * (paired[(metric, "cnn")] - paired[(metric, "bicubic")])
        valid = delta.dropna()
        improvements.append({"metric": metric, "paired_n": len(valid), "mean_improvement": float(valid.mean()),
                             "median_improvement": float(valid.median()), "improved": int((valid > 1e-12).sum()),
                             "tied": int((abs(valid) <= 1e-12).sum()), "worse": int((valid < -1e-12).sum())})
    localization = frame.loc[frame.seed == 2026].groupby(["label", "replicate", "method"], sort=False)[
        ["target_peak_inside_annotation", "prediction_peak_inside_annotation"]].mean().reset_index()
    summary = {"protocol": "task5-known-shape-position-v1", "trained": False,
               "structures": 41, "seeds": [2026, 2027, 2028], "bin_bp": 80,
               "minimum_distance_bp": 800, "maximum_profile_distance_bp": 6400,
               "smoothing": [.25, .5, .25], "top_profile_fraction": .1,
               "aggregation": "mean per-replicate/seed metrics within each structure ID; 41 paired structures",
               "class_summaries": summaries, "paired_improvements": improvements,
               "peak_annotation_checks_seed2026": localization.to_dict("records"),
               "fixed_cases": ref["structure_id"][cases].astype(str).tolist(),
               "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
               "limitations": ["morphology/profile localization surrogates, not trained independent structure detectors",
                               "dominant profile peak can lie outside the known annotation; this coverage is reported",
                               "annotation square locates interval intersection, not biological morphology outline",
                               "fixed test split; shared/overlapping windows and two channels are not independent biological samples",
                               "undefined correlations and peak locations are retained as missing, not successes",
                               "no retraining, parameter search, or replacement of strict-history divergence"]}
    output.mkdir(parents=True)
    frame.to_csv(output / "per_channel_metrics.csv", index=False)
    per_structure.to_csv(output / "structure_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "class_summary.csv", index=False)
    pd.DataFrame(improvements).to_csv(output / "paired_improvements.csv", index=False)
    for seed, arrays in zip((2026, 2027, 2028), profiles):
        np.savez_compressed(output / f"profiles_seed_{seed}.npz", **arrays,
                            structure_id=ref["structure_id"], window_start=ref["window_start"])
    (output / "summary.json").write_text(json.dumps(finite_json(summary), indent=2, allow_nan=False), encoding="utf-8")
    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5), constrained_layout=True)
    for row, index in enumerate(cases):
        start, end, lo, hi = (int(ref[key][index]) for key in ("window_start", "window_end", "annotation_start", "annotation_end"))
        vmax = max(float(np.percentile(ref["target"][index, 0], 99.5)), 1e-6)
        for column, method in enumerate(("target", "bicubic", "cnn")):
            ax = axes[row, column]
            shown = ax.imshow(ref[method][index, 0], origin="lower", cmap="magma", vmin=0, vmax=vmax,
                              extent=(start / 1e6, end / 1e6, start / 1e6, end / 1e6))
            ax.add_patch(Rectangle((lo / 1e6, lo / 1e6), (hi-lo) / 1e6, (hi-lo) / 1e6,
                                   fill=False, edgecolor="#4cc9f0", linewidth=1.1))
            ax.set(title=f"{ref['structure_id'][index]} / {method}", xlabel="Position (Mb)", ylabel="Position (Mb)")
            ax.ticklabel_format(useOffset=False, style="plain")
            ax.tick_params(labelsize=8)
        fig.colorbar(shown, ax=list(axes[row, :3]), shrink=.65, label="log1p counts; shared row scale")
        ax = axes[row, 3]
        positions = (start + (np.arange(256) + .5) * 80) / 1e6
        for method, color in (("target", "black"), ("bicubic", "#8b8b8b"), ("cnn", "#1769aa")):
            ax.plot(positions, profiles[0][method][index, 0], color=color, label=method)
        ax.axvspan(lo / 1e6, hi / 1e6, color="#bbbbbb", alpha=.2)
        ax.set(title="Excess-contact profile", xlabel="Position (Mb)", ylabel="Positive distance residual")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, frameon=False)
    fig.savefig(output / "known_structure_fidelity.png", dpi=180)
    plt.close(fig)
    print(json.dumps(finite_json({"class_summaries": summaries, "paired_improvements": improvements}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
