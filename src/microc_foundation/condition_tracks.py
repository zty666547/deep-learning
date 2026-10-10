"""Depth-normalized circular-genome contact tracks and fixed four-panel tiles."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import open_cooler


def genome_tiles(length: int, tile_bp: int = 10_000) -> list[tuple[int, int]]:
    if length <= 0 or tile_bp <= 0:
        raise ValueError("length and tile size must be positive")
    return [(start, min(start + tile_bp, length)) for start in range(0, length, tile_bp)]


def circular_contact_track(contact_map, *, band_bp: int = 10_000,
                           chunk_bins: int = 512) -> pd.DataFrame:
    """Sum off-diagonal contacts within circular bin-center distance <= band_bp.

    CPM denominator is the library's upper-triangle count sum (including its
    diagonal), not the track total. Each off-diagonal contact contributes to
    both endpoint bins. No ICE, O/E, expression or differential-test claim.
    """
    if band_bp <= 0 or chunk_bins <= 0:
        raise ValueError("band and chunk size must be positive")
    if len(contact_map.chromnames) != 1 or contact_map.storage_mode != "symmetric-upper":
        raise ValueError("one symmetric-upper chromosome required")
    chrom = contact_map.chromnames[0]
    length = int(contact_map.chromsizes[chrom])
    resolution = contact_map.binsize
    if resolution is None or band_bp >= length / 2:
        raise ValueError("fixed bins and band smaller than half the chromosome required")
    bins = contact_map.bins()[:][["chrom", "start", "end"]].copy()
    starts, ends = bins.start.to_numpy(), bins.end.to_numpy()
    n = len(bins)
    if not np.array_equal(starts, np.arange(n) * resolution) or not np.array_equal(
            ends, np.minimum(starts + resolution, length)):
        raise ValueError("non-contiguous or irregular bins")
    centers = (starts + ends) / 2
    total = float(contact_map.info["sum"])
    if not np.isfinite(total) or total <= 0:
        raise ValueError("positive finite library sum required")
    radius = int(np.ceil(band_bp / resolution)) + 1
    counts, support = np.zeros(n), np.zeros(n, dtype=int)
    selector = contact_map.matrix(balance=False)
    for begin in range(0, n, chunk_bins):
        stop = min(n, begin + chunk_bins)
        columns = np.unique(np.arange(begin - radius, stop + radius) % n)
        groups = np.split(columns, np.flatnonzero(np.diff(columns) != 1) + 1)
        row_indices = np.arange(begin, stop)
        for group in groups:
            left, right = int(group[0]), int(group[-1]) + 1
            values = np.asarray(selector[begin:stop, left:right], dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("contacts must be finite and nonnegative")
            distances = np.abs(centers[begin:stop, None] - centers[None, left:right])
            distances = np.minimum(distances, length - distances)
            eligible = ((distances <= band_bp) &
                        (row_indices[:, None] != np.arange(left, right)[None, :]))
            counts[begin:stop] += np.where(eligible, values, 0).sum(axis=1)
            support[begin:stop] += eligible.sum(axis=1)
    bins["center"] = centers
    bins["band_contacts"] = counts
    bins["possible_partner_bins"] = support
    bins["contact_cpm"] = counts * 1_000_000 / total
    return bins


def build_condition_tracks(conditions: dict[str, list[str]], *, band_bp: int = 10_000
                           ) -> tuple[pd.DataFrame, dict]:
    """Require three real conditions, with independent file paths and >=2 replicates."""
    if len(conditions) < 3 or any(len(paths) < 2 for paths in conditions.values()):
        raise ValueError("at least three conditions and two replicates each required")
    paths = [str(Path(path).resolve()) for group in conditions.values() for path in group]
    if len(paths) != len(set(paths)):
        raise ValueError("a replicate file cannot count as multiple conditions")
    result, sources = None, []
    for condition, group in conditions.items():
        if not condition.strip() or condition in ("chrom", "start", "end", "center", "mean_cpm"):
            raise ValueError("invalid condition name")
        arrays = []
        for path in group:
            with open_cooler(path) as contact_map:
                track = circular_contact_track(contact_map, band_bp=band_bp)
                source = {"condition": condition, "path": path,
                          "resolution_bp": int(contact_map.binsize),
                          "library_sum": int(contact_map.info["sum"])}
            coords = track[["chrom", "start", "end", "center"]]
            if result is None:
                result = coords.copy()
            if not coords.equals(result[["chrom", "start", "end", "center"]]):
                raise ValueError("all conditions must have identical bin coordinates")
            arrays.append(track.contact_cpm.to_numpy())
            sources.append(source)
        result[condition] = np.mean(arrays, axis=0)
        result[f"{condition}__replicate_sd"] = np.std(arrays, axis=0, ddof=1)
    result["mean_cpm"] = result[list(conditions)].mean(axis=1)
    return result, {"sources": sources, "conditions": list(conditions), "band_bp": band_bp,
                    "circular": True, "replicate_weighting": "equal within condition",
                    "condition_weighting": "equal across conditions",
                    "normalization": "off-diagonal band row sum / library sum * 1e6"}


def _draw_intervals(axis, frame: pd.DataFrame, start: int, end: int, *,
                    color: str, label_column: str, pad_bp: int = 0) -> int:
    subset = frame.loc[(frame.start < end) & (frame.end > start)].sort_values("start")
    lane_ends = []
    for row in subset.itertuples(index=False):
        left, right = max(int(row.start), start), min(int(row.end), end)
        lane = next((i for i, value in enumerate(lane_ends) if value <= left), len(lane_ends))
        if lane == len(lane_ends):
            lane_ends.append(0)
        lane_ends[lane] = right + pad_bp
        axis.broken_barh([(left, right - left)], (lane - .25, .5), facecolors=color)
        label = str(getattr(row, label_column))
        axis.text((left + right) / 2, lane + .31, label, ha="center", va="bottom",
                  fontsize=6, clip_on=True)
    axis.set_ylim(-.5, max(1, len(lane_ends)) - .15)
    axis.set_yticks([])
    return len(subset)


def render_condition_tiles(tracks: pd.DataFrame, genes: pd.DataFrame,
                           structures: pd.DataFrame, metadata: dict, output_dir: str, *,
                           tile_bp: int = 10_000) -> Path:
    """Render every fixed interval, clipping the final tile to chromosome length."""
    names = metadata["conditions"]
    if len(names) < 3 or not np.isfinite(tracks[[*names, "mean_cpm"]]).all().all():
        raise ValueError("three conditions with finite signals required")
    chroms = tracks.chrom.astype(str).unique()
    if len(chroms) != 1 or tracks.start.iloc[0] != 0:
        raise ValueError("tracks must start at zero on one chromosome")
    if not np.array_equal(tracks.start.to_numpy()[1:], tracks.end.to_numpy()[:-1]):
        raise ValueError("tracks must be ordered and contiguous")
    chrom = chroms[0]
    length = int(tracks.end.iloc[-1])
    if set(genes.chrom.astype(str)) != {chrom}:
        raise ValueError("gene reference mismatch")
    genes = genes.loc[genes.chrom.astype(str) == chrom]
    structures = structures.loc[(structures.chrom.astype(str) == chrom) &
                                structures.structure_type.isin(["CHIN", "OPCID"])]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    colors = ["#168aad", "#9b5de5", "#d1495b", "#008000", "#a06100"]
    ymax = max(float(tracks[names].to_numpy().max()) * 1.05, 1.0)
    mean_max = max(float(tracks.mean_cpm.max()) * 1.05, 1.0)
    manifest = []
    for start, end in genome_tiles(length, tile_bp):
        # Include intersecting bins; curves use centers and are clipped by xlim.
        local = tracks.loc[(tracks.start < end) & (tracks.end > start)]
        figure, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1, 1.6, 1.4]})
        x = local.center.to_numpy()
        mean = local.mean_cpm.to_numpy()
        axes[0].fill_between(x, 0, mean, color="gray", alpha=.2, label="Across-condition mean")
        for index, name in enumerate(names):
            axes[0].plot(x, local[name], color=colors[index % len(colors)], label=name,
                         linewidth=1)
        axes[0].set_ylim(0, ymax)
        axes[0].set_ylabel("Condition mean\ncontact CPM")
        axes[0].legend(loc="upper right", fontsize=7, ncol=2)
        axes[0].set_title(f"{chrom}: {start:,}–{end:,} bp | circular ±{metadata['band_bp']:,} bp")
        axes[1].fill_between(x, 0, mean, color="gray", alpha=.5)
        axes[1].set_ylim(0, mean_max)
        axes[1].set_ylabel("Mean contact\nCPM (not RNA)")
        num_genes = _draw_intervals(axes[2], genes, start, end, color="#2471a3",
                                    label_column="gene_name", pad_bp=200)
        num_structures = _draw_intervals(axes[3], structures, start, end, color="#e67e22",
                                         label_column="structure_id", pad_bp=500)
        axes[2].set_ylabel("RefSeq genes")
        axes[3].set_ylabel("WT CHIN/OPCID\nannotations")
        axes[3].set_xlabel("Genomic position (bp; 0-based half-open intervals)")
        axes[3].set_xlim(start, end)
        for axis in axes:
            axis.grid(axis="x", color="#dddddd", linewidth=.4)
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        figure.tight_layout()
        name = f"tile_{start:07d}_{end:07d}.png"
        figure.savefig(output / name, dpi=110)
        plt.close(figure)
        manifest.append({"chrom": chrom, "start": start, "end": end, "file": name,
                         "gene_segments": num_genes, "structures": num_structures})
        if len(manifest) % 50 == 0:
            print(f"Rendered {len(manifest)} tiles", flush=True)
    summary = {**metadata, "chrom": chrom, "length_bp": length, "tile_bp": tile_bp,
               "num_tiles": len(manifest), "global_signal_ylim": [0, ymax],
               "global_mean_ylim": [0, mean_max], "tiles": manifest}
    path = output / "manifest.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
