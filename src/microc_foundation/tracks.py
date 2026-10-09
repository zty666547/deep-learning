"""Genome-browser-style contact and structure tracks for Task 3."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STRUCTURE_COLORS = {"CHID": "#d62728", "CHIN": "#1f77b4", "OPCID": "#2ca02c"}
CANDIDATE_COLOR = "#ffbf00"


def centered_rolling_mean(values: np.ndarray, window: int = 9) -> np.ndarray:
    """Return an edge-aware centered rolling mean."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array):
        raise ValueError("values must be a non-empty one-dimensional array")
    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    radius = window // 2
    cumulative = np.concatenate(([0.0], np.cumsum(array)))
    result = np.empty_like(array)
    for index in range(len(array)):
        start = max(0, index - radius)
        end = min(len(array), index + radius + 1)
        result[index] = (cumulative[end] - cumulative[start]) / (end - start)
    return result


def _validate_inputs(
    windows: pd.DataFrame, structures: pd.DataFrame, candidates: pd.DataFrame
) -> None:
    window_required = {
        "chrom", "start", "end", "center", "quality_pass",
        "total_contacts_rep1", "total_contacts_rep2",
    }
    structure_required = {"chrom", "start", "end", "structure_type"}
    candidate_required = {"candidate_region_id", "chrom", "start", "end"}
    if window_required.difference(windows.columns):
        raise ValueError("window table is missing track columns")
    if structure_required.difference(structures.columns):
        raise ValueError("structure table is missing track columns")
    if candidate_required.difference(candidates.columns):
        raise ValueError("candidate table is missing track columns")


def _draw_interval_track(
    axis,
    frame: pd.DataFrame,
    categories: list[str],
    color_lookup: dict[str, str],
    *,
    category_column: str,
) -> None:
    for row_index, category in enumerate(categories):
        subset = frame.loc[frame[category_column].astype(str) == category]
        intervals = [
            (int(row.start) / 1_000_000, (int(row.end) - int(row.start)) / 1_000_000)
            for row in subset.itertuples(index=False)
        ]
        if intervals:
            axis.broken_barh(
                intervals,
                (row_index - 0.32, 0.64),
                facecolors=color_lookup[category],
                alpha=0.82,
            )
    axis.set_yticks(range(len(categories)), categories)


def render_genome_tracks(
    windows: pd.DataFrame,
    structures: pd.DataFrame,
    candidates: pd.DataFrame,
    output_dir: str,
    *,
    rolling_window: int = 9,
    local_candidate_ids: tuple[str, ...] = (
        "candidate_002", "candidate_003", "candidate_004"
    ),
    local_flank_bp: int = 100_000,
) -> Path:
    """Render whole-chromosome tracks and selected local candidate views."""

    _validate_inputs(windows, structures, candidates)
    if local_flank_bp <= 0:
        raise ValueError("local flank must be positive")
    chroms = windows["chrom"].astype(str).unique().tolist()
    if len(chroms) != 1:
        raise ValueError("track renderer currently expects one chromosome")
    chrom = chroms[0]
    selected_windows = windows.sort_values("center").copy()
    selected_structures = structures.loc[structures["chrom"].astype(str) == chrom]
    selected_candidates = candidates.loc[candidates["chrom"].astype(str) == chrom]
    positions = selected_windows["center"].to_numpy(float) / 1_000_000
    rep1 = selected_windows["total_contacts_rep1"].to_numpy(float)
    rep2 = selected_windows["total_contacts_rep2"].to_numpy(float)
    rep1_smooth = centered_rolling_mean(rep1, rolling_window)
    rep2_smooth = centered_rolling_mean(rep2, rolling_window)
    log_ratio = np.log2((rep1 + 1.0) / (rep2 + 1.0))
    quality = selected_windows["quality_pass"].astype(str).str.lower().isin(("true", "1"))

    figure, axes = plt.subplots(
        5, 1, figsize=(14, 9.5), sharex=True,
        gridspec_kw={"height_ratios": (1.25, 1.25, 1.0, 1.0, 0.85)},
        constrained_layout=True,
    )
    for axis, raw, smooth, name, color in (
        (axes[0], rep1, rep1_smooth, "rep1", "#2457a7"),
        (axes[1], rep2, rep2_smooth, "rep2", "#d35400"),
    ):
        axis.plot(positions, raw, color=color, alpha=0.25, linewidth=0.55)
        axis.plot(positions, smooth, color=color, linewidth=1.25)
        axis.set_ylabel(f"{name}\ncontacts")
    axes[0].set_title("Genome-wide Micro-C contact and structure tracks")
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].plot(positions, log_ratio, color="#6a3d9a", linewidth=0.7)
    axes[2].set_ylabel("log2\nrep1/rep2")
    for row in selected_windows.loc[~quality].itertuples(index=False):
        for axis in axes[:3]:
            axis.axvspan(
                int(row.start) / 1_000_000,
                int(row.end) / 1_000_000,
                color="lightgray", alpha=0.06, linewidth=0,
            )

    present_types = set(selected_structures["structure_type"].astype(str))
    class_names = [name for name in ("CHID", "CHIN", "OPCID") if name in present_types]
    _draw_interval_track(
        axes[3], selected_structures, class_names, STRUCTURE_COLORS,
        category_column="structure_type",
    )
    axes[3].set_ylabel("Known")
    candidate_names = selected_candidates["candidate_region_id"].astype(str).tolist()
    _draw_interval_track(
        axes[4], selected_candidates, candidate_names,
        {name: CANDIDATE_COLOR for name in candidate_names},
        category_column="candidate_region_id",
    )
    axes[4].set_ylabel("Candidates")
    axes[4].set_xlabel(f"{chrom} genomic position (Mb)")
    axes[4].set_xlim(
        float(selected_windows["start"].min()) / 1_000_000,
        float(selected_windows["end"].max()) / 1_000_000,
    )
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "genome_multitrack.png", dpi=180)
    plt.close(figure)

    local_candidates = selected_candidates.loc[
        selected_candidates["candidate_region_id"].astype(str).isin(local_candidate_ids)
    ]
    if len(local_candidates):
        local_figure, local_axes = plt.subplots(
            len(local_candidates), 1, figsize=(12, 3.2 * len(local_candidates)),
            constrained_layout=True, squeeze=False,
        )
        for axis, candidate in zip(local_axes[:, 0], local_candidates.itertuples(index=False)):
            local_start = max(0, int(candidate.start) - local_flank_bp)
            local_end = int(candidate.end) + local_flank_bp
            mask = (
                (selected_windows["center"].astype(int) >= local_start)
                & (selected_windows["center"].astype(int) <= local_end)
            )
            local_positions = selected_windows.loc[mask, "center"].to_numpy(float) / 1_000
            local_rep1 = selected_windows.loc[mask, "total_contacts_rep1"].to_numpy(float)
            local_rep2 = selected_windows.loc[mask, "total_contacts_rep2"].to_numpy(float)
            scale = max(float(np.median(np.concatenate((local_rep1, local_rep2)))), 1.0)
            axis.plot(local_positions, local_rep1 / scale, label="rep1", linewidth=1.0)
            axis.plot(local_positions, local_rep2 / scale, label="rep2", linewidth=1.0)
            axis.axvspan(
                int(candidate.start) / 1_000, int(candidate.end) / 1_000,
                color=CANDIDATE_COLOR, alpha=0.24, label="candidate region",
            )
            nearby = selected_structures.loc[
                (selected_structures["start"].astype(int) < local_end)
                & (selected_structures["end"].astype(int) > local_start)
            ]
            for structure in nearby.itertuples(index=False):
                axis.axvspan(
                    int(structure.start) / 1_000, int(structure.end) / 1_000,
                    color=STRUCTURE_COLORS.get(str(structure.structure_type), "gray"),
                    alpha=0.12,
                )
            axis.set(
                ylabel="Contacts / local median",
                title=f"{candidate.candidate_region_id} local context",
                xlim=(local_start / 1_000, local_end / 1_000),
            )
            axis.legend(loc="upper right", ncol=3)
        local_axes[-1, 0].set_xlabel(f"{chrom} genomic position (kb)")
        local_figure.savefig(output / "candidate_local_tracks.png", dpi=180)
        plt.close(local_figure)

    summary = {
        "chrom": chrom,
        "num_windows": len(selected_windows),
        "num_quality_pass": int(quality.sum()),
        "structure_counts": {
            name: int((selected_structures["structure_type"].astype(str) == name).sum())
            for name in class_names
        },
        "num_candidate_regions": len(selected_candidates),
        "rolling_window": rolling_window,
        "local_candidate_ids": local_candidates["candidate_region_id"].astype(str).tolist(),
        "gene_annotation_included": False,
        "gene_annotation_note": "No verified MG1655 gene annotation file was available.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output
