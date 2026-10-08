"""Visual review and threshold sensitivity for Task 2 candidates."""

from __future__ import annotations

import csv
import json
from contextlib import ExitStack
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .clustering import merge_candidate_windows
from .io import open_cooler


def render_candidate_heatmaps(
    cool_paths: list[str],
    windows: pd.DataFrame,
    candidate_regions: pd.DataFrame,
    output_dir: str,
) -> Path:
    """Render rep1, rep2 and difference maps for each representative window."""

    if len(cool_paths) != 2:
        raise ValueError("exactly two biological replicates are required")
    required_windows = {"window_id", "chrom", "start", "end"}
    required_regions = {
        "candidate_region_id",
        "representative_window_id",
        "novelty_score",
        "replicate_correlation",
    }
    if required_windows.difference(windows.columns):
        raise ValueError("window table is missing required columns")
    if required_regions.difference(candidate_regions.columns):
        raise ValueError("candidate region table is missing required columns")
    if candidate_regions.empty:
        raise ValueError("candidate region table is empty")

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    lookup = windows.set_index("window_id", drop=False)
    manifest: list[dict[str, object]] = []
    heatmaps: list[tuple[object, np.ndarray, np.ndarray, np.ndarray]] = []
    with ExitStack() as stack:
        coolers = [stack.enter_context(open_cooler(path)) for path in cool_paths]
        if len({int(cool.binsize) for cool in coolers}) != 1:
            raise ValueError("replicates must have the same bin size")
        for candidate in candidate_regions.itertuples(index=False):
            window_id = str(candidate.representative_window_id)
            if window_id not in lookup.index:
                raise ValueError(f"representative window not found: {window_id}")
            window = lookup.loc[window_id]
            region = f"{window['chrom']}:{int(window['start'])}-{int(window['end'])}"
            matrices = [
                np.asarray(
                    cool.matrix(balance=False, sparse=False).fetch(region), dtype=float
                )
                for cool in coolers
            ]
            log_matrices = [np.log1p(np.nan_to_num(matrix, nan=0.0)) for matrix in matrices]
            difference = log_matrices[0] - log_matrices[1]
            heatmaps.append((candidate, log_matrices[0], log_matrices[1], difference))
            manifest.append(
                {
                    "candidate_region_id": str(candidate.candidate_region_id),
                    "representative_window_id": window_id,
                    "region": region,
                    "novelty_score": float(candidate.novelty_score),
                    "replicate_correlation": float(candidate.replicate_correlation),
                    "rep1_total_contacts": float(matrices[0].sum()),
                    "rep2_total_contacts": float(matrices[1].sum()),
                }
            )

    figure, axes = plt.subplots(
        len(heatmaps),
        3,
        figsize=(10.5, 3.15 * len(heatmaps)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, (candidate, rep1, rep2, difference) in enumerate(heatmaps):
        vmax = float(np.percentile(np.concatenate((rep1.ravel(), rep2.ravel())), 99))
        difference_limit = float(np.percentile(np.abs(difference), 99))
        difference_limit = max(difference_limit, 1e-8)
        axes[row_index, 0].imshow(rep1, cmap="magma", origin="lower", vmin=0, vmax=vmax)
        axes[row_index, 1].imshow(rep2, cmap="magma", origin="lower", vmin=0, vmax=vmax)
        axes[row_index, 2].imshow(
            difference,
            cmap="coolwarm",
            origin="lower",
            vmin=-difference_limit,
            vmax=difference_limit,
        )
        candidate_id = str(candidate.candidate_region_id)
        axes[row_index, 0].set_ylabel(
            f"{candidate_id}\nnovelty={float(candidate.novelty_score):.2f}\nr={float(candidate.replicate_correlation):.3f}"
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_title("rep1 log1p contacts")
    axes[0, 1].set_title("rep2 log1p contacts")
    axes[0, 2].set_title("rep1 - rep2")
    figure.savefig(output / "candidate_heatmap_grid.png", dpi=180)
    plt.close(figure)

    with (output / "candidate_heatmap_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    return output


def evaluate_candidate_sensitivity(
    windows: pd.DataFrame,
    baseline_regions: pd.DataFrame,
    *,
    correlation_thresholds: tuple[float, ...] = (0.88, 0.90, 0.92),
    novelty_quantiles: tuple[float, ...] = (0.85, 0.90, 0.95),
    minimum_support_values: tuple[int, ...] = (2, 3),
    maximum_start_gap_bp: int = 5_120,
) -> dict[str, object]:
    """Reapply candidate thresholds and measure baseline-region persistence."""

    required = {
        "known_overlap_bp",
        "replicate_correlation",
        "novelty_score",
        "window_id",
        "chrom",
        "start",
        "end",
        "center",
        "cluster",
        "nearest_known_distance_bp",
    }
    if required.difference(windows.columns):
        raise ValueError("window table is missing sensitivity fields")
    if baseline_regions.empty:
        raise ValueError("baseline candidate regions are empty")
    table_rows: list[dict[str, object]] = []
    persistence_counts = {
        str(row.candidate_region_id): 0
        for row in baseline_regions.itertuples(index=False)
    }
    total_settings = 0
    for correlation in correlation_thresholds:
        for quantile in novelty_quantiles:
            eligible = (
                (windows["known_overlap_bp"].to_numpy(int) == 0)
                & (windows["replicate_correlation"].to_numpy(float) >= correlation)
            )
            if not eligible.any():
                continue
            threshold = float(
                np.quantile(windows.loc[eligible, "novelty_score"], quantile)
            )
            selected = windows.loc[
                eligible & (windows["novelty_score"].to_numpy(float) >= threshold)
            ]
            for minimum_support in minimum_support_values:
                regions = merge_candidate_windows(
                    selected,
                    maximum_start_gap_bp=maximum_start_gap_bp,
                    minimum_support=minimum_support,
                )
                total_settings += 1
                table_rows.append(
                    {
                        "minimum_replicate_correlation": correlation,
                        "novelty_quantile": quantile,
                        "minimum_supporting_windows": minimum_support,
                        "novelty_threshold": threshold,
                        "num_candidate_windows": len(selected),
                        "num_candidate_regions": len(regions),
                    }
                )
                for baseline in baseline_regions.itertuples(index=False):
                    center = int(baseline.representative_center)
                    reproduced = (
                        (regions["chrom"].astype(str) == str(baseline.chrom))
                        & (regions["start"].astype(int) <= center)
                        & (regions["end"].astype(int) > center)
                    ).any() if len(regions) else False
                    persistence_counts[str(baseline.candidate_region_id)] += int(reproduced)
    persistence_rows = [
        {
            "candidate_region_id": candidate_id,
            "settings_reproduced": count,
            "total_settings": total_settings,
            "persistence_fraction": count / total_settings if total_settings else 0.0,
        }
        for candidate_id, count in persistence_counts.items()
    ]
    return {
        "settings": table_rows,
        "persistence": persistence_rows,
        "summary": {
            "num_settings": total_settings,
            "candidate_region_count_min": min(
                (int(row["num_candidate_regions"]) for row in table_rows), default=0
            ),
            "candidate_region_count_max": max(
                (int(row["num_candidate_regions"]) for row in table_rows), default=0
            ),
            "median_candidate_region_count": float(
                np.median([row["num_candidate_regions"] for row in table_rows])
            )
            if table_rows
            else 0.0,
        },
    }


def write_sensitivity_results(result: dict[str, object], output_dir: str) -> Path:
    """Write sensitivity tables, summary and persistence figure."""

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    settings = pd.DataFrame(result["settings"])
    persistence = pd.DataFrame(result["persistence"])
    settings.to_csv(output / "sensitivity_settings.csv", index=False)
    persistence.to_csv(output / "candidate_persistence.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for minimum_support, group in settings.groupby("minimum_supporting_windows"):
        ordered = group.sort_values(
            ["minimum_replicate_correlation", "novelty_quantile"]
        )
        axes[0].plot(
            range(len(ordered)),
            ordered["num_candidate_regions"],
            marker="o",
            label=f"support ≥ {minimum_support}",
        )
    axes[0].set(
        xlabel="Correlation/novelty threshold combination",
        ylabel="Candidate region count",
        title="Candidate-count sensitivity",
    )
    axes[0].legend()
    axes[1].bar(
        persistence["candidate_region_id"], persistence["persistence_fraction"]
    )
    axes[1].set(
        xlabel="Baseline candidate",
        ylabel="Fraction of settings reproduced",
        title="Candidate persistence across settings",
        ylim=(0, 1),
    )
    axes[1].tick_params(axis="x", rotation=35)
    figure.savefig(output / "candidate_sensitivity.png", dpi=180)
    plt.close(figure)
    return output
