"""Genome-wide candidate-window scanning for Task 2 structure discovery."""

from __future__ import annotations

import csv
import json
from contextlib import ExitStack
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import open_cooler

FEATURE_NAMES = (
    "total_contacts",
    "mean_contact",
    "std_contact",
    "max_contact",
    "nonzero_fraction",
    "diagonal_mean",
    "near_band_mean",
    "mid_band_mean",
    "far_band_mean",
    "near_far_ratio",
    "decay_slope",
    "center_enrichment",
    "coefficient_of_variation",
    "contact_entropy",
)


def coarsen_cool(
    input_path: str,
    output_path: str,
    *,
    factor: int,
    chunksize: int = 10_000_000,
) -> Path:
    """Create a coarsened COOL file, or reuse a compatible existing output."""

    if factor <= 1:
        raise ValueError("factor must be greater than one")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open_cooler(input_path) as source:
        expected_bin_size = int(source.binsize) * factor
        if output.exists():
            with open_cooler(str(output)) as existing:
                if int(existing.binsize) != expected_bin_size:
                    raise ValueError(
                        "existing coarsened COOL has an incompatible bin size"
                    )
            return output
        import cooler

        cooler.coarsen_cooler(
            source.uri,
            str(output),
            factor=factor,
            chunksize=chunksize,
            nproc=1,
        )
    return output


def diagonal_window_starts(
    chrom_length: int,
    bin_size: int,
    window_size_bp: int,
    stride_bp: int,
) -> list[int]:
    """Return full, bin-aligned diagonal window starts including the right edge."""

    if min(chrom_length, bin_size, window_size_bp, stride_bp) <= 0:
        raise ValueError("lengths and resolutions must be positive")
    if window_size_bp > chrom_length:
        raise ValueError("window exceeds chromosome length")
    if window_size_bp % bin_size or stride_bp % bin_size:
        raise ValueError("window size and stride must be divisible by bin size")
    last_start = ((chrom_length - window_size_bp) // bin_size) * bin_size
    starts = list(range(0, last_start + 1, stride_bp))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _offset_band_mean(matrix: np.ndarray, minimum: int, maximum: int) -> float:
    values: list[np.ndarray] = []
    for offset in range(minimum, min(maximum, len(matrix) - 1) + 1):
        diagonal = np.diagonal(matrix, offset=offset)
        if len(diagonal):
            values.append(diagonal)
    return float(np.concatenate(values).mean()) if values else 0.0


def extract_contact_features(matrix: np.ndarray, eps: float = 1e-8) -> dict[str, float]:
    """Extract interpretable intensity, distance-decay and texture features."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square and two-dimensional")
    if len(values) < 8:
        raise ValueError("matrix must contain at least eight bins")
    if eps <= 0:
        raise ValueError("eps must be positive")
    finite = np.isfinite(values)
    finite_fraction = float(finite.mean())
    cleaned = np.where(finite, values, 0.0)
    if (cleaned < 0).any():
        raise ValueError("contact counts must be non-negative")

    total = float(cleaned.sum())
    mean = float(cleaned.mean())
    std = float(cleaned.std())
    diagonal_mean = float(np.diagonal(cleaned).mean())
    near_mean = _offset_band_mean(cleaned, 1, 3)
    mid_mean = _offset_band_mean(cleaned, 4, 15)
    far_mean = _offset_band_mean(cleaned, 16, min(63, len(cleaned) - 1))

    maximum_offset = min(32, len(cleaned) - 1)
    distances = np.arange(1, maximum_offset + 1, dtype=float)
    decay_values = np.asarray(
        [float(np.diagonal(cleaned, offset=int(offset)).mean()) for offset in distances]
    )
    valid_decay = decay_values > 0
    decay_slope = (
        float(np.polyfit(np.log1p(distances[valid_decay]), np.log1p(decay_values[valid_decay]), 1)[0])
        if valid_decay.sum() >= 2
        else 0.0
    )

    quarter = max(1, len(cleaned) // 4)
    center_start = (len(cleaned) - quarter) // 2
    center_mean = float(
        cleaned[
            center_start : center_start + quarter,
            center_start : center_start + quarter,
        ].mean()
    )
    positive = cleaned[cleaned > 0]
    if total > 0 and len(positive):
        probabilities = positive / positive.sum()
        entropy = float(-(probabilities * np.log(probabilities + eps)).sum())
        entropy /= float(np.log(len(positive))) if len(positive) > 1 else 1.0
    else:
        entropy = 0.0

    return {
        "finite_fraction": finite_fraction,
        "total_contacts": total,
        "mean_contact": mean,
        "std_contact": std,
        "max_contact": float(cleaned.max()),
        "nonzero_fraction": float((cleaned > 0).mean()),
        "diagonal_mean": diagonal_mean,
        "near_band_mean": near_mean,
        "mid_band_mean": mid_mean,
        "far_band_mean": far_mean,
        "near_far_ratio": near_mean / (far_mean + eps),
        "decay_slope": decay_slope,
        "center_enrichment": center_mean / (mean + eps),
        "coefficient_of_variation": std / (mean + eps),
        "contact_entropy": entropy,
    }


def matrix_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return log-count Pearson correlation over the upper triangle."""

    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("matrices must be equal square arrays")
    triangle = np.triu_indices(first.shape[0])
    x = np.log1p(np.where(np.isfinite(first[triangle]), first[triangle], 0.0))
    y = np.log1p(np.where(np.isfinite(second[triangle]), second[triangle], 0.0))
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def scan_replicate_windows(
    cool_paths: list[str],
    *,
    chrom: str,
    window_size_bp: int = 20_480,
    stride_bp: int = 2_560,
    min_total_contacts: float = 1.0,
    min_nonzero_fraction: float = 0.01,
    min_finite_fraction: float = 0.99,
) -> list[dict[str, object]]:
    """Scan identical diagonal windows across replicates and calculate features."""

    if len(cool_paths) < 2:
        raise ValueError("at least two biological replicates are required")
    if min_total_contacts < 0:
        raise ValueError("minimum total contacts must be non-negative")
    if not 0 <= min_nonzero_fraction <= 1 or not 0 <= min_finite_fraction <= 1:
        raise ValueError("fraction thresholds must lie in [0, 1]")

    with ExitStack() as stack:
        coolers = [stack.enter_context(open_cooler(path)) for path in cool_paths]
        bin_sizes = {int(cool.binsize) for cool in coolers}
        if len(bin_sizes) != 1:
            raise ValueError("all replicates must have the same bin size")
        bin_size = bin_sizes.pop()
        if any(chrom not in cool.chromnames for cool in coolers):
            raise ValueError(f"chromosome {chrom!r} is missing from a replicate")
        chrom_lengths = {int(cool.chromsizes[chrom]) for cool in coolers}
        if len(chrom_lengths) != 1:
            raise ValueError("replicates have different chromosome lengths")
        chrom_length = chrom_lengths.pop()
        starts = diagonal_window_starts(
            chrom_length, bin_size, window_size_bp, stride_bp
        )

        rows: list[dict[str, object]] = []
        for index, start in enumerate(starts, start=1):
            end = start + window_size_bp
            region = f"{chrom}:{start}-{end}"
            matrices = [
                np.asarray(
                    cool.matrix(balance=False, sparse=False).fetch(region), dtype=float
                )
                for cool in coolers
            ]
            features = [extract_contact_features(matrix) for matrix in matrices]
            quality_reasons: list[str] = []
            for replicate_index, values in enumerate(features, start=1):
                if values["total_contacts"] < min_total_contacts:
                    quality_reasons.append(f"rep{replicate_index}_low_total")
                if values["nonzero_fraction"] < min_nonzero_fraction:
                    quality_reasons.append(f"rep{replicate_index}_sparse")
                if values["finite_fraction"] < min_finite_fraction:
                    quality_reasons.append(f"rep{replicate_index}_missing")

            row: dict[str, object] = {
                "window_id": f"window_{index:05d}",
                "chrom": chrom,
                "start": start,
                "end": end,
                "center": (start + end) // 2,
                "quality_pass": not quality_reasons,
                "quality_reasons": ";".join(quality_reasons),
                "replicate_correlation": matrix_correlation(matrices[0], matrices[1]),
            }
            for feature_name in FEATURE_NAMES:
                replicate_values = [values[feature_name] for values in features]
                for replicate_index, value in enumerate(replicate_values, start=1):
                    row[f"{feature_name}_rep{replicate_index}"] = value
                row[f"{feature_name}_mean"] = float(np.mean(replicate_values))
                row[f"{feature_name}_abs_diff"] = float(
                    abs(replicate_values[0] - replicate_values[1])
                )
            rows.append(row)
    return rows


def write_window_scan(
    rows: list[dict[str, object]],
    output_dir: str,
    *,
    parameters: dict[str, object],
) -> Path:
    """Write candidate-window table, summary and genome-position quality plot."""

    if not rows:
        raise ValueError("scan produced no windows")
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "candidate_windows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    passed = [row for row in rows if row["quality_pass"]]
    correlations = np.asarray(
        [float(row["replicate_correlation"]) for row in passed], dtype=float
    )
    summary = {
        "parameters": parameters,
        "num_windows": len(rows),
        "num_quality_pass": len(passed),
        "num_quality_fail": len(rows) - len(passed),
        "quality_pass_fraction": len(passed) / len(rows),
        "replicate_correlation": {
            "mean": float(correlations.mean()) if len(correlations) else None,
            "median": float(np.median(correlations)) if len(correlations) else None,
            "minimum": float(correlations.min()) if len(correlations) else None,
            "maximum": float(correlations.max()) if len(correlations) else None,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    positions = np.asarray([int(row["center"]) for row in rows]) / 1_000_000
    figure, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, constrained_layout=True)
    axes[0].plot(
        positions,
        [float(row["total_contacts_mean"]) for row in rows],
        linewidth=0.8,
    )
    axes[0].set(ylabel="Mean total contacts", title="Genome-wide candidate-window scan")
    axes[1].plot(
        positions,
        [float(row["replicate_correlation"]) for row in rows],
        linewidth=0.8,
        color="tab:orange",
    )
    all_correlations = np.asarray(
        [float(row["replicate_correlation"]) for row in rows], dtype=float
    )
    correlation_min = max(-1.0, float(all_correlations.min()) - 0.02)
    correlation_max = min(1.0, float(all_correlations.max()) + 0.02)
    axes[1].set(
        xlabel="Genomic position (Mb)",
        ylabel="Replicate correlation",
        ylim=(correlation_min, correlation_max),
    )
    figure.savefig(output / "window_scan_overview.png", dpi=180)
    plt.close(figure)
    return output
