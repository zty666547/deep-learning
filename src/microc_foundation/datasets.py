"""Fixed-shape window extraction for supervised structure classification."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .io import open_cooler
from .normalization import normalize_matrix
from .windows import GenomicWindow, fetch_window


def aligned_window_from_center(
    cool,
    chrom: str,
    center: int,
    window_bins: int,
) -> GenomicWindow:
    """Return a fixed-bin window aligned to the source COOL bin grid."""

    if chrom not in cool.chromnames:
        raise ValueError(f"Chromosome {chrom!r} is not present in the COOL file")
    if window_bins <= 0:
        raise ValueError("window_bins must be positive")

    bin_size = int(cool.binsize)
    chrom_length = int(cool.chromsizes[chrom])
    chrom_bins = (chrom_length + bin_size - 1) // bin_size
    if window_bins > chrom_bins:
        raise ValueError("window_bins exceeds the chromosome bin count")
    if center < 0 or center > chrom_length:
        raise ValueError("center lies outside the chromosome")

    center_bin = min(center // bin_size, chrom_bins - 1)
    start_bin = center_bin - window_bins // 2
    start_bin = max(0, min(start_bin, chrom_bins - window_bins))
    end_bin = start_bin + window_bins
    start = start_bin * bin_size
    end = min(end_bin * bin_size, chrom_length)
    return GenomicWindow(chrom, start, end)


def pool_square_matrix(
    matrix: np.ndarray,
    factor: int,
    method: str = "sum",
) -> np.ndarray:
    """Aggregate non-overlapping square blocks by sum or mean."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square and two-dimensional")
    if factor <= 0:
        raise ValueError("factor must be positive")
    if values.shape[0] % factor:
        raise ValueError("matrix dimensions must be divisible by factor")
    if method not in {"sum", "mean"}:
        raise ValueError("pooling method must be 'sum' or 'mean'")

    pooled = values.reshape(
        values.shape[0] // factor,
        factor,
        values.shape[1] // factor,
        factor,
    )
    if method == "sum":
        return pooled.sum(axis=(1, 3))
    return pooled.mean(axis=(1, 3))


def build_structure_dataset(
    cool_paths: Iterable[str],
    structures: pd.DataFrame,
    output_path: str,
    *,
    window_size_bp: int = 20_480,
    pool_factor: int = 16,
    pooling: str = "sum",
    normalization: str = "log1p",
    balance: bool = False,
    limit: int | None = None,
    per_class_limit: int | None = None,
) -> Path:
    """Extract a compact NPZ tensor for each structure across all replicates."""

    required = {"structure_id", "chrom", "start", "center", "end", "structure_type"}
    missing = sorted(required.difference(structures.columns))
    if missing:
        raise ValueError(f"structures are missing columns: {', '.join(missing)}")
    if structures.empty:
        raise ValueError("structures contain no rows")
    if window_size_bp <= 0:
        raise ValueError("window_size_bp must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if per_class_limit is not None and per_class_limit <= 0:
        raise ValueError("per_class_limit must be positive")
    if limit is not None and per_class_limit is not None:
        raise ValueError("use either limit or per_class_limit, not both")

    if per_class_limit:
        selected = (
            structures.groupby("structure_type", sort=False, group_keys=False)
            .head(per_class_limit)
            .copy()
        )
    else:
        selected = structures.iloc[:limit].copy() if limit else structures.copy()
    paths = [str(Path(path).expanduser()) for path in cool_paths]
    if not paths:
        raise ValueError("at least one COOL path is required")

    matrices_by_replicate: list[np.ndarray] = []
    windows: list[GenomicWindow] | None = None
    bin_size: int | None = None
    for path in paths:
        replicate_matrices: list[np.ndarray] = []
        replicate_windows: list[GenomicWindow] = []
        with open_cooler(path) as cool:
            current_bin_size = int(cool.binsize)
            if window_size_bp % current_bin_size:
                raise ValueError("window_size_bp must be divisible by the COOL bin size")
            if bin_size is None:
                bin_size = current_bin_size
            elif current_bin_size != bin_size:
                raise ValueError("all COOL files must use the same bin size")
            window_bins = window_size_bp // current_bin_size
            if window_bins % pool_factor:
                raise ValueError("window bin count must be divisible by pool_factor")

            for row in selected.itertuples(index=False):
                window = aligned_window_from_center(
                    cool,
                    str(row.chrom),
                    int(row.center),
                    window_bins,
                )
                matrix, _ = fetch_window(cool, window, balance=balance, fill_value=0.0)
                if matrix.shape != (window_bins, window_bins):
                    raise ValueError(
                        f"Unexpected matrix shape {matrix.shape} for {row.structure_id}"
                    )
                pooled = pool_square_matrix(matrix, pool_factor, method=pooling)
                replicate_matrices.append(
                    normalize_matrix(pooled, method=normalization).astype("float32")
                )
                replicate_windows.append(window)

        if windows is None:
            windows = replicate_windows
        elif replicate_windows != windows:
            raise ValueError("replicates produced different genomic windows")
        matrices_by_replicate.append(np.stack(replicate_matrices))

    tensor = np.stack(matrices_by_replicate, axis=1)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    assert windows is not None and bin_size is not None
    np.savez_compressed(
        output,
        matrices=tensor,
        structure_id=selected["structure_id"].astype(str).to_numpy(dtype=str),
        labels=selected["structure_type"].astype(str).to_numpy(dtype=str),
        chrom=selected["chrom"].astype(str).to_numpy(dtype=str),
        annotation_start=selected["start"].astype("int64").to_numpy(),
        annotation_end=selected["end"].astype("int64").to_numpy(),
        center=selected["center"].astype("int64").to_numpy(),
        window_start=np.array([window.start for window in windows], dtype="int64"),
        window_end=np.array([window.end for window in windows], dtype="int64"),
        replicate=np.asarray([Path(path).stem for path in paths], dtype=str),
        source_bin_size=np.array(bin_size, dtype="int64"),
        pooled_bin_size=np.array(bin_size * pool_factor, dtype="int64"),
        normalization=np.array(normalization),
        pooling=np.array(pooling),
    )
    return output
