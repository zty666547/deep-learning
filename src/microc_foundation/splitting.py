"""Leakage-aware contiguous genomic train/validation/test splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def assign_contiguous_splits(
    structures: pd.DataFrame,
    chrom_length: int,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    window_size_bp: int = 20_480,
) -> pd.DataFrame:
    """Assign structures to contiguous genome segments with boundary exclusion.

    Samples whose extraction window would cross a split boundary receive the
    excluded_boundary label. This prevents overlapping local windows from
    appearing in different model splits.
    """

    required = {"structure_id", "chrom", "center", "structure_type"}
    missing = sorted(required.difference(structures.columns))
    if missing:
        raise ValueError(f"structures are missing columns: {', '.join(missing)}")
    if chrom_length <= 0:
        raise ValueError("chrom_length must be positive")
    if window_size_bp <= 0:
        raise ValueError("window_size_bp must be positive")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")

    frame = structures.copy()
    frame["center"] = pd.to_numeric(frame["center"], errors="raise").astype("int64")
    if ((frame["center"] < 0) | (frame["center"] > chrom_length)).any():
        raise ValueError("structure center lies outside the chromosome")

    train_end = int(chrom_length * train_fraction)
    validation_end = int(chrom_length * (train_fraction + validation_fraction))
    half_window = (window_size_bp + 1) // 2

    frame["split"] = "test"
    frame.loc[frame["center"] < validation_end, "split"] = "validation"
    frame.loc[frame["center"] < train_end, "split"] = "train"

    crosses_boundary = (
        (frame["center"] - train_end).abs() < half_window
    ) | ((frame["center"] - validation_end).abs() < half_window)
    frame.loc[crosses_boundary, "split"] = "excluded_boundary"
    frame["split_train_end"] = train_end
    frame["split_validation_end"] = validation_end
    frame["split_chrom_length"] = chrom_length
    frame["split_window_size_bp"] = window_size_bp
    return frame


def validate_split_class_coverage(frame: pd.DataFrame) -> None:
    """Require every retained split to contain every structure class."""

    required = {"split", "structure_type"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"split manifest is missing columns: {', '.join(missing)}")

    retained = frame.loc[frame["split"].isin(("train", "validation", "test"))]
    expected = set(retained["structure_type"].astype(str))
    for split_name in ("train", "validation", "test"):
        observed = set(
            retained.loc[retained["split"] == split_name, "structure_type"].astype(str)
        )
        missing_classes = sorted(expected.difference(observed))
        if missing_classes:
            raise ValueError(
                f"{split_name} split is missing classes: {', '.join(missing_classes)}"
            )


def write_split_manifest(frame: pd.DataFrame, output_path: str) -> Path:
    """Write a split manifest after validating class coverage."""

    validate_split_class_coverage(frame)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output
