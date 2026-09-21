"""Validated reader for known chromatin-structure annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


_ALIASES: Dict[str, str] = {
    "chr": "chrom",
    "chromosome": "chrom",
    "type": "structure_type",
    "label": "structure_type",
    "class": "structure_type",
}


def read_structures_csv(path: str) -> pd.DataFrame:
    """Read and validate ``structures.csv``.

    Required logical columns are ``chrom``, ``start`` and ``end``. Common
    aliases are normalized, while all additional columns are retained. Genomic
    coordinates use the zero-based, half-open convention.
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Structure annotation file not found: {source}")

    frame = pd.read_csv(source)
    frame.columns = [str(column).strip() for column in frame.columns]
    rename = {
        column: _ALIASES[column.lower()]
        for column in frame.columns
        if column.lower() in _ALIASES and _ALIASES[column.lower()] not in frame.columns
    }
    frame = frame.rename(columns=rename)

    required = {"chrom", "start", "end"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"structures.csv is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("structures.csv contains no rows")

    frame["chrom"] = frame["chrom"].astype(str).str.strip()
    for column in ("start", "end"):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if ((numeric % 1) != 0).any():
            raise ValueError(f"{column} coordinates must be integers")
        frame[column] = numeric.astype("int64")

    if (frame["chrom"] == "").any():
        raise ValueError("chrom values must not be empty")
    if (frame["start"] < 0).any():
        raise ValueError("start coordinates must be non-negative")
    if (frame["end"] <= frame["start"]).any():
        raise ValueError("every end coordinate must be greater than start")
    if "structure_type" in frame.columns:
        frame["structure_type"] = frame["structure_type"].astype(str).str.strip()
    return frame

