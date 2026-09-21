"""Reusable data foundations for Micro-C contact matrices."""

from .annotations import read_structures_csv, read_structures_excel, write_structures_csv
from .io import open_cooler
from .normalization import normalize_matrix
from .visualization import render_heatmap
from .windows import GenomicWindow, parse_region, window_from_center

__all__ = [
    "GenomicWindow",
    "normalize_matrix",
    "open_cooler",
    "parse_region",
    "read_structures_csv",
    "read_structures_excel",
    "render_heatmap",
    "window_from_center",
    "write_structures_csv",
]

__version__ = "0.1.0"
