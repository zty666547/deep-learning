"""Headless heatmap rendering for local contact matrices."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .windows import GenomicWindow


def render_heatmap(
    matrix: np.ndarray,
    window: GenomicWindow,
    output_path: str,
    cmap: str = "Reds",
    title: str | None = None,
    vmax_percentile: float | None = 99.0,
    dpi: int = 180,
) -> Path:
    """Render a square local contact map and return the created PNG path."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("heatmap matrix must be a square two-dimensional array")
    if values.size == 0:
        raise ValueError("heatmap matrix must not be empty")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    finite = values[np.isfinite(values)]
    vmax = None
    if vmax_percentile is not None and finite.size:
        if not 0 < vmax_percentile <= 100:
            raise ValueError("vmax_percentile must be in (0, 100]")
        vmax = float(np.percentile(finite, vmax_percentile))

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    start_mb, end_mb = window.start / 1e6, window.end / 1e6
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("#EEEEEE")
    fig, axis = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    image = axis.imshow(
        values,
        origin="lower",
        interpolation="nearest",
        cmap=color_map,
        extent=(start_mb, end_mb, start_mb, end_mb),
        vmax=vmax,
        aspect="equal",
    )
    axis.set_xlabel(f"{window.chrom} position (Mb)")
    axis.set_ylabel(f"{window.chrom} position (Mb)")
    axis.set_title(title or f"Micro-C contact map: {window.region}")
    fig.colorbar(image, ax=axis, label="Normalized contact value", shrink=0.85)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output
