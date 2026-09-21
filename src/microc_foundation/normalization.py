"""Small, explicit normalization interface for local contact matrices."""

from __future__ import annotations

from typing import Optional

import numpy as np


SUPPORTED_METHODS = ("none", "log1p", "minmax", "zscore", "max")


def normalize_matrix(
    matrix: np.ndarray,
    method: str = "log1p",
    clip_percentile: Optional[float] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize a 2-D matrix while preserving non-finite entries.

    ``clip_percentile`` is calculated only over finite values. This interface is
    deliberately local and does not replace ICE/KR balancing stored in COOL.
    """

    values = np.asarray(matrix, dtype=float).copy()
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {SUPPORTED_METHODS}")
    if eps <= 0:
        raise ValueError("eps must be positive")

    finite = np.isfinite(values)
    if not finite.any():
        return values

    if clip_percentile is not None:
        if not 0 < clip_percentile <= 100:
            raise ValueError("clip_percentile must be in (0, 100]")
        upper = np.percentile(values[finite], clip_percentile)
        values[finite] = np.minimum(values[finite], upper)

    finite_values = values[finite]
    if method == "none":
        return values
    if method == "log1p":
        if np.min(finite_values) < 0:
            raise ValueError("log1p normalization requires non-negative values")
        values[finite] = np.log1p(finite_values)
    elif method == "minmax":
        minimum = float(np.min(finite_values))
        span = float(np.max(finite_values) - minimum)
        values[finite] = 0.0 if span < eps else (finite_values - minimum) / span
    elif method == "zscore":
        mean = float(np.mean(finite_values))
        std = float(np.std(finite_values))
        values[finite] = 0.0 if std < eps else (finite_values - mean) / std
    elif method == "max":
        scale = float(np.max(np.abs(finite_values)))
        values[finite] = 0.0 if scale < eps else finite_values / scale
    return values

