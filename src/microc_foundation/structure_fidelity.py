"""Fixed image-level morphology/localization surrogates, not biological detection."""

from __future__ import annotations

import numpy as np


def distance_residual(log_map: np.ndarray, *, minimum_offset: int = 10) -> np.ndarray:
    """Center each valid offset of an already-log1p symmetric contact map."""
    x = np.asarray(log_map, dtype=float)
    if (x.ndim != 2 or x.shape[0] != x.shape[1] or not np.isfinite(x).all()
            or not np.allclose(x, x.T, rtol=1e-5, atol=1e-6)
            or minimum_offset < 1 or minimum_offset > len(x) - 3):
        raise ValueError("need a finite symmetric square log map and valid offset")
    residual = np.zeros_like(x)
    for offset in range(minimum_offset, len(x) - 2):
        rows = np.arange(len(x) - offset)
        values = np.diagonal(x, offset)
        residual[rows, rows + offset] = values - values.mean()
        residual[rows + offset, rows] = values - values.mean()
    return residual


def excess_profile(residual: np.ndarray, *, minimum_offset: int = 10,
                   maximum_offset: int = 80) -> np.ndarray:
    """Positive contact excess per genomic bin, with a fixed three-bin smoother."""
    x = np.asarray(residual, dtype=float)
    if (x.ndim != 2 or x.shape[0] != x.shape[1] or not np.isfinite(x).all()
            or not 1 <= minimum_offset <= maximum_offset <= len(x) - 3):
        raise ValueError("invalid residual map or distance band")
    indices = np.arange(len(x))
    distance = np.abs(indices[:, None] - indices[None, :])
    valid = (distance >= minimum_offset) & (distance <= maximum_offset)
    counts = valid.sum(axis=1)
    if (counts == 0).any():
        raise ValueError("profile has bins without distance-band support")
    profile = (np.maximum(x, 0) * valid).sum(axis=1) / counts
    return np.convolve(np.pad(profile, 1, mode="edge"), [.25, .5, .25], mode="valid")


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    a, b = np.asarray(first, dtype=float).ravel(), np.asarray(second, dtype=float).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("need equally shaped finite vectors")
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare_profiles(target: np.ndarray, predicted: np.ndarray, *, bin_bp: int = 80,
                     top_fraction: float = .1) -> dict:
    """Profile fidelity and dominant-peak displacement, without annotation search."""
    a, b = np.asarray(target, dtype=float), np.asarray(predicted, dtype=float)
    if (a.ndim != 1 or a.shape != b.shape or len(a) < 3 or bin_bp <= 0
            or not 0 < top_fraction <= 1 or not np.isfinite(a).all()
            or not np.isfinite(b).all()):
        raise ValueError("invalid profiles or localization settings")
    r = correlation(a, b)
    if not np.isfinite(r):
        return {"profile_correlation": r, "peak_displacement_bp": float("nan"),
                "top_profile_jaccard": float("nan")}
    count = max(1, int(np.ceil(len(a) * top_fraction)))
    # Stable order resolves ties by genomic index, not random perturbation.
    first = set(np.argsort(-a, kind="stable")[:count])
    second = set(np.argsort(-b, kind="stable")[:count])
    return {"profile_correlation": r,
            "peak_displacement_bp": int(abs(int(a.argmax()) - int(b.argmax())) * bin_bp),
            "top_profile_jaccard": len(first & second) / len(first | second)}


def annotation_shape_r(target: np.ndarray, predicted: np.ndarray, *,
                       start_bp: int, annotation_start: int, annotation_end: int,
                       bin_bp: int = 80, minimum_offset: int = 10,
                       maximum_offset: int = 80) -> tuple[float, int]:
    """Compare distance residuals within the known interval, not across all pixels."""
    a, b = np.asarray(target), np.asarray(predicted)
    if (a.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]
            or not np.isfinite(a).all() or not np.isfinite(b).all() or start_bp < 0
            or bin_bp <= 0 or annotation_start < 0 or annotation_end <= annotation_start
            or not 1 <= minimum_offset <= maximum_offset <= len(a) - 3):
        raise ValueError("invalid maps, annotation or distance settings")
    starts = start_bp + np.arange(len(a)) * bin_bp
    inside = (starts < annotation_end) & (starts + bin_bp > annotation_start)
    rows, columns = np.triu_indices(len(a), k=minimum_offset)
    keep = inside[rows] & inside[columns] & (columns - rows <= maximum_offset)
    n = int(keep.sum())
    return correlation(a[rows[keep], columns[keep]], b[rows[keep], columns[keep]]), n
