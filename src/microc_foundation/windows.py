"""Coordinate utilities for local square contact-matrix windows."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_REGION_PATTERN = re.compile(r"^(?P<chrom>[^:]+):(?P<start>[0-9,]+)-(?P<end>[0-9,]+)$")


@dataclass(frozen=True)
class GenomicWindow:
    """Zero-based, half-open genomic interval."""

    chrom: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.chrom:
            raise ValueError("chrom must not be empty")
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end <= self.start:
            raise ValueError("end must be greater than start")

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def region(self) -> str:
        return f"{self.chrom}:{self.start}-{self.end}"


def parse_region(region: str) -> GenomicWindow:
    """Parse ``chrom:start-end`` into a validated interval."""

    match = _REGION_PATTERN.fullmatch(region.strip())
    if match is None:
        raise ValueError("region must use the form chrom:start-end")
    return GenomicWindow(
        match.group("chrom"),
        int(match.group("start").replace(",", "")),
        int(match.group("end").replace(",", "")),
    )


def window_from_center(
    chrom: str,
    center: int,
    size_bp: int,
    chrom_length: int | None = None,
) -> GenomicWindow:
    """Build a window around a genomic center and optionally clamp to a chromosome.

    When possible, clamping preserves ``size_bp`` by shifting the interval away
    from the chromosome boundary. If the requested size exceeds the chromosome,
    the whole chromosome is returned.
    """

    if center < 0:
        raise ValueError("center must be non-negative")
    if size_bp <= 0:
        raise ValueError("size_bp must be positive")
    if chrom_length is not None and chrom_length <= 0:
        raise ValueError("chrom_length must be positive")
    if chrom_length is not None and center > chrom_length:
        raise ValueError("center lies outside the chromosome")

    start = center - size_bp // 2
    end = start + size_bp

    if chrom_length is None:
        if start < 0:
            start = 0
            end = size_bp
    elif size_bp >= chrom_length:
        start, end = 0, chrom_length
    else:
        if start < 0:
            start, end = 0, size_bp
        if end > chrom_length:
            start, end = chrom_length - size_bp, chrom_length

    return GenomicWindow(chrom, start, end)


def fetch_window(
    cool,
    window: GenomicWindow,
    balance: bool = True,
    fill_value: float | None = None,
) -> tuple[np.ndarray, int]:
    """Fetch a dense local matrix and return it with the COOL bin size."""

    if window.chrom not in cool.chromnames:
        raise ValueError(f"Chromosome {window.chrom!r} is not present in the COOL file")
    chrom_length = int(cool.chromsizes[window.chrom])
    if window.end > chrom_length:
        raise ValueError(
            f"Window {window.region} exceeds chromosome length {chrom_length}"
        )

    matrix = np.asarray(
        cool.matrix(balance=balance, sparse=False).fetch(window.region),
        dtype=float,
    )
    if fill_value is not None:
        matrix = np.nan_to_num(matrix, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return matrix, int(cool.binsize)
