"""Read-only checks for structure-annotation coordinates against COOL bins."""

from __future__ import annotations

import pandas as pd


def audit_annotation_bin_alignment(
    structures: pd.DataFrame,
    bins: pd.DataFrame,
    *,
    bin_size_bp: int,
) -> dict[str, object]:
    """Compare source intervals with exact COOL boundaries and a 1-bp shift.

    A direct match is evidence that workbook coordinates describe zero-based,
    half-open bin-aligned intervals. This check does not replace an explicit
    coordinate declaration from the source.
    """
    if bin_size_bp <= 0:
        raise ValueError("bin_size_bp must be positive")
    required_structures = {"structure_id", "chrom", "start", "end", "structure_type"}
    required_bins = {"chrom", "start", "end"}
    if required_structures.difference(structures.columns):
        raise ValueError("structures must contain IDs, coordinates, and structure types")
    if required_bins.difference(bins.columns):
        raise ValueError("bins must contain chromosome and start/end boundaries")
    if structures.empty or bins.empty:
        raise ValueError("structures and bins must not be empty")

    bin_starts: dict[str, set[int]] = {}
    bin_ends: dict[str, set[int]] = {}
    chrom_lengths: dict[str, int] = {}
    for chrom, group in bins.groupby(bins["chrom"].astype(str), sort=False):
        chrom_starts = group["start"].astype(int)
        chrom_ends = group["end"].astype(int)
        bin_starts[str(chrom)] = set(chrom_starts)
        bin_ends[str(chrom)] = set(chrom_ends)
        chrom_lengths[str(chrom)] = int(chrom_ends.max())

    rows: list[dict[str, object]] = []
    for row in structures.itertuples(index=False):
        chrom = str(row.chrom)
        start, end = int(row.start), int(row.end)
        starts = bin_starts.get(chrom, set())
        ends = bin_ends.get(chrom, set())
        rows.append(
            {
                "structure_id": str(row.structure_id),
                "structure_type": str(row.structure_type),
                "chrom": chrom,
                "start": start,
                "end": end,
                "valid_bounds": 0 <= start < end <= chrom_lengths.get(chrom, -1),
                "start_on_bin_boundary": start in starts,
                "end_on_bin_boundary": end in ends,
                "one_based_start_adjusted_boundary": start - 1 in starts,
                "start_mod_resolution": start % bin_size_bp,
                "end_mod_resolution": end % bin_size_bp,
            }
        )
    frame = pd.DataFrame(rows)
    by_type = {}
    for label, group in frame.groupby("structure_type", sort=True):
        by_type[str(label)] = {
            "count": len(group),
            "direct_start_boundary_count": int(group["start_on_bin_boundary"].sum()),
            "direct_end_boundary_count": int(group["end_on_bin_boundary"].sum()),
            "one_based_adjusted_start_boundary_count": int(
                group["one_based_start_adjusted_boundary"].sum()
            ),
        }
    return {
        "annotation_count": len(frame),
        "bin_size_bp": int(bin_size_bp),
        "valid_bounds_count": int(frame["valid_bounds"].sum()),
        "start_on_bin_boundary_count": int(frame["start_on_bin_boundary"].sum()),
        "end_on_bin_boundary_count": int(frame["end_on_bin_boundary"].sum()),
        "both_direct_boundaries_count": int(
            (frame["start_on_bin_boundary"] & frame["end_on_bin_boundary"]).sum()
        ),
        "one_based_adjusted_start_boundary_count": int(
            frame["one_based_start_adjusted_boundary"].sum()
        ),
        "start_and_end_multiple_of_resolution_count": int(
            (
                (frame["start_mod_resolution"] == 0)
                & (frame["end_mod_resolution"] == 0)
            ).sum()
        ),
        "by_type": by_type,
        "examples": frame.head(5).to_dict(orient="records"),
        "interpretation": (
            "Direct bin-boundary alignment supports zero-based, half-open intervals; "
            "the source does not explicitly state this convention in the workbook headers."
        ),
    }
