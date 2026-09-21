"""Command-line entry point for the minimum runnable Micro-C workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .io import open_cooler
from .normalization import SUPPORTED_METHODS, normalize_matrix
from .visualization import render_heatmap
from .windows import GenomicWindow, fetch_window, parse_region, window_from_center


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read a COOL contact map, extract one local window, and save a heatmap."
    )
    parser.add_argument("--input", required=True, help="Input .cool or .cool.gz file")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--region", help="Zero-based half-open region, e.g. chr1:0-100000")
    parser.add_argument("--chrom", help="Chromosome for center-based window selection")
    parser.add_argument("--center", type=int, help="Genomic center in base pairs")
    parser.add_argument("--size-bp", type=int, help="Window size in base pairs")
    parser.add_argument(
        "--normalization", choices=SUPPORTED_METHODS, default="log1p"
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=None,
        help="Optional upper clipping percentile before normalization",
    )
    parser.add_argument(
        "--unbalanced",
        action="store_true",
        help="Use raw counts instead of COOL balancing weights",
    )
    parser.add_argument("--cmap", default="Reds", help="Matplotlib color map")
    parser.add_argument("--title", default=None, help="Optional heatmap title")
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def _resolve_window(arguments: argparse.Namespace, cool) -> GenomicWindow:
    center_fields = (arguments.chrom, arguments.center, arguments.size_bp)
    has_center_field = any(value is not None for value in center_fields)
    has_all_center_fields = all(value is not None for value in center_fields)
    if arguments.region and has_center_field:
        raise ValueError("Use either --region or --chrom/--center/--size-bp, not both")
    if arguments.region:
        return parse_region(arguments.region)
    if not has_all_center_fields:
        raise ValueError("Provide --region or all of --chrom, --center and --size-bp")
    if arguments.chrom not in cool.chromnames:
        raise ValueError(f"Chromosome {arguments.chrom!r} is not present in the COOL file")
    return window_from_center(
        arguments.chrom,
        arguments.center,
        arguments.size_bp,
        int(cool.chromsizes[arguments.chrom]),
    )


def run(arguments: argparse.Namespace) -> str:
    with open_cooler(arguments.input) as cool:
        window = _resolve_window(arguments, cool)
        try:
            matrix, bin_size = fetch_window(
                cool,
                window,
                balance=not arguments.unbalanced,
            )
        except ValueError as exc:
            if not arguments.unbalanced and "weight" in str(exc).lower():
                raise ValueError(
                    "This COOL file has no usable balancing weights; rerun with --unbalanced"
                ) from exc
            raise

    normalized = normalize_matrix(
        matrix,
        method=arguments.normalization,
        clip_percentile=arguments.clip_percentile,
    )
    output = render_heatmap(
        normalized,
        window,
        arguments.output,
        cmap=arguments.cmap,
        title=arguments.title,
        dpi=arguments.dpi,
    )
    return (
        f"Saved {output} | region={window.region} | bins={matrix.shape[0]} "
        f"| bin_size={bin_size} bp | normalization={arguments.normalization}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        summary = run(arguments)
    except (FileNotFoundError, ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
