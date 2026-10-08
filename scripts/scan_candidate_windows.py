"""Scan the chromosome for quality-controlled Task 2 candidate windows."""

from __future__ import annotations

import argparse

from microc_foundation.discovery import scan_replicate_windows, write_window_scan


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract interpretable features from aligned diagonal windows."
    )
    parser.add_argument("--cool", action="append", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-size-bp", type=int, default=20_480)
    parser.add_argument("--stride-bp", type=int, default=2_560)
    parser.add_argument("--min-total-contacts", type=float, default=1.0)
    parser.add_argument("--min-nonzero-fraction", type=float, default=0.01)
    parser.add_argument("--min-finite-fraction", type=float, default=0.99)
    parser.add_argument("--max-zero-bin-fraction", type=float, default=0.03)
    parser.add_argument("--max-zero-run-fraction", type=float, default=0.03)
    arguments = parser.parse_args()

    rows = scan_replicate_windows(
        arguments.cool,
        chrom=arguments.chrom,
        window_size_bp=arguments.window_size_bp,
        stride_bp=arguments.stride_bp,
        min_total_contacts=arguments.min_total_contacts,
        min_nonzero_fraction=arguments.min_nonzero_fraction,
        min_finite_fraction=arguments.min_finite_fraction,
        max_zero_bin_fraction=arguments.max_zero_bin_fraction,
        max_zero_run_fraction=arguments.max_zero_run_fraction,
    )
    parameters = {
        "cool_paths": arguments.cool,
        "chrom": arguments.chrom,
        "window_size_bp": arguments.window_size_bp,
        "stride_bp": arguments.stride_bp,
        "min_total_contacts": arguments.min_total_contacts,
        "min_nonzero_fraction": arguments.min_nonzero_fraction,
        "min_finite_fraction": arguments.min_finite_fraction,
        "max_zero_bin_fraction": arguments.max_zero_bin_fraction,
        "max_zero_run_fraction": arguments.max_zero_run_fraction,
    }
    output = write_window_scan(rows, arguments.output_dir, parameters=parameters)
    passed = sum(bool(row["quality_pass"]) for row in rows)
    print(f"Saved {output} | windows={len(rows)} | quality_pass={passed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
