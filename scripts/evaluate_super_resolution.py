"""Evaluate saved Task 5 reconstructions by distance and known interval."""

from __future__ import annotations

import argparse
from pathlib import Path

from microc_foundation.super_resolution_evaluation import evaluate_reconstructions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare saved contact-map reconstructions by genomic distance and annotation."
    )
    parser.add_argument("--reconstruction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bin-size-bp", type=int, default=80)
    parser.add_argument("--near-diagonal-exclusion-bp", type=int, default=160)
    arguments = parser.parse_args()
    files = sorted(Path(arguments.reconstruction_dir).glob("seed_*/test_reconstruction.npz"))
    if not files:
        parser.error("no seed_*/test_reconstruction.npz files found")
    summary = evaluate_reconstructions(
        [str(path) for path in files],
        arguments.output_dir,
        bin_size_bp=arguments.bin_size_bp,
        near_diagonal_exclusion_bp=arguments.near_diagonal_exclusion_bp,
    )
    print(
        f"structures={summary['test_structure_count']} seeds={len(summary['seeds'])} "
        f"distance_groups={len(summary['distance_summary'])} "
        f"known_structure_groups={len(summary['known_structure_summary'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
