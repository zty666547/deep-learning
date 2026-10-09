"""Render Task 3 whole-genome and local multi-track figures."""

from __future__ import annotations

import argparse

import pandas as pd

from microc_foundation.tracks import render_genome_tracks


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot contact and structure tracks.")
    parser.add_argument("--windows", required=True)
    parser.add_argument("--structures", required=True)
    parser.add_argument("--candidate-regions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rolling-window", type=int, default=9)
    parser.add_argument(
        "--local-candidate-ids", nargs="+",
        default=("candidate_002", "candidate_003", "candidate_004"),
    )
    parser.add_argument("--local-flank-bp", type=int, default=100_000)
    arguments = parser.parse_args()
    output = render_genome_tracks(
        pd.read_csv(arguments.windows),
        pd.read_csv(arguments.structures),
        pd.read_csv(arguments.candidate_regions),
        arguments.output_dir,
        rolling_window=arguments.rolling_window,
        local_candidate_ids=tuple(arguments.local_candidate_ids),
        local_flank_bp=arguments.local_flank_bp,
    )
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
