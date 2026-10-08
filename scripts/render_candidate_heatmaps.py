"""Render rep1/rep2 heatmaps for merged Task 2 candidates."""

from __future__ import annotations

import argparse

import pandas as pd

from microc_foundation.candidate_validation import render_candidate_heatmaps


def main() -> int:
    parser = argparse.ArgumentParser(description="Render candidate heatmap review grid.")
    parser.add_argument("--cool", action="append", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--candidate-regions", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    output = render_candidate_heatmaps(
        arguments.cool,
        pd.read_csv(arguments.windows),
        pd.read_csv(arguments.candidate_regions),
        arguments.output_dir,
    )
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
