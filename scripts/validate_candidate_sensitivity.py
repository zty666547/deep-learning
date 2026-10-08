"""Evaluate how Task 2 candidates change under nearby thresholds."""

from __future__ import annotations

import argparse

import pandas as pd

from microc_foundation.candidate_validation import (
    evaluate_candidate_sensitivity,
    write_sensitivity_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run candidate threshold sensitivity.")
    parser.add_argument("--windows", required=True)
    parser.add_argument("--candidate-regions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--correlation-thresholds", nargs="+", type=float, default=(0.88, 0.90, 0.92)
    )
    parser.add_argument(
        "--novelty-quantiles", nargs="+", type=float, default=(0.85, 0.90, 0.95)
    )
    parser.add_argument("--minimum-support-values", nargs="+", type=int, default=(2, 3))
    parser.add_argument("--maximum-start-gap-bp", type=int, default=5_120)
    arguments = parser.parse_args()
    result = evaluate_candidate_sensitivity(
        pd.read_csv(arguments.windows),
        pd.read_csv(arguments.candidate_regions),
        correlation_thresholds=tuple(arguments.correlation_thresholds),
        novelty_quantiles=tuple(arguments.novelty_quantiles),
        minimum_support_values=tuple(arguments.minimum_support_values),
        maximum_start_gap_bp=arguments.maximum_start_gap_bp,
    )
    output = write_sensitivity_results(result, arguments.output_dir)
    summary = result["summary"]
    print(
        f"Saved {output} | settings={summary['num_settings']} "
        f"| region_count={summary['candidate_region_count_min']}–"
        f"{summary['candidate_region_count_max']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
