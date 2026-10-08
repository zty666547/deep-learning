"""Cluster Task 2 windows and apply the documented novelty rule."""

from __future__ import annotations

import argparse

import pandas as pd

from microc_foundation.clustering import (
    cluster_and_select_candidates,
    write_clustering_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PCA, K-means, known-structure exclusion and candidate merging."
    )
    parser.add_argument("--windows", required=True)
    parser.add_argument("--structures", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-replicate-correlation", type=float, default=0.9)
    parser.add_argument("--novelty-quantile", type=float, default=0.9)
    parser.add_argument("--maximum-start-gap-bp", type=int, default=5_120)
    parser.add_argument("--minimum-support", type=int, default=2)
    arguments = parser.parse_args()

    windows = pd.read_csv(arguments.windows)
    structures = pd.read_csv(arguments.structures)
    result = cluster_and_select_candidates(
        windows,
        structures,
        k_min=arguments.k_min,
        k_max=arguments.k_max,
        seed=arguments.seed,
        minimum_replicate_correlation=arguments.min_replicate_correlation,
        novelty_quantile=arguments.novelty_quantile,
        maximum_start_gap_bp=arguments.maximum_start_gap_bp,
        minimum_support=arguments.minimum_support,
    )
    output = write_clustering_results(result, arguments.output_dir)
    summary = result["summary"]
    print(
        f"Saved {output} | k={summary['selected_num_clusters']} "
        f"| candidate_windows={summary['num_candidate_windows_before_merge']} "
        f"| candidate_regions={summary['num_candidate_regions']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
