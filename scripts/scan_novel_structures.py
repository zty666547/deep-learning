"""Scan real Micro-C replicates for reproducible candidate regions."""

from __future__ import annotations

import argparse

from microc_foundation.annotations import read_structures_csv
from microc_foundation.novelty import scan_genome_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan full-genome candidate windows.")
    parser.add_argument("--cool", action="append", required=True)
    parser.add_argument("--structures", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-size-bp", type=int, default=20_480)
    parser.add_argument("--step-bp", type=int, default=20_480)
    parser.add_argument("--pool-factor", type=int, default=16)
    parser.add_argument("--normalization", default="log1p")
    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument("--min-total-contacts", type=float, default=0.0)
    parser.add_argument("--max-missing-fraction", type=float, default=0.5)
    parser.add_argument("--min-replicate-correlation", type=float, default=0.0)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=12)
    arguments = parser.parse_args()
    structures = read_structures_csv(arguments.structures)
    output = scan_genome_candidates(
        arguments.cool,
        structures,
        arguments.output_dir,
        window_size_bp=arguments.window_size_bp,
        step_bp=arguments.step_bp,
        pool_factor=arguments.pool_factor,
        normalization=arguments.normalization,
        num_clusters=arguments.num_clusters,
        min_total_contacts=arguments.min_total_contacts,
        max_missing_fraction=arguments.max_missing_fraction,
        min_replicate_correlation=arguments.min_replicate_correlation,
        min_cluster_size=arguments.min_cluster_size,
        top_k=arguments.top_k,
    )
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
