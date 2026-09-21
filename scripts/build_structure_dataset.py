"""Build fixed-shape Task 1 tensors from real COOL replicates."""

from __future__ import annotations

import argparse

import numpy as np

from microc_foundation import read_structures_csv
from microc_foundation.datasets import build_structure_dataset
from microc_foundation.normalization import SUPPORTED_METHODS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract aligned, pooled structure windows into a compressed NPZ."
    )
    parser.add_argument("--cool", action="append", required=True, help="Input COOL path")
    parser.add_argument("--structures", required=True, help="Normalized structures.csv")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--window-size-bp", type=int, default=20_480)
    parser.add_argument("--pool-factor", type=int, default=16)
    parser.add_argument("--pooling", choices=("sum", "mean"), default="sum")
    parser.add_argument(
        "--normalization",
        choices=SUPPORTED_METHODS,
        default="log1p",
    )
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-class-limit", type=int, default=None)
    arguments = parser.parse_args()

    structures = read_structures_csv(arguments.structures)
    output = build_structure_dataset(
        arguments.cool,
        structures,
        arguments.output,
        window_size_bp=arguments.window_size_bp,
        pool_factor=arguments.pool_factor,
        pooling=arguments.pooling,
        normalization=arguments.normalization,
        balance=arguments.balanced,
        limit=arguments.limit,
        per_class_limit=arguments.per_class_limit,
    )
    with np.load(output) as dataset:
        print(
            f"Saved {output} | shape={dataset['matrices'].shape} "
            f"| pooled_bin_size={int(dataset['pooled_bin_size'])} bp"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
