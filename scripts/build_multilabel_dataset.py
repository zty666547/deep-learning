"""Build the conflict-aware region-level Task 1 dataset."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from microc_foundation.datasets import build_multilabel_region_dataset
from microc_foundation.normalization import SUPPORTED_METHODS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract pooled Micro-C windows for multi-label regions."
    )
    parser.add_argument("--cool", action="append", required=True)
    parser.add_argument("--regions", required=True, help="region_groups.csv from audit")
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-size-bp", type=int, default=20_480)
    parser.add_argument("--pool-factor", type=int, default=16)
    parser.add_argument("--pooling", choices=("sum", "mean"), default="sum")
    parser.add_argument("--normalization", choices=SUPPORTED_METHODS, default="log1p")
    parser.add_argument("--balanced", action="store_true")
    arguments = parser.parse_args()

    regions = pd.read_csv(arguments.regions)
    output = build_multilabel_region_dataset(
        arguments.cool,
        regions,
        arguments.output,
        window_size_bp=arguments.window_size_bp,
        pool_factor=arguments.pool_factor,
        pooling=arguments.pooling,
        normalization=arguments.normalization,
        balance=arguments.balanced,
    )
    with np.load(output) as dataset:
        print(
            f"Saved {output} | shape={dataset['matrices'].shape} "
            f"| labels={dataset['class_names'].tolist()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
