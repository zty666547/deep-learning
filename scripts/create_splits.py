"""Create a leakage-aware contiguous genomic split manifest."""

from __future__ import annotations

import argparse

import pandas as pd

from microc_foundation import read_structures_csv
from microc_foundation.splitting import (
    assign_contiguous_splits,
    write_split_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assign structures to contiguous train/validation/test segments."
    )
    parser.add_argument("--structures", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chrom-length", required=True, type=int)
    parser.add_argument("--window-size-bp", type=int, default=20_480)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    arguments = parser.parse_args()

    structures = read_structures_csv(arguments.structures)
    manifest = assign_contiguous_splits(
        structures,
        arguments.chrom_length,
        train_fraction=arguments.train_fraction,
        validation_fraction=arguments.validation_fraction,
        window_size_bp=arguments.window_size_bp,
    )
    output = write_split_manifest(manifest, arguments.output)
    counts = pd.crosstab(manifest["split"], manifest["structure_type"])
    print(f"Saved {output}")
    print(counts.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
