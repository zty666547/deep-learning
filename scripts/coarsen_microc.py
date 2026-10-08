"""Coarsen a Micro-C COOL file for efficient genome-wide scanning."""

from __future__ import annotations

import argparse

from microc_foundation.discovery import coarsen_cool


def main() -> int:
    parser = argparse.ArgumentParser(description="Pool COOL bins by an integer factor.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--factor", type=int, required=True)
    parser.add_argument("--chunksize", type=int, default=10_000_000)
    arguments = parser.parse_args()
    output = coarsen_cool(
        arguments.input,
        arguments.output,
        factor=arguments.factor,
        chunksize=arguments.chunksize,
    )
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
