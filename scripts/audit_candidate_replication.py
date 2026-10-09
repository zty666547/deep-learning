"""Audit main Task 2 candidates with short-range and distance-trend controls."""

import argparse

import pandas as pd

from microc_foundation.replicate_audit import audit_replicate_consistency


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cool", action="append", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--candidate-regions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-offset", type=int, default=5)
    args = parser.parse_args()
    summary = audit_replicate_consistency(
        args.cool, pd.read_csv(args.windows), pd.read_csv(args.candidate_regions),
        args.output_dir, minimum_offset=args.minimum_offset,
    )
    print(f"Audited {summary['num_windows']} windows / {summary['num_candidate_regions']} candidate regions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
