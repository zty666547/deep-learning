"""Audit the five fixed Task 2 candidates across wild-type and mutant maps."""

import argparse

import pandas as pd

from microc_foundation.condition_candidate_audit import audit_candidate_condition_transfer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wt-rep1", required=True)
    parser.add_argument("--wt-rep2", required=True)
    parser.add_argument("--dstpa-rep1", required=True)
    parser.add_argument("--dstpa-rep2", required=True)
    parser.add_argument("--double-rep1", required=True)
    parser.add_argument("--double-rep2", required=True)
    parser.add_argument("--minimum-offset", type=int, default=5)
    args = parser.parse_args()
    candidates = pd.read_csv(args.candidates)
    audit_candidates = candidates[["candidate_region_id", "chrom", "start", "end"]].copy()
    audit_candidates["stable_candidate_key"] = [
        f"task2-main-v1:{row.chrom}:{int(row.start)}-{int(row.end)}"
        for row in audit_candidates.itertuples(index=False)
    ]
    summary = audit_candidate_condition_transfer(
        {
            "WT": [args.wt_rep1, args.wt_rep2],
            "delta_stpA": [args.dstpa_rep1, args.dstpa_rep2],
            "delta_hns_delta_stpA": [args.double_rep1, args.double_rep2],
        },
        audit_candidates,
        args.output_dir,
        minimum_offset=args.minimum_offset,
    )
    print(f"Audited {summary['num_candidate_regions']} fixed candidate regions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
