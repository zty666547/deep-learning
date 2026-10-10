"""Audit fixed Task 2 candidates at locally pooled 80, 160 and 320 bp."""

import argparse

import pandas as pd

from microc_foundation.candidate_resolution_audit import audit_candidate_resolution_robustness


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
    parser.add_argument("--distance-threshold", type=int, default=800)
    args = parser.parse_args()
    candidates = pd.read_csv(args.candidates)
    audit_candidates = candidates[["candidate_region_id", "chrom", "start", "end"]].copy()
    audit_candidates["stable_candidate_key"] = [
        f"task2-main-v1:{row.chrom}:{int(row.start)}-{int(row.end)}"
        for row in audit_candidates.itertuples(index=False)
    ]
    summary = audit_candidate_resolution_robustness(
        {
            "WT": [args.wt_rep1, args.wt_rep2],
            "delta_stpA": [args.dstpa_rep1, args.dstpa_rep2],
            "delta_hns_delta_stpA": [args.double_rep1, args.double_rep2],
        },
        audit_candidates,
        args.output_dir,
        distance_threshold_bp=args.distance_threshold,
    )
    print(f"Audited {summary['num_candidate_regions']} fixed candidates at "
          f"{summary['requested_resolutions_bp']} bp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
