"""Compare fixed Task 2 representatives to unannotated controls in mutants."""

import argparse

import pandas as pd

from microc_foundation.candidate_background_audit import audit_mutant_candidate_background


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--candidate-regions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dstpa-rep1", required=True)
    parser.add_argument("--dstpa-rep2", required=True)
    parser.add_argument("--double-rep1", required=True)
    parser.add_argument("--double-rep2", required=True)
    parser.add_argument("--minimum-offset", type=int, default=5)
    args = parser.parse_args()
    summary = audit_mutant_candidate_background(
        {
            "delta_stpA": [args.dstpa_rep1, args.dstpa_rep2],
            "delta_hns_delta_stpA": [args.double_rep1, args.double_rep2],
        },
        pd.read_csv(args.windows),
        pd.read_csv(args.candidate_regions),
        args.output_dir,
        minimum_offset=args.minimum_offset,
    )
    print(f"Audited {summary['num_candidate_representatives']} candidates and "
          f"{summary['num_unannotated_background']} background windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
