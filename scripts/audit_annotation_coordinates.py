"""Audit workbook structure coordinates against the source 10 bp COOL grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microc_foundation.annotations import read_structures_excel
from microc_foundation.coordinate_audit import audit_annotation_bin_alignment
from microc_foundation.io import open_cooler


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare supplementary structure intervals with COOL bin boundaries."
    )
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--cool", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    structures = read_structures_excel(
        arguments.workbook, chrom_map={"MG1655": "NC_000913.3"}
    )
    with open_cooler(arguments.cool) as contact_map:
        resolution = int(contact_map.binsize)
        if resolution != 10:
            raise ValueError(f"expected the source 10 bp map, got {resolution} bp")
        cool_chromosomes = set(contact_map.chromsizes.index.astype(str))
        if not set(structures["chrom"].astype(str)).issubset(cool_chromosomes):
            raise ValueError("annotation chromosomes do not match the COOL reference")
        summary = audit_annotation_bin_alignment(
            structures,
            contact_map.bins()[:],
            bin_size_bp=resolution,
        )
        summary.update(
            {
                "workbook": str(Path(arguments.workbook)),
                "cool": str(Path(arguments.cool)),
                "cool_reference": str(contact_map.chromsizes.index[0]),
                "cool_chromosome_length": int(contact_map.chromsizes.iloc[0]),
                "cool_resolution_bp": resolution,
            }
        )
    output = Path(arguments.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"annotations={summary['annotation_count']} "
        f"direct_bin_aligned={summary['both_direct_boundaries_count']} "
        f"one_based_adjusted_starts={summary['one_based_adjusted_start_boundary_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
