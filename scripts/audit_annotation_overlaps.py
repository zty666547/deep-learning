"""Audit cross-class overlaps across all structure annotations."""

from __future__ import annotations

import argparse

from microc_foundation import read_structures_csv
from microc_foundation.overlaps import audit_annotation_overlaps, write_overlap_audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit cross-class interval overlaps and exclusion feasibility."
    )
    parser.add_argument("--structures", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()

    structures = read_structures_csv(arguments.structures)
    audit = audit_annotation_overlaps(structures)
    output = write_overlap_audit(audit, arguments.output_dir)
    summary = audit["summary"]
    print(
        f"Saved {output} | structures={summary['num_structures']} "
        f"| pairs={summary['num_cross_class_overlap_pairs']} "
        f"| conflicted={summary['num_conflicted_structures']} "
        f"| components={summary['num_conflict_components']}"
    )
    print(
        "Exclusion preserves every class in every split: "
        f"{summary['exclusion_preserves_all_classes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
