"""Convert the supplied annotation workbook into normalized structures.csv."""

from __future__ import annotations

import argparse

from microc_foundation import read_structures_excel, write_structures_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract OPCID, CHIN and CHID records from the annotation workbook."
    )
    parser.add_argument("--input", required=True, help="Input .xlsx workbook")
    parser.add_argument("--output", required=True, help="Output structures.csv")
    parser.add_argument(
        "--chrom-map",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help="Chromosome-name mapping; repeat the option for multiple names",
    )
    arguments = parser.parse_args()

    mapping: dict[str, str] = {}
    for item in arguments.chrom_map:
        source, separator, target = item.partition("=")
        if not separator or not source.strip() or not target.strip():
            parser.error("--chrom-map must use SOURCE=TARGET")
        mapping[source.strip()] = target.strip()

    structures = read_structures_excel(arguments.input, chrom_map=mapping)
    output = write_structures_csv(structures, arguments.output)
    counts = structures["structure_type"].value_counts().sort_index().to_dict()
    print(f"Saved {output} | rows={len(structures)} | classes={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
