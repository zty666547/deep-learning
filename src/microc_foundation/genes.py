"""Read RefSeq GFF3 genes as zero-based, half-open genomic intervals."""

from __future__ import annotations

import gzip
from pathlib import Path
from urllib.parse import unquote

import pandas as pd


def _attributes(text: str) -> dict[str, str]:
    result = {}
    for item in text.split(";"):
        if item and item != ".":
            key, value = item.split("=", 1)
            if key in result:
                raise ValueError("duplicate GFF attribute")
            result[key] = unquote(value)
    return result


def read_genes_gff(path: str, chromsizes: dict[str, int]) -> pd.DataFrame:
    """Validate reference lengths and split explicitly circular origin-crossing genes.

    Only ``gene`` features are used, not CDS/exon duplicates. Input coordinates
    are 1-based closed; output intervals are 0-based half-open. Unknown reference
    sequences fail rather than silently aliasing a different assembly.
    """
    if not chromsizes or any(length <= 0 for length in chromsizes.values()):
        raise ValueError("positive reference lengths required")
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    declarations, circular, features = {}, set(), []
    with opener(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("##FASTA"):
                break
            if line.startswith("##sequence-region "):
                _, chrom, first, last = line.strip().split()
                if int(first) != 1 or chrom in declarations:
                    raise ValueError("invalid/duplicate sequence-region declaration")
                declarations[chrom] = int(last)
            elif line.startswith("#") or not line.strip():
                continue
            else:
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 9:
                    raise ValueError("GFF feature must contain 9 columns")
                chrom, _, kind, start, end, _, strand, _, text = fields
                if chrom not in chromsizes:
                    raise ValueError(f"unknown GFF chromosome: {chrom}")
                if kind == "region" and _attributes(text).get("Is_circular") == "true":
                    circular.add(chrom)
                if kind == "gene":
                    features.append((chrom, int(start) - 1, int(end), strand, _attributes(text)))
    if declarations != chromsizes:
        raise ValueError("GFF sequence-region lengths do not match COOL reference")
    rows, seen = [], set()
    for chrom, start, end, strand, attrs in features:
        length = chromsizes[chrom]
        identity = attrs.get("ID")
        if not identity or start < 0 or start >= length or end <= start or end - start > length:
            raise ValueError("invalid gene ID or interval")
        if strand not in ("+", "-", ".", "?"):
            raise ValueError("invalid GFF strand")
        if (identity, chrom, start, end) in seen:
            raise ValueError("duplicate gene interval")
        seen.add((identity, chrom, start, end))
        if end > length and chrom not in circular:
            raise ValueError("origin-crossing gene requires explicitly circular reference")
        pieces = [(start, min(end, length))]
        if end > length:
            pieces.append((0, end - length))
        for part, (left, right) in enumerate(pieces, 1):
            rows.append({"chrom": chrom, "start": left, "end": right,
                         "gene_id": identity, "gene_name": attrs.get("Name",
                         attrs.get("locus_tag", identity)), "strand": strand, "part": part})
    if not rows:
        raise ValueError("no gene features found")
    return pd.DataFrame(rows).sort_values(["chrom", "start", "end"]).reset_index(drop=True)
