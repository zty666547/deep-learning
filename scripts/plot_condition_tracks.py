"""Generate fixed-interval four-track plots for WT and two mutant conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microc_foundation.annotations import read_structures_csv
from microc_foundation.condition_tracks import build_condition_tracks, render_condition_tiles
from microc_foundation.discovery import coarsen_cool
from microc_foundation.genes import read_genes_gff
from microc_foundation.io import open_cooler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wt-rep1", required=True)
    parser.add_argument("--wt-rep2", required=True)
    parser.add_argument("--dstpa-rep1", required=True)
    parser.add_argument("--dstpa-rep2", required=True)
    parser.add_argument("--double-rep1", required=True)
    parser.add_argument("--double-rep2", required=True)
    parser.add_argument("--gff", required=True)
    parser.add_argument("--structures", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--coarsened-dir", required=True)
    parser.add_argument("--band-bp", type=int, default=10_000)
    parser.add_argument("--tile-bp", type=int, default=10_000)
    args = parser.parse_args()

    raw_conditions = {
        "WT 37C": [args.wt_rep1, args.wt_rep2],
        "ΔstpA": [args.dstpa_rep1, args.dstpa_rep2],
        "ΔhnsΔstpA": [args.double_rep1, args.double_rep2],
    }
    coarsened_dir = Path(args.coarsened_dir)
    coarsened_dir.mkdir(parents=True, exist_ok=True)
    conditions = {}
    for condition, source_paths in raw_conditions.items():
        converted_paths = []
        for source_path in source_paths:
            source_name = Path(source_path).name
            with open_cooler(source_path) as source:
                if (source.binsize != 10 or source.chromnames != ["NC_000913.3"] or
                        int(source.chromsizes["NC_000913.3"]) != 4_641_652):
                    raise ValueError(f"source must be 10 bp NC_000913.3: {source_name}")
            if source_name.endswith(".mapq_30.10.cool.gz"):
                target_name = source_name.replace(".mapq_30.10.cool.gz", ".mapq_30.160.cool")
            elif source_name.endswith(".mapq_30.10.cool"):
                target_name = source_name.replace(".mapq_30.10.cool", ".mapq_30.160.cool")
            else:
                raise ValueError(f"unexpected 10 bp source filename: {source_name}")
            converted_paths.append(str(coarsen_cool(
                source_path, str(coarsened_dir / target_name), factor=16
            )))
        conditions[condition] = converted_paths
    tracks, provenance = build_condition_tracks(conditions, band_bp=args.band_bp)
    chromsizes = {str(chrom): int(end) for chrom, end in
                  tracks.groupby("chrom", sort=False, observed=True).end.max().items()}
    genes = read_genes_gff(args.gff, chromsizes)
    structures = read_structures_csv(args.structures)
    if set(structures.chrom.astype(str)) != set(chromsizes):
        raise ValueError("structure annotations do not match contact-map reference")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(output / "condition_tracks.csv", index=False)
    provenance["gene_count"] = int(genes.gene_id.nunique())
    provenance["structure_counts"] = {
        str(key): int(value) for key, value in structures.structure_type.value_counts().items()
    }
    provenance["tile_bp"] = args.tile_bp
    provenance["matrix_resolution_bp"] = int(tracks.end.iloc[0] - tracks.start.iloc[0])
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_path = render_condition_tiles(
        tracks, genes, structures, provenance, str(output / "tiles"), tile_bp=args.tile_bp
    )
    print(f"Saved {len(json.loads(manifest_path.read_text())['tiles'])} tiles to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
