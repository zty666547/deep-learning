"""Rebuild all frozen three-condition tracks and 465 tiles from six raw matrices."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

_HELPER_PATH = Path(__file__).with_name("reproduce_task2.py")
_SPEC = importlib.util.spec_from_file_location("task3_reproduction_helpers", _HELPER_PATH)
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)


def normalize_source_paths(metadata: dict) -> dict:
    """Only remove directories; keep ordered source basenames and identities."""
    result = copy.deepcopy(metadata)
    for source in result["sources"]:
        source["path"] = Path(source["path"]).name
    return result


def validate_manifest(manifest: dict) -> None:
    length = manifest["length_bp"]
    tile_bp = manifest["tile_bp"]
    if length <= 0 or tile_bp <= 0 or manifest["num_tiles"] != len(manifest["tiles"]):
        raise ValueError("invalid tile count/length")
    expected = [(start, min(start + tile_bp, length)) for start in range(0, length, tile_bp)]
    actual = [(tile["start"], tile["end"]) for tile in manifest["tiles"]]
    if expected != actual:
        raise ValueError("tile coordinates/order do not cover the chromosome exactly")
    for tile in manifest["tiles"]:
        name = f"tile_{tile['start']:07d}_{tile['end']:07d}.png"
        if tile["file"] != name or tile["chrom"] != manifest["chrom"]:
            raise ValueError("tile name/chromosome differs from coordinates")


def compare_tile_pixels(actual_dir: Path, reference_dir: Path, manifest: dict) -> dict:
    validate_manifest(manifest)
    expected_files = {tile["file"] for tile in manifest["tiles"]}
    if ({p.name for p in actual_dir.glob("*.png")} != expected_files
            or {p.name for p in reference_dir.glob("*.png")} != expected_files):
        raise ValueError("PNG inventory differs from manifest")
    size = None
    for index, tile in enumerate(manifest["tiles"], start=1):
        with Image.open(actual_dir / tile["file"]) as actual, Image.open(reference_dir / tile["file"]) as reference:
            actual.load()
            reference.load()
            if actual.size != reference.size or (size is not None and actual.size != size):
                raise ValueError(f"PNG dimensions differ: {tile['file']}")
            size = actual.size
            if not np.array_equal(np.asarray(actual.convert("RGBA")), np.asarray(reference.convert("RGBA"))):
                raise ValueError(f"rendered RGBA pixels differ: {tile['file']}")
        if index % 100 == 0:
            print(f"Compared {index} decoded tile images", flush=True)
    return {"images_checked": len(manifest["tiles"]), "dimensions": list(size),
            "decoded_rgba_pixels_exact": True, "max_channel_absolute_error": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("wt-rep1", "wt-rep2", "dstpa-rep1", "dstpa-rep2", "double-rep1", "double-rep2",
                 "workbook", "gff", "reference-coarsened-dir", "reference-dir", "output-dir", "source-revision"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    raw_conditions = {"WT 37C": [args.wt_rep1, args.wt_rep2],
                      "ΔstpA": [args.dstpa_rep1, args.dstpa_rep2],
                      "ΔhnsΔstpA": [args.double_rep1, args.double_rep2]}
    raw_paths = [p for paths in raw_conditions.values() for p in paths]
    if len({Path(p).resolve() for p in raw_paths}) != 6:
        parser.error("six distinct raw replicate files required")
    output = Path(args.output_dir).resolve()
    _HELPERS.prepare_output(output)
    started = time.monotonic()
    receipt = {"protocol": "task3-raw-isolated-reproduction-v1", "status": "running",
               "scope": "six_raw_coarsening_tracks_annotations_and_all_tiles",
               "source_revision_label": args.source_revision,
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "full_course_reproduction": False, "biological_validation": False,
               "comparisons": {}, "input_sha256": {}}
    stage = "environment"
    try:
        receipt["environment"] = _HELPERS.verify_environment()
        from microc_foundation import read_structures_excel, write_structures_csv
        from microc_foundation.condition_tracks import (
            build_condition_tracks,
            render_condition_tiles,
        )
        from microc_foundation.discovery import coarsen_cool
        from microc_foundation.genes import read_genes_gff
        from microc_foundation.io import open_cooler

        reference = Path(args.reference_dir)
        pooled_reference = Path(args.reference_coarsened_dir)
        original_manifest = json.loads((reference / "tiles/manifest.json").read_text())
        validate_manifest(original_manifest)
        stage = "inputs"
        print("Hashing frozen inputs/references and rebuilding annotations", flush=True)
        inputs = [Path(args.workbook), Path(args.gff), *map(Path, raw_paths),
                  reference / "condition_tracks.csv", reference / "provenance.json",
                  reference / "tiles/manifest.json",
                  *[reference / "tiles" / tile["file"] for tile in original_manifest["tiles"]]]
        receipt["input_sha256"] = {str(p.resolve()): _HELPERS.sha256_file(p) for p in inputs}
        receipt["runner_sha256"] = _HELPERS.sha256_file(Path(__file__))
        receipt["helper_sha256"] = _HELPERS.sha256_file(_HELPER_PATH)
        structures = read_structures_excel(args.workbook, chrom_map={"MG1655": "NC_000913.3"})
        if structures.structure_type.value_counts().to_dict() != {"CHIN": 250, "OPCID": 68, "CHID": 26}:
            raise ValueError("requires frozen course annotations")
        write_structures_csv(structures, str(output / "structures.csv"))
        conditions = {}
        pooled = output / "coarsened"
        pooled.mkdir()
        for condition, paths in raw_conditions.items():
            converted = []
            for index, raw in enumerate(paths, start=1):
                stage = f"coarsen_{condition}_rep{index}"
                print(f"Rebuilding {condition} rep{index} 10→160 bp (no cache)", flush=True)
                with open_cooler(raw) as source:
                    if (source.binsize != 10 or source.chromnames != ["NC_000913.3"]
                            or int(source.chromsizes["NC_000913.3"]) != 4_641_652):
                        raise ValueError("source must be frozen 10 bp NC_000913.3")
                    expected_sum = int(source.info["sum"])
                name = Path(raw).name.removesuffix(".gz").replace(".mapq_30.10.cool", ".mapq_30.160.cool")
                if name == Path(raw).name.removesuffix(".gz"):
                    raise ValueError("unexpected source filename")
                target, old = pooled / name, pooled_reference / name
                if target.exists():
                    raise ValueError("coarsened output must be new; cache prohibited")
                receipt["input_sha256"][str(old.resolve())] = _HELPERS.sha256_file(old)
                coarsen_cool(raw, str(target), factor=16, chunksize=10_000_000)
                stage = f"compare_{condition}_rep{index}"
                receipt["comparisons"][f"{condition}_rep{index}"] = _HELPERS.compare_coolers(target, old, expected_sum)
                converted.append(str(target))
            conditions[condition] = converted
        stage = "tracks_and_genes"
        print("Recomputing circular CPM tracks, equal means and gene layers", flush=True)
        tracks, provenance = build_condition_tracks(conditions, band_bp=10_000)
        genes = read_genes_gff(args.gff, {"NC_000913.3": 4_641_652})
        genes.to_csv(output / "genes.csv", index=False)
        tracks.to_csv(output / "condition_tracks.csv", index=False)
        if len(tracks) != 29011 or int(genes.gene_id.nunique()) != 4506:
            raise ValueError("track or gene counts differ from frozen reference")
        provenance.update(gene_count=int(genes.gene_id.nunique()),
                          structure_counts={str(k): int(v) for k, v in structures.structure_type.value_counts().items()},
                          tile_bp=10_000, matrix_resolution_bp=160)
        (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
        stage = "compare_tracks_and_provenance"
        receipt["comparisons"]["tracks"] = _HELPERS.compare_tables(output / "condition_tracks.csv", reference / "condition_tracks.csv")
        original_provenance = json.loads((reference / "provenance.json").read_text())
        receipt["comparisons"]["provenance_max_float_error"] = _HELPERS.compare_json(
            normalize_source_paths(provenance), normalize_source_paths(original_provenance))
        stage = "render_all_tiles"
        print("Rendering all frozen 465 four-panel tiles", flush=True)
        manifest_path = render_condition_tiles(tracks, genes, structures, provenance, str(output / "tiles"), tile_bp=10_000)
        manifest = json.loads(manifest_path.read_text())
        stage = "compare_manifest_and_pixels"
        validate_manifest(manifest)
        receipt["comparisons"]["manifest_max_float_error"] = _HELPERS.compare_json(
            normalize_source_paths(manifest), normalize_source_paths(original_manifest))
        receipt["comparisons"]["tile_pixels"] = compare_tile_pixels(output / "tiles", reference / "tiles", manifest)
        receipt["result"] = {"conditions": provenance["conditions"], "replicates": 6,
                             "bins": len(tracks), "gene_count": provenance["gene_count"],
                             "tiles": manifest["num_tiles"], "length_bp": manifest["length_bp"],
                             "last_tile": manifest["tiles"][-1]}
        receipt["output_sha256"] = {str(p.relative_to(output)): _HELPERS.sha256_file(p)
                                    for p in sorted(output.rglob("*")) if p.is_file()}
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(status="failed", failed_stage=stage, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Task 3 isolated reproduction passed: {output / 'receipt.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
