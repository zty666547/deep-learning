"""Rebuild the full foundation from real inputs using an isolated installed wheel.

No model training or biological result is generated here. This is explicitly
foundation-only reproduction, not complete-course reproduction certification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_datasets(actual_path: Path, reference_path: Path) -> dict:
    """Require identical field sets, identities, metadata and numerical arrays."""
    with np.load(actual_path, allow_pickle=False) as actual, np.load(reference_path, allow_pickle=False) as reference:
        if set(actual.files) != set(reference.files):
            raise ValueError("dataset field sets differ from the reference")
        for key in sorted(actual.files):
            if not np.array_equal(actual[key], reference[key]):
                raise ValueError(f"reproduced dataset differs from reference field: {key}")
        values = actual["matrices"]
        if not np.isfinite(values).all():
            raise ValueError("reproduced matrices contain non-finite values")
        return {"fields_checked": sorted(actual.files), "tensor_shape": list(values.shape),
                "numerical_equality": "exact", "max_matrix_absolute_error": 0.0}


def verify_installed_environment() -> dict:
    import microc_foundation

    module = Path(microc_foundation.__file__).resolve()
    if not sys.flags.isolated or sys.prefix == sys.base_prefix:
        raise ValueError("run with a separate virtual environment's python -I")
    try:
        module.relative_to(Path(sys.prefix).resolve())
    except ValueError as error:
        raise ValueError("package was imported outside the clean virtual environment") from error
    distribution = importlib.metadata.distribution("microc-foundation")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("dir_info", {}).get("editable", False):
        raise ValueError("editable project installation is not allowed for this check")
    return {"python": sys.version, "platform": platform.platform(),
            "executable": sys.executable, "prefix": sys.prefix,
            "isolated": True, "package_file": str(module), "editable": False,
            "installation": direct_url,
            "packages": dict(sorted((dist.metadata["Name"], dist.version)
                                    for dist in importlib.metadata.distributions()))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--cool", action="append", required=True)
    parser.add_argument("--reference-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True, help="Label for the exported wheel source commit")
    args = parser.parse_args()
    if len(args.cool) != 2 or len({Path(value).resolve() for value in args.cool}) != 2:
        parser.error("provide two distinct WT replicate paths")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be new or empty; existing evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    receipt = {"protocol": "foundation-isolated-wheel-v1", "scope": "foundation_only",
               "source_revision_label": args.source_revision,
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "full_course_reproduction": False, "status": "running"}
    stage = "environment"
    try:
        receipt["environment"] = verify_installed_environment()
        from microc_foundation import read_structures_excel, write_structures_csv
        from microc_foundation.datasets import build_structure_dataset
        from microc_foundation.io import open_cooler
        from microc_foundation.splitting import assign_contiguous_splits, write_split_manifest
        from microc_foundation.visualization import render_heatmap
        from microc_foundation.windows import GenomicWindow

        stage = "input_hashes"
        inputs = [Path(args.workbook), *[Path(value) for value in args.cool], Path(args.reference_dataset)]
        receipt["input_sha256"] = {str(path.resolve()): sha256_file(path) for path in inputs}
        receipt["runner_sha256"] = sha256_file(Path(__file__))
        stage = "annotations_and_splits"
        print("Rebuilding annotations and contiguous splits from real workbook", flush=True)
        structures = read_structures_excel(args.workbook, chrom_map={"MG1655": "NC_000913.3"})
        counts = structures.structure_type.value_counts().to_dict()
        if counts != {"CHIN": 250, "OPCID": 68, "CHID": 26}:
            raise ValueError("workbook does not match the frozen 344-annotation dataset")
        for path in args.cool:
            with open_cooler(path) as cool:
                if (cool.binsize != 10 or cool.chromnames != ["NC_000913.3"]
                        or int(cool.chromsizes["NC_000913.3"]) != 4_641_652):
                    raise ValueError("WT inputs must be 10 bp NC_000913.3 of the frozen length")
        write_structures_csv(structures, str(output / "structures.csv"))
        split = assign_contiguous_splits(structures, 4_641_652, window_size_bp=20_480)
        write_split_manifest(split, str(output / "structures_split.csv"))
        receipt["annotation_counts"] = {key: int(value) for key, value in counts.items()}
        receipt["split_counts"] = {key: int(value) for key, value in split.split.value_counts().items()}
        stage = "full_window_extraction"
        print("Extracting all 344 real windows across both WT replicates", flush=True)
        dataset_path = build_structure_dataset(
            args.cool, split, str(output / "windows.npz"), window_size_bp=20_480,
            pool_factor=16, pooling="sum", normalization="log1p", balance=False,
        )
        stage = "reference_comparison"
        receipt["comparison"] = compare_datasets(dataset_path, Path(args.reference_dataset))
        stage = "heatmap"
        with np.load(dataset_path, allow_pickle=False) as data:
            region = GenomicWindow(str(data["chrom"][0]), int(data["window_start"][0]), int(data["window_end"][0]))
            render_heatmap(data["matrices"][0, 0], region, str(output / "first_window.png"),
                           title=f"Fixed first window: {data['structure_id'][0]} | WT rep1 | log1p pooled counts")
            receipt["heatmap_structure_id"] = str(data["structure_id"][0])
        receipt["output_sha256"] = {path.name: sha256_file(path)
                                    for path in sorted(output.iterdir()) if path.is_file()}
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(status="failed", failed_stage=stage,
                       error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Foundation-only reproduction passed: {output / 'receipt.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
