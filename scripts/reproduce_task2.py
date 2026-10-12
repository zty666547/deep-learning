"""Rebuild the frozen Task 2 main pipeline from raw WT data in an installed venv."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RTOL = 1e-12
ATOL = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_environment() -> dict:
    import microc_foundation

    module = Path(microc_foundation.__file__).resolve()
    distribution = importlib.metadata.distribution("microc-foundation")
    installation = json.loads(distribution.read_text("direct_url.json") or "{}")
    if (not sys.flags.isolated or sys.prefix == sys.base_prefix
            or not module.is_relative_to(Path(sys.prefix).resolve())
            or installation.get("dir_info", {}).get("editable", False)):
        raise ValueError("use separate venv python -I with a non-editable installed package")
    return {"python": sys.version, "platform": platform.platform(),
            "package_file": str(module), "installation": installation,
            "isolated": True, "editable": False,
            "packages": dict(sorted((dist.metadata["Name"], dist.version)
                                    for dist in importlib.metadata.distributions()))}


def prepare_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("output directory must be new or empty; never overwrite evidence")
    path.mkdir(parents=True, exist_ok=True)


def compare_tables(actual_path: Path, reference_path: Path) -> dict:
    actual = pd.read_csv(actual_path, keep_default_na=False)
    reference = pd.read_csv(reference_path, keep_default_na=False)
    if list(actual.columns) != list(reference.columns) or actual.shape != reference.shape:
        raise ValueError(f"table schema/order/shape differs: {actual_path.name}")
    maximum_error = 0.0
    for name in actual.columns:
        left, right = actual[name], reference[name]
        if (pd.api.types.is_float_dtype(left.dtype)
                and pd.api.types.is_float_dtype(right.dtype)):
            x, y = left.to_numpy(), right.to_numpy()
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"non-finite table values: {name}")
            if not np.allclose(x, y, rtol=RTOL, atol=ATOL):
                raise ValueError(f"table numerical values differ: {name}")
            maximum_error = max(maximum_error, float(np.max(np.abs(x - y), initial=0)))
        elif not left.equals(right):
            raise ValueError(f"table identities/discrete values differ: {name}")
    return {"rows": len(actual), "columns": len(actual.columns),
            "identities_discrete_and_order_exact": True,
            "max_float_absolute_error": maximum_error, "rtol": RTOL, "atol": ATOL}


def compare_json(actual, reference, location: str = "root") -> float:
    if isinstance(reference, dict):
        if not isinstance(actual, dict) or set(actual) != set(reference):
            raise ValueError(f"JSON key mismatch: {location}")
        return max((compare_json(actual[key], reference[key], f"{location}.{key}")
                    for key in reference), default=0.0)
    if isinstance(reference, list):
        if not isinstance(actual, list) or len(actual) != len(reference):
            raise ValueError(f"JSON list mismatch: {location}")
        return max((compare_json(x, y, f"{location}[{index}]")
                    for index, (x, y) in enumerate(zip(actual, reference))), default=0.0)
    if isinstance(reference, float):
        if (isinstance(actual, bool) or not isinstance(actual, (float, int))
                or not math.isfinite(actual) or not math.isfinite(reference)
                or not math.isclose(actual, reference, rel_tol=RTOL, abs_tol=ATOL)):
            raise ValueError(f"JSON numerical mismatch: {location}")
        return abs(actual - reference)
    if type(actual) is not type(reference) or actual != reference:
        raise ValueError(f"JSON discrete value mismatch: {location}")
    return 0.0


def compare_coolers(actual_path: Path, reference_path: Path, expected_sum: int) -> dict:
    from microc_foundation.io import open_cooler

    with open_cooler(str(actual_path)) as actual, open_cooler(str(reference_path)) as reference:
        for key in ("bin-size", "nbins", "nnz", "storage-mode", "sum"):
            if actual.info[key] != reference.info[key]:
                raise ValueError(f"pooled COOL metadata differs: {key}")
        if int(actual.info["sum"]) != expected_sum:
            raise ValueError("pooled COOL does not conserve raw upper-triangle counts")
        pd.testing.assert_series_equal(actual.chromsizes, reference.chromsizes)
        pd.testing.assert_frame_equal(actual.bins()[:], reference.bins()[:], check_exact=True)
        nnz = int(actual.info["nnz"])
        for start in range(0, nnz, 1_000_000):
            end = min(start + 1_000_000, nnz)
            pd.testing.assert_frame_equal(actual.pixels()[start:end],
                                          reference.pixels()[start:end], check_exact=True)
        return {"bin_size": int(actual.binsize), "bins": int(actual.info["nbins"]),
                "pixels_checked": nnz, "count_sum": expected_sum,
                "bin_coordinates_and_all_pixels_exact": True, "count_conservation": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--cool", action="append", required=True)
    parser.add_argument("--reference-coarsened", action="append", required=True)
    parser.add_argument("--reference-scan", required=True)
    parser.add_argument("--reference-clustering", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if (len(args.cool) != 2 or len(args.reference_coarsened) != 2
            or len({Path(p).resolve() for p in args.cool}) != 2
            or len({Path(p).resolve() for p in args.reference_coarsened}) != 2):
        parser.error("provide two distinct raw WT replicates and two ordered pooled references")
    output = Path(args.output_dir).resolve()
    prepare_output(output)
    started = time.monotonic()
    receipt = {"protocol": "task2-main-v1-isolated-reproduction-v1",
               "scope": "annotations_coarsening_scan_clustering_candidates",
               "source_revision_label": args.source_revision,
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "full_course_reproduction": False, "biological_validation": False,
               "status": "running", "comparisons": {}, "input_sha256": {}}
    stage = "environment"
    try:
        receipt["environment"] = verify_environment()
        from microc_foundation import read_structures_excel, write_structures_csv
        from microc_foundation.clustering import (
            cluster_and_select_candidates,
            write_clustering_results,
        )
        from microc_foundation.discovery import (
            coarsen_cool,
            scan_replicate_windows,
            write_window_scan,
        )
        from microc_foundation.io import open_cooler

        stage = "inputs_and_annotations"
        print("Checking installed environment and rebuilding workbook annotations", flush=True)
        reference_scan, reference_cluster = Path(args.reference_scan), Path(args.reference_clustering)
        inputs = [Path(args.workbook), *map(Path, args.cool), *map(Path, args.reference_coarsened),
                  reference_scan / "candidate_windows.csv", reference_scan / "summary.json",
                  *[reference_cluster / name for name in
                    ("window_clusters.csv", "candidate_windows.csv", "candidate_regions.csv", "summary.json")]]
        receipt["input_sha256"] = {str(path.resolve()): sha256_file(path) for path in inputs}
        receipt["runner_sha256"] = sha256_file(Path(__file__))
        structures = read_structures_excel(args.workbook, chrom_map={"MG1655": "NC_000913.3"})
        if structures.structure_type.value_counts().to_dict() != {"CHIN": 250, "OPCID": 68, "CHID": 26}:
            raise ValueError("requires frozen 344 course annotations")
        write_structures_csv(structures, str(output / "structures.csv"))
        pooled = []
        for index, (raw, reference) in enumerate(zip(args.cool, args.reference_coarsened), start=1):
            stage = f"coarsen_rep{index}"
            print(f"Rebuilding WT rep{index} 10→160 bp from raw COOL (no cache)", flush=True)
            with open_cooler(raw) as source:
                if (source.binsize != 10 or source.chromnames != ["NC_000913.3"]
                        or int(source.chromsizes["NC_000913.3"]) != 4_641_652):
                    raise ValueError("requires frozen 10 bp WT reference")
                expected_sum = int(source.info["sum"])
            target = output / f"WT_rep{index}.160bp.cool"
            if target.exists():
                raise ValueError("fresh coarsened output must not exist")
            coarsen_cool(raw, str(target), factor=16, chunksize=10_000_000)
            stage = f"compare_coarsen_rep{index}"
            receipt["comparisons"][f"coarsened_rep{index}"] = compare_coolers(target, Path(reference), expected_sum)
            pooled.append(str(target))
        stage = "scan"
        print("Recomputing all 1807 windows and fixed QC/features", flush=True)
        parameters = {"cool_paths": pooled, "chrom": "NC_000913.3", "window_size_bp": 20480,
                      "stride_bp": 2560, "min_total_contacts": 1.0, "min_nonzero_fraction": 0.01,
                      "min_finite_fraction": 0.99, "max_zero_bin_fraction": 0.03,
                      "max_zero_run_fraction": 0.03}
        rows = scan_replicate_windows(**parameters)
        scan = write_window_scan(rows, str(output / "scan"), parameters=parameters)
        stage = "compare_scan"
        receipt["comparisons"]["scan_table"] = compare_tables(scan / "candidate_windows.csv", reference_scan / "candidate_windows.csv")
        actual_scan = json.loads((scan / "summary.json").read_text())
        original_scan = json.loads((reference_scan / "summary.json").read_text())
        actual_scan["parameters"]["cool_paths"] = ["rep1", "rep2"]
        original_scan["parameters"]["cool_paths"] = ["rep1", "rep2"]
        receipt["comparisons"]["scan_summary_max_float_error"] = compare_json(actual_scan, original_scan)
        stage = "clustering"
        print("Recomputing frozen PCA, k=2–8, novelty and region merging", flush=True)
        result = cluster_and_select_candidates(pd.read_csv(scan / "candidate_windows.csv"), structures,
                                              k_min=2, k_max=8, seed=2026,
                                              minimum_replicate_correlation=0.9, novelty_quantile=0.9,
                                              maximum_start_gap_bp=5120, minimum_support=2)
        cluster = write_clustering_results(result, str(output / "clustering"))
        stage = "compare_clustering"
        for name in ("window_clusters.csv", "candidate_windows.csv", "candidate_regions.csv"):
            receipt["comparisons"][name] = compare_tables(cluster / name, reference_cluster / name)
        original = json.loads((reference_cluster / "summary.json").read_text())
        receipt["comparisons"]["clustering_summary_max_float_error"] = compare_json(result["summary"], original)
        if (len(rows), len(result["windows"]), len(result["candidate_windows"]),
                len(result["candidate_regions"])) != (1807, 1498, 25, 5):
            raise ValueError("output does not match the frozen main-v1 sample counts")
        receipt["result"] = {"scan_windows": len(rows), "quality_windows": result["summary"]["num_quality_windows"],
                             "clusters": result["summary"]["selected_num_clusters"],
                             "candidate_windows": len(result["candidate_windows"]),
                             "candidate_regions": result["candidate_regions"].to_dict(orient="records")}
        receipt["output_sha256"] = {str(p.relative_to(output)): sha256_file(p)
                                    for p in sorted(output.rglob("*")) if p.is_file()}
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(status="failed", failed_stage=stage, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Task 2 main pipeline reproduction passed: {output / 'receipt.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
