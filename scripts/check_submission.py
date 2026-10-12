"""Check that the repository contains the minimum reproducible submission evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def validate_super_resolution_evidence(root: Path) -> dict:
    """Check corrected multi-seed artifacts and recompute reported averages."""
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    seeds, runs = summary["seeds"], summary["runs"]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or len(runs) != len(seeds):
        raise ValueError("Task5 requires at least three distinct seeds and matching runs")
    for seed, run in zip(seeds, runs):
        directory = root / f"seed_{seed}"
        stored = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if stored != run or run["seed"] != seed:
            raise ValueError("Task5 run summary disagrees with stored metrics")
        if run.get("intensity_correction") != "log1p(expm1(low_log_counts)/4)" or "7x7" not in run.get("ssim_definition", ""):
            raise ValueError("Task5 requires corrected count scale and local SSIM")
        for filename in ("model.pt", "history.csv", "example_comparison.png"):
            if not (directory / filename).is_file():
                raise ValueError(f"Missing Task5 artifact: {directory / filename}")
        with (directory / "per_sample_metrics.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected = run["sample_counts"]["test"] * run["target_shape"][0] * 2
        keys = {(row["structure_id"], row["replicate_index"], row["method"]) for row in rows}
        if len(rows) != expected or len(keys) != expected:
            raise ValueError("Task5 per-sample metrics have missing or duplicate rows")
        paired_keys = [
            {(row["structure_id"], row["replicate_index"]) for row in rows if row["method"] == method}
            for method in ("bicubic", "cnn")
        ]
        if paired_keys[0] != paired_keys[1]:
            raise ValueError("Task5 method sample identities do not match")
        for method, label in (("bicubic", "bicubic_baseline"), ("cnn", "cnn")):
            selected = [row for row in rows if row["method"] == method]
            if len(selected) * 2 != expected:
                raise ValueError("Task5 methods have mismatched sample counts")
            for metric in ("psnr", "ssim", "mse"):
                values = [float(row[metric]) for row in selected]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("Task5 contains non-finite metrics")
                if not math.isclose(sum(values) / len(values), run[label][metric], rel_tol=1e-5, abs_tol=1e-7):
                    raise ValueError("Task5 per-sample mean disagrees with reported metric")
    for method in ("bicubic_baseline", "cnn"):
        for metric in ("psnr", "ssim", "mse"):
            values = [run[method][metric] for run in runs]
            mean = sum(values) / len(values)
            std = math.sqrt(sum((value - mean)**2 for value in values) / (len(values) - 1))
            aggregate = summary["aggregate"][method][metric]
            if not math.isclose(mean, aggregate["mean"], rel_tol=1e-5, abs_tol=1e-7) or not math.isclose(std, aggregate["std"], rel_tol=1e-5, abs_tol=1e-7):
                raise ValueError("Task5 multi-seed aggregate is inconsistent")
    return summary


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing evidence summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Missing evidence table: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_task3_evidence(root: Path) -> dict:
    """Check current three-condition tracks and every tiled PNG, not old WT scans.

    This checks stored evidence consistency, not image semantics or a raw-data rerun.
    """
    from PIL import Image

    provenance = _read_json(root / "provenance.json")
    manifest = _read_json(root / "tiles/manifest.json")
    conditions = ["WT 37C", "ΔstpA", "ΔhnsΔstpA"]
    length, resolution, tile_bp = 4_641_652, 160, 10_000
    if (provenance.get("conditions") != conditions
            or provenance.get("matrix_resolution_bp") != resolution
            or provenance.get("tile_bp") != tile_bp
            or provenance.get("band_bp") != 10_000
            or provenance.get("circular") is not True
            or provenance.get("replicate_weighting") != "equal within condition"
            or provenance.get("condition_weighting") != "equal across conditions"
            or provenance.get("normalization") != "off-diagonal band row sum / library sum * 1e6"
            or provenance.get("gene_count", 0) <= 0
            or provenance.get("structure_counts") != {"CHIN": 250, "OPCID": 68, "CHID": 26}):
        raise ValueError("Task 3 provenance disagrees with the frozen three-condition protocol")
    if any(manifest.get(key) != value for key, value in provenance.items()):
        raise ValueError("Task 3 tile manifest disagrees with provenance")
    sources = provenance.get("sources", [])
    paths = [str(Path(source["path"]).resolve()) for source in sources]
    if (len(sources) != 6 or len(set(paths)) != 6
            or any(sum(source["condition"] == name for source in sources) != 2 for name in conditions)
            or any(source["resolution_bp"] != resolution or source["library_sum"] <= 0
                   or not Path(source["path"]).is_file() for source in sources)):
        raise ValueError("Task 3 requires six distinct existing replicate files, two per condition")
    rows = _read_csv(root / "condition_tracks.csv")
    expected_starts = list(range(0, length, resolution))
    if len(rows) != len(expected_starts):
        raise ValueError("Task 3 bin track has missing or extra rows")
    for row, start in zip(rows, expected_starts):
        end = min(start + resolution, length)
        if (row["chrom"] != "NC_000913.3" or int(row["start"]) != start
                or int(row["end"]) != end or float(row["center"]) != (start + end) / 2):
            raise ValueError("Task 3 bin coordinates have gaps, overlaps or incorrect endpoints")
        values = [float(row[name]) for name in conditions]
        signals = values + [float(row[f"{name}__replicate_sd"]) for name in conditions]
        mean = float(row["mean_cpm"])
        if (not all(math.isfinite(value) and value >= 0 for value in signals)
                or not math.isfinite(mean)
                or not math.isclose(mean, sum(values) / 3, rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError("Task 3 signals are invalid or across-condition mean is inconsistent")
    expected_tiles = list(range(0, length, tile_bp))
    tiles = manifest.get("tiles", [])
    if (manifest.get("chrom") != "NC_000913.3" or manifest.get("length_bp") != length
            or manifest.get("num_tiles") != len(expected_tiles) or len(tiles) != len(expected_tiles)):
        raise ValueError("Task 3 manifest does not cover the complete frozen chromosome")
    dimensions = None
    for tile, start in zip(tiles, expected_tiles):
        end = min(start + tile_bp, length)
        filename = f"tile_{start:07d}_{end:07d}.png"
        if (tile.get("chrom") != "NC_000913.3" or tile.get("start") != start
                or tile.get("end") != end or tile.get("file") != filename
                or tile.get("gene_segments", -1) < 0 or tile.get("structures", -1) < 0):
            raise ValueError("Task 3 tile coordinates or filenames are inconsistent")
        try:
            with Image.open(root / "tiles" / filename) as picture:
                if picture.format != "PNG" or min(picture.size) <= 0:
                    raise ValueError("invalid PNG format/dimensions")
                if dimensions is None:
                    dimensions = picture.size
                if picture.size != dimensions:
                    raise ValueError("inconsistent PNG dimensions")
                picture.verify()
        except (OSError, ValueError, SyntaxError) as error:
            raise ValueError(f"Task 3 missing or invalid tile PNG: {filename}") from error
    return {"num_conditions": len(conditions), "num_replicates": len(sources),
            "num_bins": len(rows), "num_tiles": len(tiles), "length_bp": length,
            "last_tile": [expected_tiles[-1], length]}


def validate_task2_validation_evidence(output_root: Path) -> dict:
    """Check cross-condition, CPM, resolution, and mutant-background outputs."""
    raw_root = output_root / "task2_condition_transfer"
    cpm_root = output_root / "task2_condition_transfer_cpm"
    raw = _read_json(raw_root / "summary.json")
    cpm = _read_json(cpm_root / "summary.json")
    if raw.get("normalization") != "raw_counts" or cpm.get("normalization") != "library_cpm":
        raise ValueError("Task 2 must preserve raw-count and library-CPM audits separately")
    if raw.get("candidate_keys") != cpm.get("candidate_keys") or len(raw.get("candidate_keys", [])) != 5:
        raise ValueError("Task 2 cross-condition audits must use the same five fixed candidates")
    expected_contrasts = {
        "within:WT", "within:delta_stpA", "within:delta_hns_delta_stpA",
        "WT_vs_delta_stpA", "WT_vs_delta_hns_delta_stpA",
    }
    transfer_candidate_ids = {}
    for label, root in (("raw_counts", raw_root), ("library_cpm", cpm_root)):
        rows = _read_csv(root / "candidate_condition_summary.csv")
        if len(rows) != 25 or {row["condition_contrast"] for row in rows} != expected_contrasts:
            raise ValueError("Task 2 condition summary is incomplete")
        candidate_ids = {row["candidate_region_id"] for row in rows}
        if len(candidate_ids) != 5:
            raise ValueError("Task 2 condition summary must contain five unique candidates")
        transfer_candidate_ids[label] = candidate_ids
    if transfer_candidate_ids["raw_counts"] != transfer_candidate_ids["library_cpm"]:
        raise ValueError("Raw-count and CPM summaries must use identical candidate identities")

    resolution_root = output_root / "task2_resolution_audit"
    resolution = _read_json(resolution_root / "summary.json")
    if resolution.get("requested_resolutions_bp") != [80, 160, 320] or resolution.get("num_candidate_regions") != 5:
        raise ValueError("Task 2 resolution audit must contain all five candidates at 80/160/320 bp")
    resolution_rows = _read_csv(resolution_root / "candidate_resolution_summary.csv")
    base_rows = _read_csv(raw_root / "candidate_condition_summary.csv")
    resolution_160 = {
        (row["candidate_region_id"], row["condition_contrast"]): row
        for row in resolution_rows if int(row["resolution_bp"]) == 160
    }
    if len(resolution_160) != 25:
        raise ValueError("Task 2 160 bp resolution audit is incomplete")
    for row in base_rows:
        key = (row["candidate_region_id"], row["condition_contrast"])
        if key not in resolution_160 or not math.isclose(
            float(row["distance_centered_correlation_median"]),
            float(resolution_160[key]["distance_centered_correlation_median"]),
            rel_tol=1e-8, abs_tol=1e-10,
        ):
            raise ValueError("Local 160 bp pooling disagrees with the existing 160 bp audit")

    background_root = output_root / "task2_mutant_background"
    background = _read_json(background_root / "summary.json")
    if (background.get("num_candidate_representatives") != 5
            or background.get("num_unannotated_background") != 329
            or set(background.get("normalizations", [])) != {"raw_counts", "library_cpm"}):
        raise ValueError("Task 2 mutant-background audit has unexpected sample counts or normalizations")
    group_rows = _read_csv(background_root / "candidate_background_summary.csv")
    if len(group_rows) != 8:
        raise ValueError("Task 2 mutant-background group summary is incomplete")
    values = {
        (row["normalization"], row["condition"], row["group"]): float(
            row["distance_centered_correlation_median"],
        )
        for row in group_rows
    }
    for normalization in ("raw_counts", "library_cpm"):
        for condition in ("delta_stpA", "delta_hns_delta_stpA"):
            candidate = values.get((normalization, condition, "candidate_representative"))
            background_value = values.get((normalization, condition, "unannotated_background"))
            if candidate is None or background_value is None or candidate <= background_value:
                raise ValueError("Candidate representative median must exceed background in all four fixed comparisons")
    return {"num_candidates": 5, "num_resolution_levels": 3,
            "num_background_windows": 329, "normalizations": 2}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate project submission evidence.")
    parser.add_argument("--output-root", default="outputs")
    arguments = parser.parse_args()
    required_docs = (
        "README.md",
        "docs/development-log.md",
        "docs/submission-roadmap.md",
        "docs/report-outline.md",
        "docs/video-script.md",
        "docs/task1-multilabel-results.md",
        "docs/task2-clustering-results.md",
        "docs/task3-visualization-results.md",
        "docs/task5-super-resolution-results.md",
        "docs/quality-review.md",
        "docs/project-walkthrough.md",
        "docs/task2-replication-audit.md",
        "docs/task2-condition-transfer-audit.md",
        "docs/task2-resolution-audit.md",
        "docs/task2-mutant-background-control.md",
        "docs/task2-candidate-registry.csv",
        "docs/task2-cluster-evidence.md",
        "docs/task1-occlusion-audit.md",
    )
    missing = [path for path in required_docs if not Path(path).is_file()]
    if missing:
        raise SystemExit("Missing required documentation: " + ", ".join(missing))
    output_root = Path(arguments.output_root)
    summaries = {
        "task2": output_root / "task2_clustering/summary.json",
        "task2_replication": output_root / "task2_replication_audit/summary.json",
        "task5": output_root / "task5_corrected/summary.json",
    }
    missing_outputs = [str(path) for path in summaries.values() if not path.is_file()]
    if missing_outputs:
        raise SystemExit("Missing generated outputs: " + ", ".join(missing_outputs))
    loaded = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in summaries.items()}
    task5 = validate_super_resolution_evidence(output_root / "task5_corrected")
    task2_validation = validate_task2_validation_evidence(output_root)
    task3 = validate_task3_evidence(output_root / "task3/condition-comparison")
    print("Evidence file check passed; course coverage and final delivery NOT certified")
    print(f"task2 candidates={loaded['task2']['num_candidate_regions']}")
    print(f"task2 replication_audited_windows={loaded['task2_replication']['num_windows']}")
    print(f"task2 fixed_candidates={task2_validation['num_candidates']} across {task2_validation['num_resolution_levels']} resolutions")
    print(f"task2 mutant_background_windows={task2_validation['num_background_windows']} with {task2_validation['normalizations']} normalizations")
    print(f"task3 conditions={task3['num_conditions']} replicates={task3['num_replicates']} bins={task3['num_bins']} tiles={task3['num_tiles']}")
    print(f"task3 chromosome_length={task3['length_bp']} last_tile={task3['last_tile']}")
    print(f"task5 cnn_psnr={task5['aggregate']['cnn']['psnr']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
