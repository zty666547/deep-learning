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
        "docs/task2-candidate-registry.csv",
    )
    missing = [path for path in required_docs if not Path(path).is_file()]
    if missing:
        raise SystemExit("Missing required documentation: " + ", ".join(missing))
    output_root = Path(arguments.output_root)
    summaries = {
        "task2": output_root / "task2_clustering/summary.json",
        "task2_replication": output_root / "task2_replication_audit/summary.json",
        "task3": output_root / "task3_tracks/summary.json",
        "task5": output_root / "task5_corrected/summary.json",
    }
    missing_outputs = [str(path) for path in summaries.values() if not path.is_file()]
    if missing_outputs:
        raise SystemExit("Missing generated outputs: " + ", ".join(missing_outputs))
    loaded = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in summaries.items()}
    task5 = validate_super_resolution_evidence(output_root / "task5_corrected")
    print("Evidence file check passed; course coverage and final delivery NOT certified")
    print(f"task2 candidates={loaded['task2']['num_candidate_regions']}")
    print(f"task2 replication_audited_windows={loaded['task2_replication']['num_windows']}")
    print(f"task3 windows={loaded['task3']['num_windows']}")
    print(f"task5 cnn_psnr={task5['aggregate']['cnn']['psnr']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
