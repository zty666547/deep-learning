"""Check that the repository contains the minimum reproducible submission evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    )
    missing = [path for path in required_docs if not Path(path).is_file()]
    if missing:
        raise SystemExit("Missing required documentation: " + ", ".join(missing))
    output_root = Path(arguments.output_root)
    summaries = {
        "task2": output_root / "task2_candidate_validation/summary.json",
        "task3": output_root / "task3_tracks/summary.json",
        "task5": output_root / "task5_super_resolution/seed_2026/metrics.json",
    }
    missing_outputs = [str(path) for path in summaries.values() if not path.is_file()]
    if missing_outputs:
        raise SystemExit("Missing generated outputs: " + ", ".join(missing_outputs))
    loaded = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in summaries.items()}
    print("Submission evidence check passed")
    print(f"task2 candidates={loaded['task2'].get('num_candidate_regions', 'see report')}")
    print(f"task3 windows={loaded['task3']['num_windows']}")
    print(f"task5 cnn_psnr={loaded['task5']['cnn']['psnr']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
