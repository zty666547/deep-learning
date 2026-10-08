"""Analyze stable and unstable test errors across trained checkpoints."""

from __future__ import annotations

import argparse

from microc_foundation.error_analysis import analyze_repeated_model_errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze per-sample test errors across repeated model runs."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()

    summary = analyze_repeated_model_errors(
        arguments.dataset,
        arguments.checkpoint,
        arguments.output_dir,
    )
    print(
        f"Analyzed {summary['num_test_samples']} test samples | "
        f"always_correct={summary['always_correct']} | mixed={summary['mixed']} "
        f"| always_wrong={summary['always_wrong']}"
    )
    for class_name, values in summary["per_class"].items():
        print(
            f"{class_name} | support={values['support']} "
            f"| always_correct={values['always_correct']} "
            f"| mixed={values['mixed']} | always_wrong={values['always_wrong']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
