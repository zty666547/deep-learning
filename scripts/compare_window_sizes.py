"""Compare genomic window sizes across fixed random seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from microc_foundation.experiments import (
    summarize_window_results,
    write_seed_summary,
    write_window_comparison,
)
from microc_foundation.training import SUPPORTED_IMBALANCE_STRATEGIES, train_baseline


def _parse_dataset_spec(value: str) -> tuple[int, str]:
    try:
        window_text, path = value.split("=", maxsplit=1)
        window_size = int(window_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("dataset must use WINDOW_SIZE_BP=PATH") from error
    if window_size <= 0 or not path:
        raise argparse.ArgumentTypeError("window size and path must be non-empty and positive")
    return window_size, path


def _validate_dataset_alignment(
    dataset_paths: dict[int, str],
) -> dict[str, dict[str, object]]:
    reference: dict[str, np.ndarray] | None = None
    alignment_fields = (
        "structure_id",
        "labels",
        "split",
        "annotation_start",
        "annotation_end",
    )
    pooled_resolution: int | None = None
    annotation_coverage: dict[str, dict[str, object]] = {}
    for window_size, path in sorted(dataset_paths.items()):
        with np.load(Path(path).expanduser()) as dataset:
            current_resolution = int(dataset["pooled_bin_size"])
            if pooled_resolution is None:
                pooled_resolution = current_resolution
            elif current_resolution != pooled_resolution:
                raise ValueError("window datasets do not use one pooled resolution")
            matrix_size = int(dataset["matrices"].shape[-1])
            if dataset["matrices"].shape[-2] != matrix_size:
                raise ValueError(f"matrices are not square in {path}")
            if matrix_size * current_resolution != window_size:
                raise ValueError(
                    f"declared window {window_size} does not match matrix shape "
                    f"and pooled resolution in {path}"
                )
            current = {field: dataset[field].copy() for field in alignment_fields}
            annotation_lengths = (
                dataset["annotation_end"] - dataset["annotation_start"]
            )
            longer_mask = annotation_lengths > window_size
            labels = dataset["labels"].astype(str)
            annotation_coverage[str(window_size)] = {
                "longer_than_window": int(longer_mask.sum()),
                "by_class": {
                    class_name: int((longer_mask & (labels == class_name)).sum())
                    for class_name in sorted(np.unique(labels).tolist())
                },
            }
        if reference is None:
            reference = current
        elif any(
            not np.array_equal(reference[field], current[field])
            for field in alignment_fields
        ):
            raise ValueError("window datasets do not contain aligned samples")
    return annotation_coverage


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare genomic window sizes with repeated CNN training."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset_spec,
        required=True,
        metavar="WINDOW_SIZE_BP=PATH",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--imbalance-strategy",
        default="focal",
        choices=SUPPORTED_IMBALANCE_STRATEGIES,
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"))
    arguments = parser.parse_args()

    dataset_paths = dict(arguments.dataset)
    if len(dataset_paths) != len(arguments.dataset):
        raise ValueError("each window size may be specified only once")
    annotation_coverage = _validate_dataset_alignment(dataset_paths)
    for window_size, coverage in annotation_coverage.items():
        print(
            f"window={window_size}bp | annotations_longer_than_window="
            f"{coverage['longer_than_window']} | by_class={coverage['by_class']}"
        )

    window_results: dict[int, list[dict[str, object]]] = {}
    for window_size, dataset_path in sorted(dataset_paths.items()):
        results: list[dict[str, object]] = []
        for seed in arguments.seeds:
            result = train_baseline(
                dataset_path,
                f"{arguments.output_dir}/{window_size}bp/seed_{seed}",
                epochs=arguments.epochs,
                batch_size=arguments.batch_size,
                learning_rate=arguments.learning_rate,
                weight_decay=arguments.weight_decay,
                patience=arguments.patience,
                seed=seed,
                device_name=arguments.device,
                imbalance_strategy=arguments.imbalance_strategy,
                focal_gamma=arguments.focal_gamma,
            )
            results.append(result)
            print(
                f"window={window_size}bp seed={seed} "
                f"| accuracy={result['test']['accuracy']:.4f} "
                f"| macro_f1={result['test']['macro_f1']:.4f}"
            )
        window_results[window_size] = results
        window_summary = summarize_window_results({window_size: results})[
            "window_sizes"
        ][str(window_size)]
        write_seed_summary(
            window_summary, f"{arguments.output_dir}/{window_size}bp"
        )

    comparison = summarize_window_results(window_results)
    comparison["annotation_coverage"] = annotation_coverage
    output = write_window_comparison(comparison, arguments.output_dir)
    print(f"Saved comparison to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
