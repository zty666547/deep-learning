"""Compare pooled genomic resolutions across fixed random seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from microc_foundation.experiments import (
    summarize_resolution_results,
    write_resolution_comparison,
    write_seed_summary,
)
from microc_foundation.training import SUPPORTED_IMBALANCE_STRATEGIES, train_baseline


def _parse_dataset_spec(value: str) -> tuple[int, str]:
    try:
        resolution_text, path = value.split("=", maxsplit=1)
        resolution = int(resolution_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("dataset must use RESOLUTION_BP=PATH") from error
    if resolution <= 0 or not path:
        raise argparse.ArgumentTypeError("resolution and path must be non-empty and positive")
    return resolution, path


def _validate_dataset_alignment(dataset_paths: dict[int, str]) -> None:
    reference: dict[str, np.ndarray] | None = None
    alignment_fields = (
        "structure_id",
        "labels",
        "split",
        "window_start",
        "window_end",
    )
    for resolution, path in sorted(dataset_paths.items()):
        with np.load(Path(path).expanduser()) as dataset:
            stored_resolution = int(dataset["pooled_bin_size"])
            if stored_resolution != resolution:
                raise ValueError(
                    f"declared resolution {resolution} does not match "
                    f"{stored_resolution} in {path}"
                )
            current = {field: dataset[field].copy() for field in alignment_fields}
        if reference is None:
            reference = current
        elif any(
            not np.array_equal(reference[field], current[field])
            for field in alignment_fields
        ):
            raise ValueError("resolution datasets do not contain aligned samples")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare pooled resolutions with repeated CNN training."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset_spec,
        required=True,
        metavar="RESOLUTION_BP=PATH",
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
        raise ValueError("each resolution may be specified only once")
    _validate_dataset_alignment(dataset_paths)

    resolution_results: dict[int, list[dict[str, object]]] = {}
    for resolution, dataset_path in sorted(dataset_paths.items()):
        results: list[dict[str, object]] = []
        for seed in arguments.seeds:
            result = train_baseline(
                dataset_path,
                f"{arguments.output_dir}/{resolution}bp/seed_{seed}",
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
                f"resolution={resolution}bp seed={seed} "
                f"| accuracy={result['test']['accuracy']:.4f} "
                f"| macro_f1={result['test']['macro_f1']:.4f}"
            )
        resolution_results[resolution] = results
        resolution_summary = summarize_resolution_results({resolution: results})[
            "resolutions"
        ][str(resolution)]
        write_seed_summary(
            resolution_summary, f"{arguments.output_dir}/{resolution}bp"
        )

    comparison = summarize_resolution_results(resolution_results)
    output = write_resolution_comparison(comparison, arguments.output_dir)
    print(f"Saved comparison to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
