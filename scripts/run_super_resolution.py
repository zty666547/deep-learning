"""Run the optional contact-map super-resolution experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from microc_foundation.super_resolution import train_super_resolution


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight contact-map super-resolution CNN.")
    parser.add_argument("--low-dataset", required=True)
    parser.add_argument("--high-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seeds", type=int, nargs="+", help="Run a fixed multi-seed protocol")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"))
    arguments = parser.parse_args()
    seeds = arguments.seeds or [arguments.seed]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    results = []
    for seed in seeds:
        output = f"{arguments.output_dir}/seed_{seed}" if arguments.seeds else arguments.output_dir
        result = train_super_resolution(
            arguments.low_dataset, arguments.high_dataset, output,
            epochs=arguments.epochs, batch_size=arguments.batch_size,
            patience=arguments.patience, seed=seed, device_name=arguments.device,
        )
        results.append(result)
        print(f"seed={seed} | bicubic_psnr={result['bicubic_baseline']['psnr']:.3f} "
              f"cnn_psnr={result['cnn']['psnr']:.3f}", flush=True)
    if arguments.seeds:
        aggregate = {}
        for method in ("bicubic_baseline", "cnn"):
            aggregate[method] = {}
            for metric in ("psnr", "ssim", "mse"):
                values = np.asarray([result[method][metric] for result in results])
                aggregate[method][metric] = {"mean": float(values.mean()),
                                             "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
        Path(arguments.output_dir, "summary.json").write_text(
            json.dumps({"seeds": seeds, "aggregate": aggregate, "runs": results}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
