"""Run the optional contact-map super-resolution experiment."""

from __future__ import annotations

import argparse

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
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"))
    arguments = parser.parse_args()
    result = train_super_resolution(
        arguments.low_dataset,
        arguments.high_dataset,
        arguments.output_dir,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        patience=arguments.patience,
        seed=arguments.seed,
        device_name=arguments.device,
    )
    print(
        f"Saved {arguments.output_dir} | "
        f"bicubic_psnr={result['bicubic_baseline']['psnr']:.3f} "
        f"cnn_psnr={result['cnn']['psnr']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
