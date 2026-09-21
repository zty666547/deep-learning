"""Train and evaluate the first Task 1 CNN baseline."""

from __future__ import annotations

import argparse

from microc_foundation.training import train_baseline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a two-replicate CNN with validation early stopping."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"))
    arguments = parser.parse_args()

    results = train_baseline(
        arguments.dataset,
        arguments.output_dir,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        patience=arguments.patience,
        seed=arguments.seed,
        device_name=arguments.device,
    )
    test = results["test"]
    print(
        f"Completed baseline | best_epoch={results['best_epoch']} "
        f"| test_accuracy={test['accuracy']:.4f} "
        f"| test_macro_f1={test['macro_f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
