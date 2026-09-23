"""Run the Task 1 CNN baseline across several fixed random seeds."""

from __future__ import annotations

import argparse

from microc_foundation.experiments import summarize_seed_results, write_seed_summary
from microc_foundation.training import SUPPORTED_IMBALANCE_STRATEGIES, train_baseline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repeat the CNN baseline and summarize seed-to-seed variation."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"))
    parser.add_argument(
        "--imbalance-strategy",
        default="weighted_ce",
        choices=SUPPORTED_IMBALANCE_STRATEGIES,
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    arguments = parser.parse_args()

    results: list[dict[str, object]] = []
    for seed in arguments.seeds:
        result = train_baseline(
            arguments.dataset,
            f"{arguments.output_dir}/seed_{seed}",
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
            f"seed={seed} | accuracy={result['test']['accuracy']:.4f} "
            f"| macro_f1={result['test']['macro_f1']:.4f}"
        )

    summary = summarize_seed_results(results)
    output = write_seed_summary(summary, arguments.output_dir)
    aggregate = summary["aggregate"]
    print(
        f"Saved {output} | macro_f1="
        f"{aggregate['test_macro_f1']['mean']:.4f} ± "
        f"{aggregate['test_macro_f1']['std']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
