"""Compare class-imbalance strategies across fixed random seeds."""

from __future__ import annotations

import argparse

from microc_foundation.experiments import (
    summarize_strategy_results,
    write_seed_summary,
    write_strategy_comparison,
)
from microc_foundation.training import SUPPORTED_IMBALANCE_STRATEGIES, train_baseline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare imbalance strategies with repeated CNN training."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=SUPPORTED_IMBALANCE_STRATEGIES,
        default=list(SUPPORTED_IMBALANCE_STRATEGIES),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"))
    arguments = parser.parse_args()

    strategy_results: dict[str, list[dict[str, object]]] = {}
    for strategy in arguments.strategies:
        results: list[dict[str, object]] = []
        for seed in arguments.seeds:
            result = train_baseline(
                arguments.dataset,
                f"{arguments.output_dir}/{strategy}/seed_{seed}",
                epochs=arguments.epochs,
                batch_size=arguments.batch_size,
                learning_rate=arguments.learning_rate,
                weight_decay=arguments.weight_decay,
                patience=arguments.patience,
                seed=seed,
                device_name=arguments.device,
                imbalance_strategy=strategy,
                focal_gamma=arguments.focal_gamma,
            )
            results.append(result)
            print(
                f"strategy={strategy} seed={seed} "
                f"| accuracy={result['test']['accuracy']:.4f} "
                f"| macro_f1={result['test']['macro_f1']:.4f}"
            )
        strategy_results[strategy] = results
        strategy_summary = summarize_strategy_results({strategy: results})[
            "strategies"
        ][strategy]
        write_seed_summary(strategy_summary, f"{arguments.output_dir}/{strategy}")

    comparison = summarize_strategy_results(strategy_results)
    output = write_strategy_comparison(comparison, arguments.output_dir)
    print(f"Saved comparison to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
