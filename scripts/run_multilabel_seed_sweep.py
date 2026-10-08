"""Train the region-level multi-label model across fixed random seeds."""

from __future__ import annotations

import argparse

from microc_foundation.multilabel import (
    summarize_multilabel_runs,
    train_multilabel,
    write_multilabel_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Repeat the multi-label Task 1 model.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"))
    arguments = parser.parse_args()

    results: list[dict[str, object]] = []
    for seed in arguments.seeds:
        result = train_multilabel(
            arguments.dataset,
            f"{arguments.output_dir}/seed_{seed}",
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            patience=arguments.patience,
            seed=seed,
            device_name=arguments.device,
        )
        results.append(result)
        print(
            f"seed={seed} | macro_f1={result['test']['macro_f1']:.4f} "
            f"| exact_match={result['test']['exact_match_ratio']:.4f}"
        )

    summary = summarize_multilabel_runs(results)
    output = write_multilabel_summary(summary, arguments.output_dir)
    aggregate = summary["aggregate"]
    print(
        f"Saved {output} | macro_f1="
        f"{aggregate['test_macro_f1']['mean']:.4f} ± "
        f"{aggregate['test_macro_f1']['std']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
