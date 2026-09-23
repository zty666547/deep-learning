"""Helpers for aggregating repeated baseline runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


_METRICS = ("accuracy", "macro_f1", "macro_recall")


def summarize_seed_results(results: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate repeated runs without hiding individual seed results."""

    if not results:
        raise ValueError("at least one seed result is required")
    class_names = list(results[0]["class_names"])
    rows: list[dict[str, float | int]] = []
    for result in results:
        if list(result["class_names"]) != class_names:
            raise ValueError("all runs must use the same class order")
        test = result["test"]
        row: dict[str, float | int] = {
            "seed": int(result["seed"]),
            "best_epoch": int(result["best_epoch"]),
            "epochs_ran": int(result["epochs_ran"]),
        }
        for metric in _METRICS:
            row[f"test_{metric}"] = float(test[metric])
        for class_name in class_names:
            row[f"test_recall_{class_name}"] = float(
                test["per_class"][class_name]["recall"]
            )
        rows.append(row)

    aggregate: dict[str, dict[str, float]] = {}
    metric_columns = [f"test_{metric}" for metric in _METRICS]
    metric_columns.extend(f"test_recall_{name}" for name in class_names)
    for column in metric_columns:
        values = np.asarray([float(row[column]) for row in rows])
        aggregate[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "num_runs": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
        "class_names": class_names,
        "aggregate": aggregate,
        "runs": rows,
    }


def write_seed_summary(summary: dict[str, object], output_dir: str) -> Path:
    """Write JSON, CSV and a compact seed-stability figure."""

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = summary["runs"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    seeds = [int(row["seed"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for metric, label in (
        ("test_accuracy", "Accuracy"),
        ("test_macro_f1", "Macro F1"),
        ("test_macro_recall", "Macro recall"),
    ):
        axis.plot(seeds, [float(row[metric]) for row in rows], marker="o", label=label)
    axis.set(
        xlabel="Random seed",
        ylabel="Test score",
        title="Task 1 baseline stability across seeds",
        ylim=(0, 1),
    )
    axis.set_xticks(seeds)
    axis.legend()
    figure.savefig(output / "seed_stability.png", dpi=180)
    plt.close(figure)
    return output


def summarize_strategy_results(
    strategy_results: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    """Aggregate seed summaries for multiple imbalance strategies."""

    if not strategy_results:
        raise ValueError("at least one strategy result is required")
    summaries = {
        strategy: summarize_seed_results(results)
        for strategy, results in strategy_results.items()
    }
    class_orders = {tuple(summary["class_names"]) for summary in summaries.values()}
    if len(class_orders) != 1:
        raise ValueError("all strategies must use the same class order")
    return {"strategies": summaries, "class_names": list(next(iter(class_orders)))}


def write_strategy_comparison(summary: dict[str, object], output_dir: str) -> Path:
    """Write JSON, CSV and a compact strategy comparison figure."""

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    class_names = summary["class_names"]
    rows: list[dict[str, object]] = []
    for strategy, strategy_summary in summary["strategies"].items():
        row: dict[str, object] = {
            "strategy": strategy,
            "num_runs": strategy_summary["num_runs"],
        }
        aggregate = strategy_summary["aggregate"]
        for metric in _METRICS:
            for statistic in ("mean", "std"):
                row[f"test_{metric}_{statistic}"] = aggregate[f"test_{metric}"][
                    statistic
                ]
        for class_name in class_names:
            for statistic in ("mean", "std"):
                row[f"test_recall_{class_name}_{statistic}"] = aggregate[
                    f"test_recall_{class_name}"
                ][statistic]
        rows.append(row)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    strategies = [str(row["strategy"]) for row in rows]
    metric_keys = ["test_macro_f1", *[f"test_recall_{name}" for name in class_names]]
    labels = ["Macro F1", *[f"{name} recall" for name in class_names]]
    x_positions = np.arange(len(strategies))
    width = 0.8 / len(metric_keys)
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for index, (metric, label) in enumerate(zip(metric_keys, labels)):
        aggregate_values = [
            summary["strategies"][strategy]["aggregate"][metric]
            for strategy in strategies
        ]
        offset = (index - (len(metric_keys) - 1) / 2) * width
        axis.bar(
            x_positions + offset,
            [value["mean"] for value in aggregate_values],
            width,
            yerr=[value["std"] for value in aggregate_values],
            capsize=3,
            label=label,
        )
    axis.set(
        xticks=x_positions,
        xticklabels=strategies,
        ylabel="Test score (mean ± sample SD)",
        title="Task 1 class-imbalance strategy comparison",
        ylim=(0, 1),
    )
    axis.legend(ncol=2)
    figure.savefig(output / "strategy_comparison.png", dpi=180)
    plt.close(figure)
    return output
