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
