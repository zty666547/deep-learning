"""Rebuild paired windows, retrain frozen corrected SR models and all strata."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_HELPER_PATH = Path(__file__).with_name("reproduce_task1_training.py")
_SPEC = importlib.util.spec_from_file_location("task5_training_helpers", _HELPER_PATH)
_TRAINING = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRAINING)
HELPERS = _TRAINING.HELPERS
SEEDS = (2026, 2027, 2028)
SETTINGS = {"epochs": 20, "batch_size": 16, "learning_rate": 0.001,
            "weight_decay": 0.0001, "patience": 5, "device_name": "cpu"}
STRATUM_FILES = ("distance_metrics_by_sample.csv", "known_structure_metrics_by_sample.csv",
                 "distance_metrics_summary.csv", "known_structure_metrics_summary.csv")


def compare_npz(actual_path: Path, reference_path: Path) -> dict:
    """Require every array's fields, dtype, shape and values to match exactly."""
    with np.load(actual_path, allow_pickle=False) as actual, np.load(reference_path, allow_pickle=False) as reference:
        if set(actual.files) != set(reference.files):
            raise ValueError("NPZ field inventory differs")
        shapes = {}
        for key in sorted(actual.files):
            x, y = actual[key], reference[key]
            if x.dtype != y.dtype or x.shape != y.shape:
                raise ValueError(f"NPZ dtype/shape differs: {key}")
            if np.issubdtype(x.dtype, np.number) and (not np.isfinite(x).all() or not np.isfinite(y).all()):
                raise ValueError(f"non-finite NPZ values: {key}")
            if not np.array_equal(x, y):
                raise ValueError(f"NPZ identities/order/values differ: {key}")
            shapes[key] = list(x.shape)
    return {"fields_checked": sorted(shapes), "array_shapes": shapes,
            "dtype_shape_identity_and_values_exact": True, "max_array_absolute_error": 0.0}


def normalize_evaluation_paths(summary: dict) -> dict:
    """Strip directories only; preserve each seed folder and filename in order."""
    result = copy.deepcopy(summary)
    result["input_files"] = [str(Path(p).parent.name + "/" + Path(p).name) for p in result["input_files"]]
    return result


def aggregate_runs(runs: list[dict]) -> dict:
    if [r["seed"] for r in runs] != list(SEEDS):
        raise ValueError("aggregate requires ordered frozen three seeds")
    aggregate = {}
    for method in ("bicubic_baseline", "cnn"):
        aggregate[method] = {}
        for metric in ("psnr", "ssim", "mse"):
            values = np.asarray([r[method][metric] for r in runs])
            if not np.isfinite(values).all():
                raise ValueError("non-finite aggregate metric")
            aggregate[method][metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    return {"seeds": list(SEEDS), "aggregate": aggregate, "runs": runs}


def validate_reference(root: Path, seed: int) -> dict:
    result = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    if (root.name != f"seed_{seed}" or result["seed"] != seed or result["device"] != "cpu"
            or result["sample_counts"] != {"train": 248, "validation": 54, "test": 41}
            or result["excluded_after_alignment"] != 1 or result["input_shape"] != [2, 128, 128]
            or result["target_shape"] != [2, 256, 256]
            or result["protocol"] != "count-corrected factor-two; shared train-only scale; symmetric CNN"
            or result["intensity_correction"] != "log1p(expm1(low_log_counts)/4)"
            or result["ssim_definition"] != "per-replicate 7x7 valid uniform-window SSIM, population moments"):
        raise ValueError("reference differs from frozen fair corrected protocol")
    return result


def compare_checkpoint(actual: Path, reference: Path, seed: int) -> dict:
    stored = torch.load(reference, map_location="cpu", weights_only=True)
    if stored.get("seed") != seed:
        raise ValueError("Task 5 reference checkpoint must contain matching seed")
    return _TRAINING.compare_checkpoints(actual, reference, seed)


def compare_run(actual: Path, reference: Path, seed: int) -> dict:
    result = {"seed": seed, "checks": {}, "errors": {}}
    for name, operation in {
        "metrics": lambda: {"max_float_absolute_error": HELPERS.compare_json(
            json.loads((actual / "metrics.json").read_text()), json.loads((reference / "metrics.json").read_text()))},
        "history": lambda: HELPERS.compare_tables(actual / "history.csv", reference / "history.csv"),
        "per_sample_metrics": lambda: HELPERS.compare_tables(actual / "per_sample_metrics.csv", reference / "per_sample_metrics.csv"),
        "checkpoint": lambda: compare_checkpoint(actual / "model.pt", reference / "model.pt", seed),
        "reconstruction": lambda: compare_npz(actual / "test_reconstruction.npz", reference / "test_reconstruction.npz"),
    }.items():
        try:
            result["checks"][name] = operation()
        except (ValueError, AssertionError, KeyError, OSError) as error:
            result["errors"][name] = str(error)
    result["matched"] = (not result["errors"]
                         and result["checks"]["checkpoint"]["all_state_tensors_exact"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--cool", action="append", required=True)
    for name in ("reference-low", "reference-high", "reference-runs", "reference-strata", "output-dir", "source-revision"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if len(args.cool) != 2 or len({Path(p).resolve() for p in args.cool}) != 2:
        parser.error("provide two distinct ordered raw WT replicates")
    output = Path(args.output_dir).resolve()
    HELPERS.prepare_output(output)
    started = time.monotonic()
    receipt = {"protocol": "task5-corrected-raw-training-reproduction-v1", "status": "running",
               "scope": "raw_paired_windows_three_corrected_trainings_full_reconstructions_and_strata",
               "source_revision_label": args.source_revision,
               "started_utc": datetime.now(timezone.utc).isoformat(), "settings": SETTINGS,
               "seeds": list(SEEDS), "retrained": True, "full_course_reproduction": False,
               "biological_validation": False, "historical_cpu_threads_recorded": False,
               "settings_provenance": "documented method and original CLI defaults; full historical CLI not in reference artifacts",
               "split_guard_bp": {"low": 30_720, "high": 20_480}, "runs": [], "strata_checks": {}, "comparison_errors": {}}
    stage = "environment"
    try:
        receipt["environment"] = HELPERS.verify_environment()
        torch.set_num_threads(1)
        receipt["torch_execution"] = {"threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                                      "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
        from microc_foundation import read_structures_excel, write_structures_csv
        from microc_foundation.datasets import build_structure_dataset
        from microc_foundation.io import open_cooler
        from microc_foundation.splitting import assign_contiguous_splits, write_split_manifest
        from microc_foundation.super_resolution import (
            load_aligned_super_resolution_data,
            train_super_resolution,
        )
        from microc_foundation.super_resolution_evaluation import evaluate_reconstructions

        stage = "reference_identity_and_input_hashes"
        reference_runs, reference_strata = Path(args.reference_runs), Path(args.reference_strata)
        for seed in SEEDS:
            validate_reference(reference_runs / f"seed_{seed}", seed)
        inputs = [Path(args.workbook), *map(Path, args.cool), Path(args.reference_low), Path(args.reference_high),
                  reference_runs / "summary.json", reference_strata / "summary.json",
                  *[reference_strata / name for name in STRATUM_FILES],
                  *[reference_runs / f"seed_{seed}" / name for seed in SEEDS for name in
                    ("model.pt", "history.csv", "metrics.json", "per_sample_metrics.csv", "test_reconstruction.npz")]]
        receipt["input_sha256"] = {str(p.resolve()): HELPERS.sha256_file(p) for p in inputs}
        receipt["script_sha256"] = {name: HELPERS.sha256_file(Path(__file__).with_name(name)) for name in
                                    (Path(__file__).name, "reproduce_task1_training.py", "reproduce_task2.py", "reproduce_foundation.py")}
        stage = "raw_annotations"
        structures = read_structures_excel(args.workbook, chrom_map={"MG1655": "NC_000913.3"})
        if structures.structure_type.value_counts().to_dict() != {"CHIN": 250, "OPCID": 68, "CHID": 26}:
            raise ValueError("requires frozen 344 annotations")
        for p in args.cool:
            with open_cooler(p) as cool:
                if (cool.binsize != 10 or cool.chromnames != ["NC_000913.3"]
                        or int(cool.chromsizes["NC_000913.3"]) != 4_641_652):
                    raise ValueError("requires frozen raw WT 10 bp chromosome")
        write_structures_csv(structures, str(output / "structures.csv"))
        datasets = {}
        receipt["dataset_comparisons"] = {}
        for name, factor, guard, reference in (("low", 16, 30_720, args.reference_low), ("high", 8, 20_480, args.reference_high)):
            stage = f"rebuild_{name}_raw_windows"
            print(f"Rebuilding all {name} windows from raw data: pooling={factor}, split guard={guard} bp", flush=True)
            split = assign_contiguous_splits(structures, 4_641_652, window_size_bp=guard)
            write_split_manifest(split, str(output / f"{name}_split.csv"))
            datasets[name] = build_structure_dataset(args.cool, split, str(output / f"{name}_windows.npz"),
                                                      window_size_bp=20_480, pool_factor=factor, pooling="sum",
                                                      normalization="log1p", balance=False)
            receipt["dataset_comparisons"][name] = compare_npz(datasets[name], Path(reference))
        stage = "alignment_and_leakage_validation"
        aligned = load_aligned_super_resolution_data(str(datasets["low"]), str(datasets["high"]))
        counts = {name: int((aligned["split"] == name).sum()) for name in ("train", "validation", "test")}
        excluded = aligned["ids"][aligned["split"] == "excluded"].tolist()
        if counts != {"train": 248, "validation": 54, "test": 41} or excluded != ["CHIN_179"]:
            raise ValueError("retained split or excluded structure differs")
        receipt["aligned_counts"], receipt["excluded_ids"] = counts, excluded
        del aligned
        metrics_runs = []
        for seed in SEEDS:
            stage = f"train_seed_{seed}"
            print(f"Retraining fixed corrected SR seed={seed}; CPU threads=1; no warm start", flush=True)
            run = output / f"seed_{seed}"
            run_started = time.monotonic()
            metrics = train_super_resolution(str(datasets["low"]), str(datasets["high"]), str(run), seed=seed, **SETTINGS)
            metrics_runs.append(metrics)
            stage = f"compare_seed_{seed}"
            comparison = compare_run(run, reference_runs / f"seed_{seed}", seed)
            comparison.update(elapsed_seconds=time.monotonic() - run_started,
                              best_epoch=metrics["best_epoch"], epochs_ran=metrics["epochs_ran"], cnn=metrics["cnn"])
            receipt["runs"].append(comparison)
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Seed {seed}: matched={comparison['matched']}, best_epoch={metrics['best_epoch']}", flush=True)
        stage = "aggregate_and_stratified_evaluation"
        aggregate = aggregate_runs(metrics_runs)
        (output / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        strata_dir = output / "strata"
        evaluation = evaluate_reconstructions([str(output / f"seed_{s}" / "test_reconstruction.npz") for s in SEEDS],
                                             str(strata_dir), bin_size_bp=80, near_diagonal_exclusion_bp=160)
        operations = {"aggregate": lambda: {"max_float_absolute_error": HELPERS.compare_json(
            aggregate, json.loads((reference_runs / "summary.json").read_text()))},
                      "strata_summary": lambda: {"max_float_absolute_error": HELPERS.compare_json(
                          normalize_evaluation_paths(evaluation), normalize_evaluation_paths(
                              json.loads((reference_strata / "summary.json").read_text())))}}
        operations.update({name: (lambda n=name: HELPERS.compare_tables(strata_dir / n, reference_strata / n)) for name in STRATUM_FILES})
        for name, operation in operations.items():
            try:
                receipt["strata_checks"][name] = operation()
            except (ValueError, AssertionError, KeyError, OSError) as error:
                receipt["comparison_errors"][name] = str(error)
        stage = "output_hashes"
        receipt["output_sha256"] = {str(p.relative_to(output)): HELPERS.sha256_file(p) for p in sorted(output.rglob("*"))
                                    if p.is_file() and p.name != "receipt.json"}
        receipt["status"] = "passed" if not receipt["comparison_errors"] and all(r["matched"] for r in receipt["runs"]) else "diverged"
    except Exception as error:
        receipt.update(status="failed", failed_stage=stage, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Task 5 reproduction: {receipt['status']}; same derived pairs, not independent biology", flush=True)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
