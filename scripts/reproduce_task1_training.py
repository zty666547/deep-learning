"""Retrain the three frozen Focal runs from raw data; never tune to match history."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def load_helper(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = load_helper("task1_training_checks", "reproduce_task2.py")
FOUNDATION = load_helper("task1_training_foundation", "reproduce_foundation.py")
SEEDS = (2026, 2027, 2028)
SETTINGS = {"epochs": 40, "batch_size": 32, "learning_rate": 0.001,
            "weight_decay": 0.0001, "patience": 8, "device_name": "cpu",
            "imbalance_strategy": "focal", "focal_gamma": 2.0}


def validate_reference(root: Path, seed: int) -> dict:
    stored = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    if (stored["seed"] != seed or root.name != f"seed_{seed}"
            or stored["device"] != "cpu" or stored["imbalance_strategy"] != "focal"
            or stored["focal_gamma"] != 2.0
            or stored["class_names"] != ["CHID", "CHIN", "OPCID"]
            or stored["sample_counts"] != {"train": 248, "validation": 55, "test": 41}):
        raise ValueError("reference identity/settings differ from frozen Focal protocol")
    return stored


def compare_checkpoints(actual_path: Path, reference_path: Path, seed: int) -> dict:
    actual = torch.load(actual_path, map_location="cpu", weights_only=True)
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    actual_metadata = {k: v for k, v in actual.items() if k != "model_state"}
    reference_metadata = {k: v for k, v in reference.items() if k != "model_state"}
    legacy = "seed" not in reference_metadata
    if actual_metadata.get("seed") != seed or reference_metadata.get("seed", seed) != seed:
        raise ValueError("checkpoint seed disagrees with run identity")
    # Old checkpoints lack only seed: validate it in metrics and retain this fact.
    reference_metadata.setdefault("seed", seed)
    metadata_error = HELPERS.compare_json(actual_metadata, reference_metadata)
    left, right = actual["model_state"], reference["model_state"]
    if set(left) != set(right) or not left:
        raise ValueError("checkpoint state keys differ or are empty")
    differences = {}
    for key in sorted(left):
        x, y = left[key], right[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"tensor shape/dtype differs: {key}")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError(f"non-finite checkpoint tensor: {key}")
        if not torch.equal(x, y):
            differences[key] = float((x.to(torch.float64) - y.to(torch.float64)).abs().max())
    return {"state_tensors_checked": len(left), "all_state_tensors_exact": not differences,
            "max_tensor_absolute_error": max(differences.values(), default=0.0),
            "differing_tensors": differences, "metadata_max_float_error": metadata_error,
            "legacy_reference_missing_seed": legacy,
            "seed_source": "metrics.json and directory" if legacy else "checkpoint and metrics.json"}


def compare_run(actual: Path, reference: Path, seed: int) -> dict:
    result = {"seed": seed, "checks": {}, "errors": {}}
    for name, operation in {
        "metrics": lambda: {"max_float_absolute_error": HELPERS.compare_json(
            json.loads((actual / "metrics.json").read_text(encoding="utf-8")),
            json.loads((reference / "metrics.json").read_text(encoding="utf-8")))},
        "history": lambda: HELPERS.compare_tables(actual / "history.csv", reference / "history.csv"),
        "checkpoint": lambda: compare_checkpoints(actual / "model.pt", reference / "model.pt", seed),
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
    parser.add_argument("--reference-dataset", required=True)
    parser.add_argument("--reference-root", required=True, help="Existing focal directory with three seed subdirectories")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-revision", required=True, help="Label for the installed wheel's source commit")
    args = parser.parse_args()
    if len(args.cool) != 2 or len({Path(p).resolve() for p in args.cool}) != 2:
        parser.error("provide two distinct ordered raw WT replicate paths")
    output = Path(args.output_dir).resolve()
    HELPERS.prepare_output(output)
    started = time.monotonic()
    receipt = {"protocol": "task1-raw-focal-training-reproduction-v1", "status": "running",
               "scope": "raw_windows_and_three_default_focal_trainings",
               "source_revision_label": args.source_revision,
               "started_utc": datetime.now(timezone.utc).isoformat(), "retrained": True,
               "full_course_reproduction": False, "biological_validation": False,
               "settings": SETTINGS, "seeds": list(SEEDS), "runs": [],
               "historical_cpu_threads_recorded": False,
               "settings_provenance": "documented method plus original CLI defaults; historical optimizer CLI not saved"}
    stage = "environment"
    try:
        receipt["environment"] = HELPERS.verify_environment()
        torch.set_num_threads(1)
        receipt["torch_execution"] = {"threads": torch.get_num_threads(),
                                      "interop_threads": torch.get_num_interop_threads(),
                                      "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
        from microc_foundation import read_structures_excel, write_structures_csv
        from microc_foundation.datasets import build_structure_dataset
        from microc_foundation.io import open_cooler
        from microc_foundation.splitting import assign_contiguous_splits, write_split_manifest
        from microc_foundation.training import train_baseline

        stage = "input_hashes_and_reference_identity"
        references = [Path(args.reference_root) / f"seed_{seed}" for seed in SEEDS]
        for seed, root in zip(SEEDS, references):
            validate_reference(root, seed)
        inputs = [Path(args.workbook), *map(Path, args.cool), Path(args.reference_dataset),
                  *[root / name for root in references for name in ("model.pt", "history.csv", "metrics.json")]]
        receipt["input_sha256"] = {str(p.resolve()): HELPERS.sha256_file(p) for p in inputs}
        receipt["script_sha256"] = {name: HELPERS.sha256_file(Path(__file__).with_name(name)) for name in
                                    (Path(__file__).name, "reproduce_task2.py", "reproduce_foundation.py")}
        stage = "raw_window_reconstruction"
        print("Reconstructing all 344 windows from raw workbook and two 10 bp WT matrices", flush=True)
        structures = read_structures_excel(args.workbook, chrom_map={"MG1655": "NC_000913.3"})
        if structures.structure_type.value_counts().to_dict() != {"CHIN": 250, "OPCID": 68, "CHID": 26}:
            raise ValueError("requires frozen 344 annotations")
        for path in args.cool:
            with open_cooler(path) as cool:
                if (cool.binsize != 10 or cool.chromnames != ["NC_000913.3"]
                        or int(cool.chromsizes["NC_000913.3"]) != 4_641_652):
                    raise ValueError("requires frozen 10 bp WT chromosome")
        write_structures_csv(structures, str(output / "structures.csv"))
        split = assign_contiguous_splits(structures, 4_641_652, window_size_bp=20_480)
        write_split_manifest(split, str(output / "structures_split.csv"))
        dataset = build_structure_dataset(args.cool, split, str(output / "windows.npz"),
                                          window_size_bp=20_480, pool_factor=16, pooling="sum",
                                          normalization="log1p", balance=False)
        receipt["dataset_comparison"] = FOUNDATION.compare_datasets(dataset, Path(args.reference_dataset))
        with np.load(dataset, allow_pickle=False) as data:
            counts = {name: int((data["split"] == name).sum()) for name in ("train", "validation", "test")}
            if data["matrices"].shape != (344, 2, 128, 128) or counts != {"train": 248, "validation": 55, "test": 41}:
                raise ValueError("rebuilt dataset differs from frozen shape/split")
        for seed, root in zip(SEEDS, references):
            stage = f"train_seed_{seed}"
            print(f"Training fixed Focal seed={seed}, CPU threads=1; no checkpoint warm start", flush=True)
            run = output / f"seed_{seed}"
            run_started = time.monotonic()
            metrics = train_baseline(str(dataset), str(run), seed=seed, **SETTINGS)
            stage = f"compare_seed_{seed}"
            comparison = compare_run(run, root, seed)
            comparison.update(elapsed_seconds=time.monotonic() - run_started,
                              best_epoch=metrics["best_epoch"], epochs_ran=metrics["epochs_ran"],
                              accuracy=metrics["test"]["accuracy"], macro_f1=metrics["test"]["macro_f1"])
            receipt["runs"].append(comparison)
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Seed {seed}: matched={comparison['matched']}, best_epoch={metrics['best_epoch']}", flush=True)
        stage = "output_hashes"
        receipt["output_sha256"] = {str(p.relative_to(output)): HELPERS.sha256_file(p)
                                    for p in sorted(output.rglob("*")) if p.is_file() and p.name != "receipt.json"}
        receipt["status"] = "passed" if all(run["matched"] for run in receipt["runs"]) else "diverged"
    except Exception as error:
        receipt.update(status="failed", failed_stage=stage, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Task 1 training reproduction: {receipt['status']}; same data, not independent validation", flush=True)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
