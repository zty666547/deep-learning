"""Read-only loss-reduction diagnostic; never retrain or rewrite original receipts."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

_SPEC = importlib.util.spec_from_file_location("task5_readonly_checks", Path(__file__).with_name("reproduce_task5.py"))
_CHECKS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHECKS)


def history_differences(actual: Path, reference: Path) -> dict:
    left, right = pd.read_csv(actual), pd.read_csv(reference)
    if (list(left.columns) != ["epoch", "train_loss", "validation_loss"]
            or list(left.columns) != list(right.columns) or left.shape != right.shape
            or not left.epoch.equals(right.epoch)):
        raise ValueError("history schema, epoch count or order differs")
    result = {}
    for key in left.columns:
        x, y = left[key].to_numpy(), right[key].to_numpy()
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("non-finite history")
        result[key] = {"max_absolute_error": float(np.max(np.abs(x - y))),
                       "outside_frozen_tolerance": int((~np.isclose(x, y, rtol=1e-12, atol=1e-12)).sum())}
    return {"rows": len(left), "columns": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reproduction-dir", "reference-runs", "reference-strata", "output-json"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    destination = Path(args.output_json)
    if destination.exists() or not destination.parent.is_dir():
        parser.error("use a new output file in an existing directory; never overwrite evidence")
    output, references = Path(args.reproduction_dir).resolve(), Path(args.reference_runs).resolve()
    original = json.loads((output / "receipt.json").read_text())
    if (original["protocol"] != "task5-corrected-raw-training-reproduction-v1"
            or original["status"] not in ("passed", "diverged") or len(original["runs"]) != 3):
        parser.error("requires all three finished frozen reproduction runs")
    result = {"scope": "readonly_best_checkpoint_validation_not_historical_retraining",
              "environment": _CHECKS.HELPERS.verify_environment(), "original_status": original["status"],
              "original_receipt_sha256": _CHECKS.HELPERS.sha256_file(output / "receipt.json"),
              "script_sha256": {name: _CHECKS.HELPERS.sha256_file(Path(__file__).with_name(name)) for name in
                                 (Path(__file__).name, "reproduce_task5.py")}, "runs": []}
    from microc_foundation.super_resolution import (
        ContactSuperResolutionCNN,
        load_aligned_super_resolution_data,
    )

    default_threads = torch.get_num_threads()
    if default_threads == 1:
        raise ValueError("default CPU threads equal one; this diagnostic has no distinct second condition")
    result["observed_default_cpu_threads"] = default_threads
    for name in ("low_windows.npz", "high_windows.npz"):
        if _CHECKS.HELPERS.sha256_file(output / name) != original["output_sha256"][name]:
            raise ValueError("reproduced input tensor hash changed")
    aligned = load_aligned_super_resolution_data(str(output / "low_windows.npz"), str(output / "high_windows.npz"))
    validation = aligned["split"] == "validation"
    for seed in _CHECKS.SEEDS:
        run, reference = output / f"seed_{seed}", references / f"seed_{seed}"
        for filename in ("model.pt", "metrics.json", "history.csv"):
            if (_CHECKS.HELPERS.sha256_file(run / filename) != original["output_sha256"][f"seed_{seed}/{filename}"]
                    or _CHECKS.HELPERS.sha256_file(reference / filename) != original["input_sha256"][str(reference / filename)]):
                raise ValueError("new or reference model/history/metrics changed")
        new_metrics = json.loads((run / "metrics.json").read_text())
        old_metrics = json.loads((reference / "metrics.json").read_text())
        if new_metrics["best_epoch"] != old_metrics["best_epoch"]:
            raise ValueError("best epochs differ; this diagnostic compares the same selected epoch only")
        checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
        model = ContactSuperResolutionCNN(2)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        scale = {name: np.asarray(checkpoint[name], dtype=np.float32)[None, :, None, None]
                 for name in ("low_mean", "low_std", "high_mean", "high_std")}
        inputs = torch.from_numpy((aligned["low"][validation] - scale["low_mean"]) / scale["low_std"])
        targets = torch.from_numpy((aligned["high"][validation] - scale["high_mean"]) / scale["high_std"])
        losses, predictions = {}, {}
        for threads in (1, default_threads):
            torch.set_num_threads(threads)
            total, chunks = 0.0, []
            with torch.no_grad():
                for start in range(0, len(inputs), 16):
                    predicted = model(inputs[start:start + 16])
                    chunks.append(predicted)
                    total += float(F.mse_loss(predicted, targets[start:start + 16]).item()) * len(predicted)
            losses[str(threads)] = total / len(inputs)
            predictions[threads] = torch.cat(chunks)
        best_epoch = new_metrics["best_epoch"]
        old = pd.read_csv(reference / "history.csv")
        new = pd.read_csv(run / "history.csv")
        old_loss = float(old.loc[old.epoch == best_epoch, "validation_loss"].iloc[0])
        new_loss = float(new.loc[new.epoch == best_epoch, "validation_loss"].iloc[0])
        result["runs"].append({"seed": seed, "best_epoch": best_epoch,
                               "history": history_differences(run / "history.csv", reference / "history.csv"),
                               "reference_best_validation_loss": old_loss, "reproduced_best_validation_loss": new_loss,
                               "readonly_validation_loss_by_threads": losses,
                               "validation_predictions_exact_across_threads": bool(torch.equal(predictions[1], predictions[default_threads])),
                               "one_thread_matches_new_best_loss": bool(np.isclose(losses["1"], new_loss, rtol=1e-12, atol=1e-12)),
                               "default_threads_matches_old_best_loss": bool(np.isclose(losses[str(default_threads)], old_loss, rtol=1e-12, atol=1e-12))})
    strata_reference = Path(args.reference_strata).resolve() / "summary.json"
    if (_CHECKS.HELPERS.sha256_file(output / "strata/summary.json") != original["output_sha256"]["strata/summary.json"]
            or _CHECKS.HELPERS.sha256_file(strata_reference) != original["input_sha256"][str(strata_reference)]):
        raise ValueError("serialized strata summary changed")
    result["serialized_strata_summary_recheck"] = _CHECKS.compare_evaluation_summaries(
        output / "strata/summary.json", strata_reference)
    if _CHECKS.HELPERS.sha256_file(output / "receipt.json") != result["original_receipt_sha256"]:
        raise ValueError("original reproduction receipt changed during diagnostic")
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved read-only diagnostic; original strict reproduction status unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
