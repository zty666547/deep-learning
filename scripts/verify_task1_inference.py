"""Verify existing Focal checkpoints on reconstructed data without retraining."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import torch

from microc_foundation.metrics import classification_metrics
from microc_foundation.model import MicroCCNN


def verify_checkpoint(dataset: Path, run_dir: Path) -> tuple[dict, dict]:
    stored = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    names = checkpoint["class_names"]
    if (names != ["CHID", "CHIN", "OPCID"] or names != stored["class_names"]
            or checkpoint.get("seed", stored["seed"]) != stored["seed"]
            or checkpoint["imbalance_strategy"] != "focal" or stored["imbalance_strategy"] != "focal"):
        raise ValueError("checkpoint metadata disagrees with frozen Focal run")
    with np.load(dataset, allow_pickle=False) as data:
        matrices, splits = data["matrices"].astype(np.float32), data["split"].astype(str)
        ids, labels = data["structure_id"].astype(str), data["labels"].astype(str)
    if (list(matrices.shape[1:]) != checkpoint["input_shape"] or not np.isfinite(matrices).all()
            or len(set(ids)) != len(ids) or not set(labels).issubset(names)):
        raise ValueError("dataset shape, values or identities disagree with checkpoint")
    counts = {name: int((splits == name).sum()) for name in ("train", "validation", "test")}
    if counts != stored["sample_counts"] or any(value == 0 for value in counts.values()):
        raise ValueError("dataset split counts disagree with frozen run")
    train = matrices[splits == "train"]
    expected_mean, expected_std = train.mean(axis=(0, 2, 3)), train.std(axis=(0, 2, 3))
    expected_std = np.where(expected_std < 1e-6, 1.0, expected_std)
    mean, std = np.asarray(checkpoint["channel_mean"], dtype=np.float32), np.asarray(checkpoint["channel_std"], dtype=np.float32)
    if (not np.array_equal(mean, expected_mean) or not np.array_equal(std, expected_std)
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("checkpoint scaler differs from reconstructed training-only statistics")
    test = splits == "test"
    if ids[test].tolist() != stored["test_structure_ids"] or labels[test].tolist() != stored["test_targets"]:
        raise ValueError("test identities/order/targets disagree with frozen run")
    model = MicroCCNN(matrices.shape[1], len(names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    inputs = torch.from_numpy((matrices[test] - mean[None, :, None, None]) / std[None, :, None, None])
    with torch.inference_mode():
        logits = torch.cat([model(inputs[start:start + 32]) for start in range(0, len(inputs), 32)])
    if not torch.isfinite(logits).all():
        raise ValueError("model produced non-finite logits")
    predicted = logits.argmax(dim=1).numpy()
    predictions = [names[index] for index in predicted]
    if predictions != stored["test_predictions"]:
        raise ValueError("reconstructed-input predictions differ from stored predictions")
    metrics = classification_metrics(np.array([names.index(value) for value in labels[test]]), predicted, names)
    if any(metrics[key] != stored["test"][key] for key in metrics):
        raise ValueError("recomputed classification metrics differ from stored metrics")
    return {"seed": int(stored["seed"]), "seed_source": "checkpoint" if "seed" in checkpoint else "metrics.json (legacy checkpoint lacks seed)",
            "test_structures": int(test.sum()),
            "scaler_exact": True, "predictions_exact": True, "metrics_exact": True, "metrics": metrics}, {
                "structure_id": ids[test].tolist(), "targets": labels[test].tolist(),
                "predictions": predictions, "logits": logits.numpy().tolist(),
            }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if len(args.run_dir) != 3 or len({Path(value).resolve() for value in args.run_dir}) != 3:
        parser.error("three distinct frozen Focal run directories are required")
    import microc_foundation

    module = Path(microc_foundation.__file__).resolve()
    installation = json.loads(importlib.metadata.distribution("microc-foundation").read_text("direct_url.json") or "{}")
    if (not sys.flags.isolated or sys.prefix == sys.base_prefix
            or not module.is_relative_to(Path(sys.prefix).resolve())
            or installation.get("dir_info", {}).get("editable", False)):
        parser.error("use isolated python -I with a non-editable package in a separate venv")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    receipt = {"protocol": "task1-installed-inference-v1", "retrained": False,
               "full_course_reproduction": False, "status": "running", "runs": [],
               "environment": {"python": sys.version, "package_file": str(module), "installation": installation,
                               "isolated": True, "editable": False},
               "runner_sha256": sha256_file(Path(__file__)),
               "input_sha256": {args.dataset: sha256_file(Path(args.dataset))}}
    try:
        for directory in args.run_dir:
            root = Path(directory)
            for name in ("model.pt", "metrics.json"):
                receipt["input_sha256"][str(root / name)] = sha256_file(root / name)
            result, predictions = verify_checkpoint(Path(args.dataset), root)
            if root.name != f"seed_{result['seed']}":
                raise ValueError("run directory name disagrees with metrics seed")
            receipt["runs"].append(result)
            (output / f"seed_{result['seed']}_predictions.json").write_text(
                json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")
        if ([result["seed"] for result in receipt["runs"]] != [2026, 2027, 2028]
                or any(result["test_structures"] != 41 for result in receipt["runs"])):
            raise ValueError("requires fixed seeds 2026/2027/2028 and all 41 existing test structures")
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Installed-checkpoint inference passed: 3 models x 41 existing test structures; no retraining")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
