"""Synthetic checkpoint fixtures exercise checks; these are not course results."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from microc_foundation.metrics import classification_metrics
from microc_foundation.model import MicroCCNN

_SPEC = importlib.util.spec_from_file_location(
    "verify_inference", Path(__file__).parents[1] / "scripts/verify_task1_inference.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.fixture
def evidence(tmp_path):
    torch.set_num_threads(1)
    names = ["CHID", "CHIN", "OPCID"]
    matrices = np.random.default_rng(12).uniform(size=(9, 2, 8, 8)).astype(np.float32)
    splits = np.array(["train"] * 3 + ["validation"] * 3 + ["test"] * 3)
    labels, ids = np.array(names * 3), np.array([f"toy_{index}" for index in range(9)])
    dataset = tmp_path / "toy.npz"
    np.savez(dataset, matrices=matrices, split=splits, labels=labels, structure_id=ids)
    mean, std = matrices[:3].mean(axis=(0, 2, 3)), matrices[:3].std(axis=(0, 2, 3))
    with torch.random.fork_rng():
        torch.manual_seed(12)
        model = MicroCCNN(2, 3)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy((matrices[-3:] - mean[None, :, None, None]) /
                                            std[None, :, None, None])).argmax(dim=1).numpy()
    checkpoint = {"model_state": model.state_dict(), "class_names": names, "seed": 2026,
                  "imbalance_strategy": "focal", "input_shape": [2, 8, 8],
                  "channel_mean": mean.tolist(), "channel_std": std.tolist()}
    stored = {"class_names": names, "seed": 2026, "imbalance_strategy": "focal",
              "sample_counts": {"train": 3, "validation": 3, "test": 3},
              "test_structure_ids": ids[-3:].tolist(), "test_targets": names,
              "test_predictions": [names[index] for index in prediction],
              "test": classification_metrics(np.arange(3), prediction, names)}
    torch.save(checkpoint, tmp_path / "model.pt")
    (tmp_path / "metrics.json").write_text(json.dumps(stored))
    return dataset, tmp_path, checkpoint, stored


def test_inference_reproduces_stored_predictions_and_all_metrics(evidence):
    dataset, directory, _, stored = evidence
    result, predictions = _MODULE.verify_checkpoint(dataset, directory)
    assert result["metrics"] == stored["test"]
    assert predictions["predictions"] == stored["test_predictions"]
    assert result["scaler_exact"] and result["predictions_exact"] and result["metrics_exact"]


def test_inference_rejects_scaler_not_fitted_on_reconstructed_train(evidence):
    dataset, directory, checkpoint, _ = evidence
    checkpoint["channel_mean"][0] += 0.01
    torch.save(checkpoint, directory / "model.pt")
    with pytest.raises(ValueError, match="training-only"):
        _MODULE.verify_checkpoint(dataset, directory)


def test_legacy_checkpoint_without_seed_records_metrics_provenance(evidence):
    dataset, directory, checkpoint, _ = evidence
    checkpoint.pop("seed")
    torch.save(checkpoint, directory / "model.pt")
    result, _ = _MODULE.verify_checkpoint(dataset, directory)
    assert result["seed"] == 2026
    assert result["seed_source"] == "metrics.json (legacy checkpoint lacks seed)"


def test_inference_rejects_changed_test_order(evidence):
    dataset, directory, _, stored = evidence
    stored["test_structure_ids"].reverse()
    (directory / "metrics.json").write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="identities/order"):
        _MODULE.verify_checkpoint(dataset, directory)


@pytest.mark.parametrize("field", ["prediction", "metric"])
def test_inference_rejects_inconsistent_report(evidence, field):
    dataset, directory, _, stored = evidence
    if field == "prediction":
        before = stored["test_predictions"][0]
        stored["test_predictions"][0] = "CHIN" if before != "CHIN" else "CHID"
    else:
        stored["test"]["accuracy"] += 0.01
    (directory / "metrics.json").write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="predictions|metrics"):
        _MODULE.verify_checkpoint(dataset, directory)
