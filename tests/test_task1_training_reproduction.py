import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "task1_training_reproduction", Path(__file__).parents[1] / "scripts/reproduce_task1_training.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def checkpoint_pair(tmp_path, legacy=True):
    reference = {"model_state": {"weight": torch.tensor([1.0, 2.0]), "count": torch.tensor(3)},
                 "channel_mean": [0.1], "channel_std": [1.0]}
    actual = copy.deepcopy(reference)
    actual["seed"] = 2026
    if not legacy:
        reference["seed"] = 2026
    left, right = tmp_path / "actual.pt", tmp_path / "reference.pt"
    torch.save(actual, left)
    torch.save(reference, right)
    return left, right, actual


@pytest.mark.parametrize("legacy", [True, False])
def test_checkpoint_checks_all_state_and_legacy_seed(tmp_path, legacy):
    left, right, _ = checkpoint_pair(tmp_path, legacy)
    result = _MODULE.compare_checkpoints(left, right, 2026)
    assert result["all_state_tensors_exact"]
    assert result["state_tensors_checked"] == 2
    assert result["legacy_reference_missing_seed"] is legacy


def test_checkpoint_records_tensor_divergence_without_relaxation(tmp_path):
    left, right, actual = checkpoint_pair(tmp_path)
    actual["model_state"]["weight"][0] += 0.25
    torch.save(actual, left)
    result = _MODULE.compare_checkpoints(left, right, 2026)
    assert not result["all_state_tensors_exact"]
    assert result["differing_tensors"] == {"weight": 0.25}


@pytest.mark.parametrize("mutation,match", [
    (lambda x: x.update(seed=2027), "seed"),
    (lambda x: x.update(channel_mean=[0.2]), "numerical"),
    (lambda x: x["model_state"].update(extra=torch.tensor(1)), "keys"),
    (lambda x: x["model_state"].update(weight=torch.tensor([float("nan"), 2.0])), "non-finite"),
    (lambda x: x["model_state"].update(weight=torch.tensor([1.0])), "shape"),
])
def test_checkpoint_rejects_corruption(tmp_path, mutation, match):
    left, right, actual = checkpoint_pair(tmp_path)
    mutation(actual)
    torch.save(actual, left)
    with pytest.raises(ValueError, match=match):
        _MODULE.compare_checkpoints(left, right, 2026)


def test_reference_identity_is_frozen(tmp_path):
    root = tmp_path / "seed_2026"
    root.mkdir()
    metrics = {"seed": 2026, "device": "cpu", "imbalance_strategy": "focal", "focal_gamma": 2.0,
               "class_names": ["CHID", "CHIN", "OPCID"],
               "sample_counts": {"train": 248, "validation": 55, "test": 41}}
    (root / "metrics.json").write_text(json.dumps(metrics))
    _MODULE.validate_reference(root, 2026)
    metrics["sample_counts"]["validation"] = 54
    (root / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="identity/settings"):
        _MODULE.validate_reference(root, 2026)


def test_run_reports_all_missing_evidence_not_success(tmp_path):
    result = _MODULE.compare_run(tmp_path, tmp_path / "missing", 2026)
    assert not result["matched"]
    assert set(result["errors"]) == {"metrics", "history", "checkpoint"}
