import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

_SPEC = importlib.util.spec_from_file_location("task5_reproduction", Path(__file__).parents[1] / "scripts/reproduce_task5.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_all_npz_arrays_exact(tmp_path):
    for name in ("actual", "reference"):
        np.savez(tmp_path / f"{name}.npz", cnn=np.ones((2, 2, 4, 4), dtype=np.float32), ids=np.array(["a", "b"]))
    result = _MODULE.compare_npz(tmp_path / "actual.npz", tmp_path / "reference.npz")
    assert result["dtype_shape_identity_and_values_exact"]
    assert result["array_shapes"]["cnn"] == [2, 2, 4, 4]


@pytest.mark.parametrize("change,match", [
    ({"ids": np.array(["b", "a"])}, "identities"),
    ({"cnn": np.ones((2, 2, 4, 4), dtype=np.float64)}, "dtype/shape"),
    ({"cnn": np.ones((1, 2, 4, 4), dtype=np.float32)}, "dtype/shape"),
    ({"cnn": np.full((2, 2, 4, 4), np.nan, dtype=np.float32)}, "non-finite"),
    ({"cnn": np.zeros((2, 2, 4, 4), dtype=np.float32)}, "values"),
])
def test_npz_rejects_order_dtype_shape_and_numerical_changes(tmp_path, change, match):
    data = {"cnn": np.ones((2, 2, 4, 4), dtype=np.float32), "ids": np.array(["a", "b"])}
    np.savez(tmp_path / "reference.npz", **data)
    data.update(change)
    np.savez(tmp_path / "actual.npz", **data)
    with pytest.raises(ValueError, match=match):
        _MODULE.compare_npz(tmp_path / "actual.npz", tmp_path / "reference.npz")


def test_npz_rejects_missing_field(tmp_path):
    np.savez(tmp_path / "a.npz", target=np.zeros(3))
    np.savez(tmp_path / "b.npz", target=np.zeros(3), cnn=np.zeros(3))
    with pytest.raises(ValueError, match="inventory"):
        _MODULE.compare_npz(tmp_path / "a.npz", tmp_path / "b.npz")


def test_path_normalization_preserves_seed_identity_and_does_not_mutate():
    data = {"input_files": ["/old/seed_2026/test_reconstruction.npz", "/old/seed_2027/test_reconstruction.npz"], "seeds": [2026, 2027]}
    before = copy.deepcopy(data)
    normalized = _MODULE.normalize_evaluation_paths(data)
    assert data == before
    assert normalized["input_files"] == ["seed_2026/test_reconstruction.npz", "seed_2027/test_reconstruction.npz"]


def test_aggregation_requires_ordered_seeds_and_sample_sd():
    runs = [{"seed": s, "cnn": {"mse": v, "psnr": v, "ssim": v},
             "bicubic_baseline": {"mse": 1.0, "psnr": 1.0, "ssim": 1.0}} for s, v in zip([2026, 2027, 2028], [1.0, 2.0, 3.0])]
    assert _MODULE.aggregate_runs(runs)["aggregate"]["cnn"]["mse"] == {"mean": 2.0, "std": 1.0}
    with pytest.raises(ValueError, match="ordered"):
        _MODULE.aggregate_runs(runs[::-1])
    runs[0]["cnn"]["mse"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        _MODULE.aggregate_runs(runs)


def test_missing_outputs_are_divergence_not_success(tmp_path):
    result = _MODULE.compare_run(tmp_path, tmp_path / "missing", 2026)
    assert not result["matched"]
    assert len(result["errors"]) == 5


def test_sr_checkpoint_does_not_accept_legacy_missing_seed(tmp_path):
    data = {"model_state": {"weight": torch.ones(3)}, "seed": 2026, "task": "contact_super_resolution"}
    torch.save(data, tmp_path / "actual.pt")
    torch.save(data, tmp_path / "reference.pt")
    assert _MODULE.compare_checkpoint(tmp_path / "actual.pt", tmp_path / "reference.pt", 2026)["all_state_tensors_exact"]
    data.pop("seed")
    torch.save(data, tmp_path / "reference.pt")
    with pytest.raises(ValueError, match="must contain matching seed"):
        _MODULE.compare_checkpoint(tmp_path / "actual.pt", tmp_path / "reference.pt", 2026)


def test_reference_refuses_old_unfair_protocol(tmp_path):
    root = tmp_path / "seed_2026"
    root.mkdir()
    record = {"seed": 2026, "device": "cpu", "sample_counts": {"train": 248, "validation": 54, "test": 41},
              "excluded_after_alignment": 1, "input_shape": [2, 128, 128], "target_shape": [2, 256, 256],
              "protocol": "count-corrected factor-two; shared train-only scale; symmetric CNN",
              "intensity_correction": "log1p(expm1(low_log_counts)/4)",
              "ssim_definition": "per-replicate 7x7 valid uniform-window SSIM, population moments"}
    (root / "metrics.json").write_text(json.dumps(record))
    _MODULE.validate_reference(root, 2026)
    record["intensity_correction"] = "no correction"
    (root / "metrics.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="fair corrected"):
        _MODULE.validate_reference(root, 2026)
