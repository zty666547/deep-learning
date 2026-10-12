import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts/reproduce_foundation.py"
_SPEC = importlib.util.spec_from_file_location("foundation_reproduction", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_comparison_requires_exact_values_and_metadata(tmp_path):
    actual, reference = tmp_path / "actual.npz", tmp_path / "reference.npz"
    payload = {"matrices": np.ones((2, 2, 3, 3)), "structure_id": np.array(["A", "B"]),
               "split": np.array(["train", "test"])}
    np.savez(actual, **payload)
    np.savez(reference, **payload)
    assert _MODULE.compare_datasets(actual, reference)["max_matrix_absolute_error"] == 0
    payload["split"] = np.array(["train", "validation"])
    np.savez(reference, **payload)
    with pytest.raises(ValueError, match="split"):
        _MODULE.compare_datasets(actual, reference)


def test_comparison_rejects_missing_fields_and_small_numerical_changes(tmp_path):
    actual, reference = tmp_path / "actual.npz", tmp_path / "reference.npz"
    np.savez(actual, matrices=np.ones((1, 2, 3, 3)))
    np.savez(reference, matrices=np.ones((1, 2, 3, 3)), extra=np.array(1))
    with pytest.raises(ValueError, match="field sets"):
        _MODULE.compare_datasets(actual, reference)
    np.savez(reference, matrices=np.ones((1, 2, 3, 3)) + 1e-8)
    with pytest.raises(ValueError, match="matrices"):
        _MODULE.compare_datasets(actual, reference)


def test_file_hash_is_content_based(tmp_path):
    file = tmp_path / "evidence"
    file.write_bytes(b"abc")
    assert _MODULE.sha256_file(file) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_environment_rejects_nonisolated_test_process(monkeypatch):
    monkeypatch.setattr(_MODULE.sys, "prefix", _MODULE.sys.base_prefix)
    with pytest.raises(ValueError, match="python -I"):
        _MODULE.verify_installed_environment()
