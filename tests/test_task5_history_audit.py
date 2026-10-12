import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location("task5_history_audit", Path(__file__).parents[1] / "scripts/audit_task5_history.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_history_quantifies_beyond_frozen_tolerance(tmp_path):
    reference = pd.DataFrame({"epoch": [1.0, 2.0], "train_loss": [0.2, 0.1], "validation_loss": [0.3, 0.2]})
    actual = reference.copy()
    actual.loc[0, "train_loss"] += 1e-8
    actual.to_csv(tmp_path / "actual.csv", index=False)
    reference.to_csv(tmp_path / "reference.csv", index=False)
    result = _MODULE.history_differences(tmp_path / "actual.csv", tmp_path / "reference.csv")
    assert result["columns"]["train_loss"]["max_absolute_error"] == pytest.approx(1e-8)
    assert result["columns"]["train_loss"]["outside_frozen_tolerance"] == 1
    assert result["columns"]["epoch"]["outside_frozen_tolerance"] == 0


def test_history_rejects_epoch_order_and_nonfinite(tmp_path):
    data = pd.DataFrame({"epoch": [1.0, 2.0], "train_loss": [0.2, 0.1], "validation_loss": [0.3, 0.2]})
    data.to_csv(tmp_path / "reference.csv", index=False)
    data.iloc[::-1].to_csv(tmp_path / "actual.csv", index=False)
    with pytest.raises(ValueError, match="order"):
        _MODULE.history_differences(tmp_path / "actual.csv", tmp_path / "reference.csv")
    data.loc[0, "train_loss"] = float("inf")
    data.to_csv(tmp_path / "actual.csv", index=False)
    with pytest.raises(ValueError, match="non-finite"):
        _MODULE.history_differences(tmp_path / "actual.csv", tmp_path / "reference.csv")
