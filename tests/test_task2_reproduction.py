import importlib.util
from pathlib import Path

import cooler
import pandas as pd
import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts/reproduce_task2.py"
_SPEC = importlib.util.spec_from_file_location("task2_reproduction", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_table_comparison_checks_all_values_and_order(tmp_path):
    actual, reference = tmp_path / "actual.csv", tmp_path / "reference.csv"
    frame = pd.DataFrame({"window_id": ["A", "B"], "start": [0, 10], "score": [0.2, 0.3]})
    frame.to_csv(actual, index=False)
    frame.to_csv(reference, index=False)
    assert _MODULE.compare_tables(actual, reference)["rows"] == 2
    frame.iloc[::-1].to_csv(reference, index=False)
    with pytest.raises(ValueError, match="window_id"):
        _MODULE.compare_tables(actual, reference)
    frame.loc[0, "score"] += 1e-6
    frame.to_csv(reference, index=False)
    with pytest.raises(ValueError, match="score"):
        _MODULE.compare_tables(actual, reference)


def test_table_comparison_rejects_schema_and_discrete_changes(tmp_path):
    actual, reference = tmp_path / "actual.csv", tmp_path / "reference.csv"
    pd.DataFrame({"start": [0], "quality_pass": [True]}).to_csv(actual, index=False)
    pd.DataFrame({"start": [1], "quality_pass": [True]}).to_csv(reference, index=False)
    with pytest.raises(ValueError, match="start"):
        _MODULE.compare_tables(actual, reference)
    pd.DataFrame({"start": [0]}).to_csv(reference, index=False)
    with pytest.raises(ValueError, match="schema"):
        _MODULE.compare_tables(actual, reference)


def test_json_checks_nested_centers_and_types():
    data = {"centers": [[0.3]], "count": 2, "passed": True}
    assert _MODULE.compare_json(data, data) == 0
    assert _MODULE.compare_json({**data, "centers": [[0.3 + 1e-14]]}, data) < 1e-12
    with pytest.raises(ValueError, match="centers"):
        _MODULE.compare_json({**data, "centers": [[0.3001]]}, data)
    with pytest.raises(ValueError, match="passed"):
        _MODULE.compare_json({**data, "passed": 1}, data)
    with pytest.raises(ValueError, match="count"):
        _MODULE.compare_json({**data, "count": 2.0}, data)


def test_output_guard_preserves_old_evidence(tmp_path):
    output = tmp_path / "run"
    _MODULE.prepare_output(output)
    (output / "receipt.json").write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError, match="never overwrite"):
        _MODULE.prepare_output(output)
    assert (output / "receipt.json").read_text() == "existing"


def test_cooler_comparison_checks_every_pixel_and_count_sum(tmp_path):
    bins = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 10], "end": [10, 20]})
    pixels = pd.DataFrame({"bin1_id": [0, 0, 1], "bin2_id": [0, 1, 1], "count": [2, 3, 4]})
    actual, reference = tmp_path / "actual.cool", tmp_path / "reference.cool"
    cooler.create_cooler(str(actual), bins, pixels)
    cooler.create_cooler(str(reference), bins, pixels)
    assert _MODULE.compare_coolers(actual, reference, 9)["pixels_checked"] == 3
    with pytest.raises(ValueError, match="conserve"):
        _MODULE.compare_coolers(actual, reference, 10)
    pixels["count"] = [3, 2, 4]
    altered = tmp_path / "altered.cool"
    cooler.create_cooler(str(altered), bins, pixels)
    with pytest.raises(AssertionError):
        _MODULE.compare_coolers(actual, altered, 9)


def test_environment_rejects_unisolated_base_environment(monkeypatch):
    monkeypatch.setattr(_MODULE.sys, "prefix", _MODULE.sys.base_prefix)
    with pytest.raises(ValueError, match="python -I"):
        _MODULE.verify_environment()
