import json
import runpy
from pathlib import Path

import pytest

validate = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_submission.py"))[
    "validate_super_resolution_evidence"
]


def _evidence(tmp_path):
    seeds = [2026, 2027, 2028]
    metrics = {"psnr": 20.0, "ssim": 0.5, "mse": 0.2}
    runs = []
    for seed in seeds:
        directory = tmp_path / f"seed_{seed}"
        directory.mkdir()
        run = {
            "seed": seed, "intensity_correction": "log1p(expm1(low_log_counts)/4)",
            "ssim_definition": "7x7 local SSIM", "sample_counts": {"test": 1},
            "target_shape": [1, 8, 8], "bicubic_baseline": metrics, "cnn": metrics,
        }
        (directory / "metrics.json").write_text(json.dumps(run))
        for artifact in ("model.pt", "history.csv", "example_comparison.png"):
            (directory / artifact).touch()
        (directory / "per_sample_metrics.csv").write_text(
            "structure_id,replicate_index,method,psnr,ssim,mse\n"
            "A,1,bicubic,20,0.5,0.2\nA,1,cnn,20,0.5,0.2\n"
        )
        runs.append(run)
    summary = {
        "seeds": seeds, "runs": runs,
        "aggregate": {
            method: {metric: {"mean": value, "std": 0} for metric, value in metrics.items()}
            for method in ("bicubic_baseline", "cnn")
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    return summary


def test_submission_recomputes_valid_evidence(tmp_path):
    assert validate(tmp_path) == _evidence_result(tmp_path)


def _evidence_result(tmp_path):
    return json.loads((tmp_path / "summary.json").read_text())


@pytest.fixture(autouse=True)
def evidence(tmp_path):
    _evidence(tmp_path)


def test_submission_rejects_tampered_aggregate(tmp_path):
    summary = _evidence_result(tmp_path)
    summary["aggregate"]["cnn"]["psnr"]["mean"] = 30
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="aggregate"):
        validate(tmp_path)


def test_submission_rejects_old_intensity_protocol(tmp_path):
    summary = _evidence_result(tmp_path)
    summary["runs"][0]["intensity_correction"] = "none"
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "seed_2026/metrics.json").write_text(json.dumps(summary["runs"][0]))
    with pytest.raises(ValueError, match="corrected count scale"):
        validate(tmp_path)


def test_submission_rejects_mismatched_method_ids(tmp_path):
    path = tmp_path / "seed_2026/per_sample_metrics.csv"
    path.write_text(path.read_text().replace("A,1,cnn", "B,1,cnn"))
    with pytest.raises(ValueError, match="identities"):
        validate(tmp_path)


def test_submission_rejects_nonfinite_metrics(tmp_path):
    path = tmp_path / "seed_2026/per_sample_metrics.csv"
    path.write_text(path.read_text().replace("cnn,20", "cnn,nan"))
    with pytest.raises(ValueError, match="non-finite"):
        validate(tmp_path)


def test_submission_rejects_duplicate_seeds(tmp_path):
    summary = _evidence_result(tmp_path)
    summary["seeds"][1] = 2026
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="distinct seeds"):
        validate(tmp_path)
