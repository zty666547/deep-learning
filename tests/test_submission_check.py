import csv
import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "check_submission.py"
_SPEC = importlib.util.spec_from_file_location("check_submission", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
validate_task2_validation_evidence = _MODULE.validate_task2_validation_evidence


CONTRASTS = (
    "within:WT", "within:delta_stpA", "within:delta_hns_delta_stpA",
    "WT_vs_delta_stpA", "WT_vs_delta_hns_delta_stpA",
)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def task2_evidence(tmp_path):
    root = tmp_path / "outputs"
    keys = [f"candidate_{index:03d}" for index in range(1, 6)]
    summary_rows = []
    resolution_rows = []
    for candidate in keys:
        for contrast in CONTRASTS:
            row = {"candidate_region_id": candidate, "condition_contrast": contrast,
                   "distance_centered_correlation_median": "0.5"}
            summary_rows.append(row)
            for resolution in (80, 160, 320):
                resolution_rows.append({"resolution_bp": resolution, **row})
    for name, normalization in (("task2_condition_transfer", "raw_counts"),
                                ("task2_condition_transfer_cpm", "library_cpm")):
        path = root / name
        write_json(path / "summary.json", {"normalization": normalization, "candidate_keys": keys})
        write_csv(path / "candidate_condition_summary.csv", list(summary_rows[0]), summary_rows)
    path = root / "task2_resolution_audit"
    write_json(path / "summary.json", {"requested_resolutions_bp": [80, 160, 320],
                                        "num_candidate_regions": 5})
    write_csv(path / "candidate_resolution_summary.csv", list(resolution_rows[0]), resolution_rows)
    path = root / "task2_mutant_background"
    write_json(path / "summary.json", {"num_candidate_representatives": 5,
                                        "num_unannotated_background": 329,
                                        "normalizations": ["raw_counts", "library_cpm"]})
    background_rows = []
    for normalization in ("raw_counts", "library_cpm"):
        for condition in ("delta_stpA", "delta_hns_delta_stpA"):
            background_rows.extend([
                {"normalization": normalization, "condition": condition,
                 "group": "candidate_representative", "distance_centered_correlation_median": "0.6"},
                {"normalization": normalization, "condition": condition,
                 "group": "unannotated_background", "distance_centered_correlation_median": "0.3"},
            ])
    write_csv(path / "candidate_background_summary.csv", list(background_rows[0]), background_rows)
    return root


def test_submission_evidence_check_accepts_complete_fixed_results(task2_evidence):
    result = validate_task2_validation_evidence(task2_evidence)
    assert result == {"num_candidates": 5, "num_resolution_levels": 3,
                      "num_background_windows": 329, "normalizations": 2}


def test_submission_evidence_check_rejects_160bp_mismatch(task2_evidence):
    path = task2_evidence / "task2_resolution_audit/candidate_resolution_summary.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = next(row for row in rows if row["resolution_bp"] == "160")
    selected["distance_centered_correlation_median"] = "0.7"
    write_csv(path, list(rows[0]), rows)
    with pytest.raises(ValueError, match="disagrees"):
        validate_task2_validation_evidence(task2_evidence)


def test_submission_evidence_check_rejects_background_reversal(task2_evidence):
    path = task2_evidence / "task2_mutant_background/candidate_background_summary.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["normalization"] == "library_cpm" and row["condition"] == "delta_stpA" and row["group"] == "candidate_representative":
            row["distance_centered_correlation_median"] = "0.2"
    write_csv(path, list(rows[0]), rows)
    with pytest.raises(ValueError, match="exceed background"):
        validate_task2_validation_evidence(task2_evidence)
