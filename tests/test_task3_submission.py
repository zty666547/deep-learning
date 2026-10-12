"""Synthetic file fixtures validate acceptance logic, not experimental results."""

import csv
import importlib.util
import io
import json
from pathlib import Path

import pytest
from PIL import Image

_SCRIPT = Path(__file__).parents[1] / "scripts/check_submission.py"
_SPEC = importlib.util.spec_from_file_location("task3_submission", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
validate = _MODULE.validate_task3_evidence


def save_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "condition-comparison"
    (root / "tiles").mkdir(parents=True)
    names = ["WT 37C", "ΔstpA", "ΔhnsΔstpA"]
    sources = []
    for index in range(6):
        source = tmp_path / f"replicate_{index}.cool"
        source.touch()  # Existence only; checker does not certify COOL contents.
        sources.append({"condition": names[index // 2], "path": str(source),
                        "resolution_bp": 160, "library_sum": 10})
    provenance = {"sources": sources, "conditions": names, "band_bp": 10_000,
                  "circular": True, "replicate_weighting": "equal within condition",
                  "condition_weighting": "equal across conditions",
                  "normalization": "off-diagonal band row sum / library sum * 1e6",
                  "gene_count": 4506, "structure_counts": {"CHIN": 250, "OPCID": 68, "CHID": 26},
                  "tile_bp": 10_000, "matrix_resolution_bp": 160}
    save_json(root / "provenance.json", provenance)
    fields = ["chrom", "start", "end", "center", *names,
              *[f"{name}__replicate_sd" for name in names], "mean_cpm"]
    with (root / "condition_tracks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for start in range(0, 4_641_652, 160):
            end = min(start + 160, 4_641_652)
            writer.writerow({"chrom": "NC_000913.3", "start": start, "end": end,
                             "center": (start + end) / 2, **dict.fromkeys(names, 2),
                             **dict.fromkeys([f"{name}__replicate_sd" for name in names], 0),
                             "mean_cpm": 2})
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(buffer, format="PNG")
    tiles = []
    for start in range(0, 4_641_652, 10_000):
        end = min(start + 10_000, 4_641_652)
        name = f"tile_{start:07d}_{end:07d}.png"
        (root / "tiles" / name).write_bytes(buffer.getvalue())
        tiles.append({"chrom": "NC_000913.3", "start": start, "end": end,
                      "file": name, "gene_segments": 0, "structures": 0})
    save_json(root / "tiles/manifest.json", {**provenance, "chrom": "NC_000913.3",
                                            "length_bp": 4_641_652, "num_tiles": 465, "tiles": tiles})
    return root


def test_task3_accepts_current_465_tiles_and_clipped_endpoint(evidence):
    result = validate(evidence)
    assert result == {"num_conditions": 3, "num_replicates": 6, "num_bins": 29_011,
                      "num_tiles": 465, "length_bp": 4_641_652, "last_tile": [4_640_000, 4_641_652]}


@pytest.mark.parametrize("column,value,match", [
    ("start", "1", "coordinates"), ("mean_cpm", "3", "mean"), ("WT 37C", "nan", "signals"),
])
def test_task3_rejects_invalid_bin_tracks(evidence, column, value, match):
    path = evidence / "condition_tracks.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0][column] = value
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match=match):
        validate(evidence)


@pytest.mark.parametrize("damage", ["gap", "missing", "corrupt"])
def test_task3_rejects_invalid_tile_evidence(evidence, damage):
    path = evidence / "tiles/manifest.json"
    manifest = json.loads(path.read_text())
    image = evidence / "tiles" / manifest["tiles"][0]["file"]
    if damage == "gap":
        manifest["tiles"][1]["start"] += 1
        save_json(path, manifest)
    elif damage == "missing":
        image.unlink()
    else:
        image.write_bytes(b"not a PNG")
    with pytest.raises(ValueError, match="Task 3"):
        validate(evidence)


def test_task3_rejects_reused_replicate(evidence):
    path = evidence / "provenance.json"
    provenance = json.loads(path.read_text())
    provenance["sources"][1]["path"] = provenance["sources"][0]["path"]
    save_json(path, provenance)
    manifest_path = evidence / "tiles/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"] = provenance["sources"]
    save_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="six distinct"):
        validate(evidence)


def test_task3_rejects_stale_manifest_provenance(evidence):
    path = evidence / "tiles/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["conditions"] = ["WT 37C"]
    save_json(path, manifest)
    with pytest.raises(ValueError, match="manifest disagrees"):
        validate(evidence)
