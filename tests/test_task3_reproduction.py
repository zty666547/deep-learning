import copy
import importlib.util
from pathlib import Path

import pytest
from PIL import Image

_SCRIPT = Path(__file__).parents[1] / "scripts/reproduce_task3.py"
_SPEC = importlib.util.spec_from_file_location("task3_reproduction", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def manifest():
    return {"chrom": "chr1", "length_bp": 25, "tile_bp": 10, "num_tiles": 3,
            "tiles": [{"chrom": "chr1", "start": a, "end": b,
                       "file": f"tile_{a:07d}_{b:07d}.png"} for a, b in [(0, 10), (10, 20), (20, 25)]]}


def test_path_normalization_preserves_identity_and_original():
    original = {"sources": [{"path": "/old/rep1.cool", "condition": "WT"}]}
    updated = _MODULE.normalize_source_paths(original)
    assert updated == {"sources": [{"path": "rep1.cool", "condition": "WT"}]}
    assert original["sources"][0]["path"] == "/old/rep1.cool"


@pytest.mark.parametrize("field,value", [("start", 11), ("end", 19), ("chrom", "chr2"), ("file", "wrong.png")])
def test_manifest_rejects_gap_overlap_and_wrong_identity(field, value):
    data = manifest()
    _MODULE.validate_manifest(data)
    data["tiles"][1][field] = value
    with pytest.raises(ValueError):
        _MODULE.validate_manifest(data)


def test_manifest_rejects_missing_last_segment():
    data = manifest()
    data["tiles"].pop()
    data["num_tiles"] = 2
    with pytest.raises(ValueError, match="cover"):
        _MODULE.validate_manifest(data)


def test_pixel_comparison_checks_all_images_and_changes(tmp_path):
    actual, reference = tmp_path / "actual", tmp_path / "reference"
    actual.mkdir()
    reference.mkdir()
    data = manifest()
    for tile in data["tiles"]:
        for output in (actual, reference):
            Image.new("RGBA", (3, 4), (10, 20, 30, 255)).save(output / tile["file"])
    assert _MODULE.compare_tile_pixels(actual, reference, data)["images_checked"] == 3
    changed = actual / data["tiles"][-1]["file"]
    Image.new("RGBA", (3, 4), (11, 20, 30, 255)).save(changed)
    with pytest.raises(ValueError, match="pixels"):
        _MODULE.compare_tile_pixels(actual, reference, data)
    changed.unlink()
    with pytest.raises(ValueError, match="inventory"):
        _MODULE.compare_tile_pixels(actual, reference, data)


def test_pixel_comparison_rejects_corruption_and_dimensions(tmp_path):
    data = copy.deepcopy(manifest())
    actual, reference = tmp_path / "actual", tmp_path / "reference"
    actual.mkdir()
    reference.mkdir()
    for tile in data["tiles"]:
        Image.new("RGB", (3, 4)).save(actual / tile["file"])
        Image.new("RGB", (3, 4)).save(reference / tile["file"])
    Image.new("RGB", (4, 4)).save(actual / data["tiles"][0]["file"])
    with pytest.raises(ValueError, match="dimensions"):
        _MODULE.compare_tile_pixels(actual, reference, data)
    (actual / data["tiles"][0]["file"]).write_bytes(b"broken PNG")
    with pytest.raises(OSError):
        _MODULE.compare_tile_pixels(actual, reference, data)
