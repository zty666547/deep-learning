import gzip
import shutil

import cooler
import pandas as pd
import pytest

from microc_foundation.io import open_cooler
from microc_foundation.windows import GenomicWindow, fetch_window


@pytest.fixture
def tiny_cool(tmp_path):
    path = tmp_path / "tiny.cool"
    bins = pd.DataFrame(
        {"chrom": ["chr1"] * 3, "start": [0, 1000, 2000], "end": [1000, 2000, 3000]}
    )
    pixels = pd.DataFrame(
        {"bin1_id": [0, 0, 1, 1, 2], "bin2_id": [0, 1, 1, 2, 2], "count": [5, 2, 4, 1, 3]}
    )
    cooler.create_cooler(str(path), bins, pixels)
    return path


def test_open_and_fetch_cool(tiny_cool):
    with open_cooler(str(tiny_cool)) as contact_map:
        matrix, bin_size = fetch_window(
            contact_map, GenomicWindow("chr1", 0, 3000), balance=False
        )
    assert matrix.shape == (3, 3)
    assert bin_size == 1000


def test_open_gzipped_cool(tiny_cool):
    compressed = tiny_cool.with_suffix(".cool.gz")
    with tiny_cool.open("rb") as source, gzip.open(compressed, "wb") as target:
        shutil.copyfileobj(source, target)
    with open_cooler(str(compressed)) as contact_map:
        assert contact_map.chromnames == ["chr1"]

