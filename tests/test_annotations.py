import pandas as pd
import pytest

from microc_foundation.annotations import read_structures_csv


def test_reader_normalizes_common_column_aliases(tmp_path):
    path = tmp_path / "structures.csv"
    pd.DataFrame(
        {"chr": ["chr1"], "start": [10], "end": [20], "type": ["CHIN"]}
    ).to_csv(path, index=False)
    result = read_structures_csv(str(path))
    assert list(result[["chrom", "start", "end", "structure_type"]].iloc[0]) == [
        "chr1",
        10,
        20,
        "CHIN",
    ]


def test_reader_rejects_reversed_interval(tmp_path):
    path = tmp_path / "structures.csv"
    pd.DataFrame({"chrom": ["chr1"], "start": [20], "end": [10]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="greater"):
        read_structures_csv(str(path))

