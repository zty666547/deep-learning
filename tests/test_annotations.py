import pandas as pd
import pytest

from microc_foundation.annotations import read_structures_csv, read_structures_excel


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


def test_excel_reader_merges_structure_sheets_and_maps_chromosome(monkeypatch, tmp_path):
    workbook = tmp_path / "annotations.xlsx"
    workbook.touch()
    sheets = {
        "Supplementary Table 4": pd.DataFrame(
            {
                "OPCID_ID": ["OPCID_1"],
                "Chr": ["MG1655"],
                "Start": [100],
                "End": [300],
                "RedC signal WT": [2],
                "RedC signal WT HS": [4],
            }
        ),
        "Supplementary Table 5": pd.DataFrame(
            {
                "CHIN_ID": ["CHIN_1"],
                "Chr": ["MG1655"],
                "Start": [400],
                "Center": [450],
                "End": [500],
                "RedC signal WT": [1],
                "RedC signal WT HS": [3],
                "Resonse to H-NS knockout": ["unchanged"],
            }
        ),
        "Supplementary Table 6": pd.DataFrame(
            {
                "CHID_ID": ["CHID_1"],
                "Chr": ["MG1655"],
                "Start": [600],
                "End": [800],
                "RedC signal WT": [5],
                "RedC signal WT HS": [6],
            }
        ),
    }
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: sheets)

    result = read_structures_excel(
        str(workbook), chrom_map={"MG1655": "NC_000913.3"}
    )

    assert result["structure_type"].tolist() == ["OPCID", "CHIN", "CHID"]
    assert result["center"].tolist() == [200, 450, 700]
    assert result["chrom"].unique().tolist() == ["NC_000913.3"]
    assert result["source_chrom"].unique().tolist() == ["MG1655"]
