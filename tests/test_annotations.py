import pandas as pd
import pytest

from microc_foundation.annotations import read_structures_csv, read_structures_excel
from microc_foundation.coordinate_audit import audit_annotation_bin_alignment


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


def test_coordinate_audit_compares_direct_and_one_based_adjusted_boundaries():
    structures = pd.DataFrame(
        {
            "structure_id": ["A", "B"],
            "chrom": ["chr1", "chr1"],
            "start": [10, 30],
            "end": [30, 50],
            "structure_type": ["CHIN", "OPCID"],
        }
    )
    bins = pd.DataFrame(
        {
            "chrom": ["chr1"] * 5,
            "start": [0, 10, 20, 30, 40],
            "end": [10, 20, 30, 40, 50],
        }
    )

    result = audit_annotation_bin_alignment(structures, bins, bin_size_bp=10)

    assert result["both_direct_boundaries_count"] == 2
    assert result["one_based_adjusted_start_boundary_count"] == 0
    assert result["start_and_end_multiple_of_resolution_count"] == 2


def test_coordinate_audit_rejects_wrong_reference_chromosome():
    structures = pd.DataFrame(
        {
            "structure_id": ["A"],
            "chrom": ["chr2"],
            "start": [10],
            "end": [20],
            "structure_type": ["CHIN"],
        }
    )
    bins = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10]})

    result = audit_annotation_bin_alignment(structures, bins, bin_size_bp=10)

    assert result["valid_bounds_count"] == 0
    assert result["both_direct_boundaries_count"] == 0
