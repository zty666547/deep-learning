import pandas as pd
import pytest

from microc_foundation.splitting import (
    assign_contiguous_splits,
    validate_split_class_coverage,
)


def test_contiguous_split_excludes_boundary_crossing_windows():
    structures = pd.DataFrame(
        {
            "structure_id": ["a", "b", "c", "d"],
            "chrom": ["chr1"] * 4,
            "center": [100, 699, 800, 900],
            "structure_type": ["A", "A", "A", "A"],
        }
    )
    result = assign_contiguous_splits(
        structures,
        1000,
        train_fraction=0.7,
        validation_fraction=0.15,
        window_size_bp=40,
    )
    assert result["split"].tolist() == [
        "train",
        "excluded_boundary",
        "validation",
        "test",
    ]


def test_split_coverage_requires_each_class_in_each_split():
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "validation", "test"],
            "structure_type": ["A", "B", "A", "A"],
        }
    )
    with pytest.raises(ValueError, match="validation split is missing classes: B"):
        validate_split_class_coverage(frame)
