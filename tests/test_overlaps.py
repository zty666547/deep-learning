import pandas as pd
import pytest

from microc_foundation.overlaps import (
    audit_annotation_overlaps,
    find_cross_class_annotation_overlaps,
)


def test_cross_class_annotation_overlaps_exclude_same_class_and_touching():
    rows = [
        {
            "structure_id": "A",
            "true_class": "CHID",
            "chrom": "chr1",
            "annotation_start": 10,
            "annotation_end": 30,
        },
        {
            "structure_id": "B",
            "true_class": "CHIN",
            "chrom": "chr1",
            "annotation_start": 20,
            "annotation_end": 40,
        },
        {
            "structure_id": "C",
            "true_class": "CHID",
            "chrom": "chr1",
            "annotation_start": 25,
            "annotation_end": 35,
        },
        {
            "structure_id": "D",
            "true_class": "OPCID",
            "chrom": "chr1",
            "annotation_start": 40,
            "annotation_end": 50,
        },
    ]
    overlaps = find_cross_class_annotation_overlaps(rows)
    assert overlaps == [
        {
            "left_structure_id": "A",
            "left_class": "CHID",
            "right_structure_id": "B",
            "right_class": "CHIN",
            "overlap_bp": 10,
        },
        {
            "left_structure_id": "B",
            "left_class": "CHIN",
            "right_structure_id": "C",
            "right_class": "CHID",
            "overlap_bp": 10,
        },
    ]


def test_overlap_audit_reports_exclusion_failure():
    frame = pd.DataFrame(
        {
            "structure_id": ["A", "B", "C", "D"],
            "chrom": ["chr1"] * 4,
            "start": [10, 20, 100, 200],
            "end": [30, 40, 120, 220],
            "structure_type": ["CHID", "CHIN", "CHIN", "CHID"],
            "split": ["test", "test", "train", "train"],
        }
    )
    audit = audit_annotation_overlaps(frame)
    assert audit["summary"]["num_cross_class_overlap_pairs"] == 1
    assert audit["summary"]["num_conflicted_structures"] == 2
    assert audit["summary"]["num_region_groups"] == 3
    assert audit["summary"]["num_multilabel_regions"] == 1
    assert audit["summary"]["region_label_set_counts"] == {
        "CHID": 1,
        "CHID;CHIN": 1,
        "CHIN": 1,
    }
    assert audit["summary"]["remaining_after_exclusion"]["test"] == {
        "CHID": 0,
        "CHIN": 0,
    }
    assert audit["summary"]["exclusion_preserves_all_classes"] is False


def test_overlap_audit_requires_unique_ids():
    frame = pd.DataFrame(
        {
            "structure_id": ["A", "A"],
            "chrom": ["chr1", "chr1"],
            "start": [10, 20],
            "end": [30, 40],
            "structure_type": ["CHID", "CHIN"],
            "split": ["train", "test"],
        }
    )
    with pytest.raises(ValueError, match="unique"):
        audit_annotation_overlaps(frame)
