"""Audit cross-class overlaps in chromatin-structure annotations."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_cross_class_annotation_overlaps(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return positive-length annotation overlaps between different classes."""

    overlaps: list[dict[str, object]] = []
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["chrom"]),
            int(row["annotation_start"]),
            int(row["annotation_end"]),
        ),
    )
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            if left["chrom"] != right["chrom"]:
                break
            if int(right["annotation_start"]) >= int(left["annotation_end"]):
                break
            if left["true_class"] == right["true_class"]:
                continue
            overlap_start = max(
                int(left["annotation_start"]), int(right["annotation_start"])
            )
            overlap_end = min(
                int(left["annotation_end"]), int(right["annotation_end"])
            )
            if overlap_start < overlap_end:
                overlaps.append(
                    {
                        "left_structure_id": left["structure_id"],
                        "left_class": left["true_class"],
                        "right_structure_id": right["structure_id"],
                        "right_class": right["true_class"],
                        "overlap_bp": overlap_end - overlap_start,
                    }
                )
    return overlaps


def _build_components(
    structure_ids: list[str],
    overlaps: list[dict[str, object]],
) -> list[list[str]]:
    parent = {structure_id: structure_id for structure_id in structure_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    conflicted: set[str] = set()
    for overlap in overlaps:
        left_id = str(overlap["left_structure_id"])
        right_id = str(overlap["right_structure_id"])
        conflicted.update((left_id, right_id))
        union(left_id, right_id)
    grouped: dict[str, list[str]] = {}
    for structure_id in sorted(conflicted):
        grouped.setdefault(find(structure_id), []).append(structure_id)
    return sorted(grouped.values(), key=lambda values: (values[0], len(values)))


def audit_annotation_overlaps(structures: pd.DataFrame) -> dict[str, object]:
    """Build pair, component and exclusion-feasibility tables."""

    required = {
        "structure_id",
        "chrom",
        "start",
        "end",
        "structure_type",
        "split",
    }
    missing = sorted(required.difference(structures.columns))
    if missing:
        raise ValueError(f"structures are missing columns: {', '.join(missing)}")
    if structures["structure_id"].astype(str).duplicated().any():
        raise ValueError("structure_id values must be unique")

    frame = structures.copy()
    frame["structure_id"] = frame["structure_id"].astype(str)
    frame["structure_type"] = frame["structure_type"].astype(str)
    frame["split"] = frame["split"].astype(str)
    rows = [
        {
            "structure_id": str(row.structure_id),
            "true_class": str(row.structure_type),
            "chrom": str(row.chrom),
            "annotation_start": int(row.start),
            "annotation_end": int(row.end),
        }
        for row in frame.itertuples(index=False)
    ]
    overlaps = find_cross_class_annotation_overlaps(rows)
    lookup = frame.set_index("structure_id", drop=False)
    pair_rows: list[dict[str, object]] = []
    neighbors: dict[str, list[str]] = {
        structure_id: [] for structure_id in frame["structure_id"]
    }
    for overlap in overlaps:
        left_id = str(overlap["left_structure_id"])
        right_id = str(overlap["right_structure_id"])
        left = lookup.loc[left_id]
        right = lookup.loc[right_id]
        neighbors[left_id].append(right_id)
        neighbors[right_id].append(left_id)
        pair_rows.append(
            {
                **overlap,
                "chrom": str(left["chrom"]),
                "overlap_start": max(int(left["start"]), int(right["start"])),
                "overlap_end": min(int(left["end"]), int(right["end"])),
                "left_split": str(left["split"]),
                "right_split": str(right["split"]),
            }
        )

    components = _build_components(frame["structure_id"].tolist(), overlaps)
    component_by_id: dict[str, str] = {}
    component_rows: list[dict[str, object]] = []
    for index, members in enumerate(components, start=1):
        component_id = f"conflict_{index:03d}"
        selected = frame.loc[frame["structure_id"].isin(members)]
        for member in members:
            component_by_id[member] = component_id
        component_rows.append(
            {
                "component_id": component_id,
                "chrom": str(selected["chrom"].iloc[0]),
                "start": int(selected["start"].min()),
                "end": int(selected["end"].max()),
                "num_structures": len(members),
                "classes": ";".join(sorted(selected["structure_type"].unique())),
                "splits": ";".join(sorted(selected["split"].unique())),
                "structure_ids": ";".join(members),
            }
        )

    structure_rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        structure_id = str(row.structure_id)
        neighbor_ids = sorted(neighbors[structure_id])
        neighbor_classes = sorted(
            {str(lookup.loc[neighbor_id, "structure_type"]) for neighbor_id in neighbor_ids}
        )
        structure_rows.append(
            {
                "structure_id": structure_id,
                "chrom": str(row.chrom),
                "start": int(row.start),
                "end": int(row.end),
                "structure_type": str(row.structure_type),
                "split": str(row.split),
                "has_cross_class_overlap": bool(neighbor_ids),
                "cross_class_overlap_count": len(neighbor_ids),
                "overlap_classes": ";".join(neighbor_classes),
                "overlap_structure_ids": ";".join(neighbor_ids),
                "conflict_component_id": component_by_id.get(structure_id, ""),
            }
        )

    grouped_member_sets = [set(members) for members in components]
    grouped_ids = set().union(*grouped_member_sets) if grouped_member_sets else set()
    grouped_member_sets.extend(
        {structure_id}
        for structure_id in frame["structure_id"].tolist()
        if structure_id not in grouped_ids
    )
    region_specs: list[dict[str, object]] = []
    for members in grouped_member_sets:
        selected = frame.loc[frame["structure_id"].isin(members)]
        classes = sorted(selected["structure_type"].unique().tolist())
        region_specs.append(
            {
                "chrom": str(selected["chrom"].iloc[0]),
                "start": int(selected["start"].min()),
                "end": int(selected["end"].max()),
                "num_structures": len(members),
                "label_count": len(classes),
                "label_set": ";".join(classes),
                "is_multilabel": len(classes) > 1,
                "splits": ";".join(sorted(selected["split"].unique())),
                "structure_ids": ";".join(sorted(members)),
            }
        )
    region_specs.sort(
        key=lambda row: (str(row["chrom"]), int(row["start"]), int(row["end"]))
    )
    region_rows = [
        {"region_id": f"region_{index:03d}", **row}
        for index, row in enumerate(region_specs, start=1)
    ]

    retained_splits = ("train", "validation", "test")
    class_names = sorted(frame["structure_type"].unique().tolist())
    conflicted_ids = {key for key, values in neighbors.items() if values}
    by_split_class: dict[str, dict[str, dict[str, int | float]]] = {}
    remaining_after_exclusion: dict[str, dict[str, int]] = {}
    for split_name in retained_splits:
        by_split_class[split_name] = {}
        remaining_after_exclusion[split_name] = {}
        for class_name in class_names:
            subset = frame.loc[
                (frame["split"] == split_name)
                & (frame["structure_type"] == class_name)
            ]
            conflicted_count = int(subset["structure_id"].isin(conflicted_ids).sum())
            total = len(subset)
            by_split_class[split_name][class_name] = {
                "total": total,
                "conflicted": conflicted_count,
                "conflict_fraction": conflicted_count / total if total else 0.0,
            }
            remaining_after_exclusion[split_name][class_name] = total - conflicted_count

    pair_class_counts = Counter(
        "|".join(sorted((str(row["left_class"]), str(row["right_class"]))))
        for row in pair_rows
    )
    multilabel_regions = [row for row in region_rows if row["is_multilabel"]]
    summary = {
        "num_structures": len(frame),
        "num_cross_class_overlap_pairs": len(pair_rows),
        "num_conflicted_structures": len(conflicted_ids),
        "num_conflict_components": len(component_rows),
        "num_region_groups": len(region_rows),
        "num_multilabel_regions": len(multilabel_regions),
        "max_multilabel_region_span_bp": max(
            (int(row["end"]) - int(row["start"]) for row in multilabel_regions),
            default=0,
        ),
        "max_structures_per_multilabel_region": max(
            (int(row["num_structures"]) for row in multilabel_regions),
            default=0,
        ),
        "region_label_set_counts": dict(
            sorted(Counter(str(row["label_set"]) for row in region_rows).items())
        ),
        "pair_counts_by_class": dict(sorted(pair_class_counts.items())),
        "by_split_and_class": by_split_class,
        "remaining_after_exclusion": remaining_after_exclusion,
        "exclusion_preserves_all_classes": all(
            count > 0
            for split_counts in remaining_after_exclusion.values()
            for count in split_counts.values()
        ),
    }
    return {
        "summary": summary,
        "pairs": pair_rows,
        "structures": structure_rows,
        "components": component_rows,
        "regions": region_rows,
    }


def write_overlap_audit(audit: dict[str, object], output_dir: str) -> Path:
    """Write overlap tables, JSON summary and a conflict-rate figure."""

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(audit["summary"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for filename, key in (
        ("cross_class_overlap_pairs.csv", "pairs"),
        ("structure_conflicts.csv", "structures"),
        ("conflict_components.csv", "components"),
        ("region_groups.csv", "regions"),
    ):
        rows = audit[key]
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    summary = audit["summary"]
    by_split_class = summary["by_split_and_class"]
    splits = list(by_split_class)
    class_names = list(next(iter(by_split_class.values())))
    x_positions = np.arange(len(class_names))
    width = 0.8 / len(splits)
    figure, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for index, split_name in enumerate(splits):
        offset = (index - (len(splits) - 1) / 2) * width
        axis.bar(
            x_positions + offset,
            [
                by_split_class[split_name][class_name]["conflict_fraction"]
                for class_name in class_names
            ],
            width,
            label=split_name,
        )
    axis.set(
        xticks=x_positions,
        xticklabels=class_names,
        xlabel="Structure class",
        ylabel="Fraction with cross-class overlap",
        title="Cross-class annotation conflicts by split",
        ylim=(0, 1),
    )
    axis.legend()
    figure.savefig(output / "conflict_rates.png", dpi=180)
    plt.close(figure)
    return output
