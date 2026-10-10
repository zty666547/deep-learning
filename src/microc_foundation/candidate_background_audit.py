"""Independent-condition candidate-versus-unannotated-background controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .replicate_audit import audit_replicate_consistency


def select_candidate_and_background_windows(
    windows: pd.DataFrame, candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Select fixed candidate representatives and non-overlapping unannotated windows."""
    required_windows = {"window_id", "chrom", "start", "end", "quality_pass", "known_overlap_bp"}
    required_candidates = {"candidate_region_id", "chrom", "start", "end", "representative_window_id"}
    if required_windows.difference(windows.columns) or required_candidates.difference(candidates.columns):
        raise ValueError("window or candidate table is missing required columns")
    if windows.window_id.duplicated().any() or candidates.candidate_region_id.duplicated().any():
        raise ValueError("window and candidate identifiers must be unique")
    quality = windows.quality_pass.astype(str).str.lower().eq("true")
    selected = windows.loc[quality].copy()
    representatives = set(candidates.representative_window_id.astype(str))
    if not representatives.issubset(set(selected.window_id.astype(str))):
        raise ValueError("candidate representatives must be QC-passed windows")

    candidate_neighborhood = np.zeros(len(selected), dtype=bool)
    for candidate in candidates.itertuples(index=False):
        candidate_neighborhood |= (
            selected.chrom.eq(candidate.chrom).to_numpy()
            & selected.start.lt(int(candidate.end)).to_numpy()
            & selected.end.gt(int(candidate.start)).to_numpy()
        )
    is_representative = selected.window_id.astype(str).isin(representatives)
    has_known_overlap = pd.to_numeric(selected.known_overlap_bp, errors="coerce").fillna(0).gt(0)
    background = selected.loc[~candidate_neighborhood & ~has_known_overlap & ~is_representative]
    representatives_frame = selected.loc[is_representative]
    if len(representatives_frame) != len(representatives) or background.empty:
        raise ValueError("candidate representatives and unannotated controls must be non-empty and unique")
    return pd.concat([representatives_frame, background], ignore_index=True).sort_values(
        ["chrom", "start", "window_id"], kind="stable",
    ).reset_index(drop=True)


def audit_mutant_candidate_background(
    conditions: dict[str, list[str]], windows: pd.DataFrame, candidates: pd.DataFrame,
    output_dir: str, *, minimum_offset: int = 5,
) -> dict:
    """Compare candidate representatives with non-overlapping background in mutants."""
    if not conditions or any(len(paths) != 2 for paths in conditions.values()):
        raise ValueError("each independent condition must have exactly two replicates")
    paths = [str(Path(path).resolve()) for group in conditions.values() for path in group]
    if len(paths) != len(set(paths)):
        raise ValueError("a replicate file cannot count as multiple conditions")
    selected = select_candidate_and_background_windows(windows, candidates)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = []
    for normalization in ("raw_counts", "library_cpm"):
        for condition, cool_paths in conditions.items():
            condition_output = output / normalization / condition
            audit_replicate_consistency(
                cool_paths, selected, candidates, str(condition_output),
                minimum_offset=minimum_offset, normalization=normalization,
            )
            table = pd.read_csv(condition_output / "window_replication_audit.csv")
            table.insert(0, "normalization", normalization)
            table.insert(0, "condition", condition)
            tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(output / "candidate_background_window_metrics.csv", index=False)
    metrics = ("full_log_correlation", "offdiagonal_log_correlation",
               "distance_centered_correlation")
    rows = []
    for (normalization, condition, group_name), group in combined.groupby(
            ["normalization", "condition", "group"], sort=False):
        record = {"normalization": normalization, "condition": condition,
                  "group": group_name, "num_windows": len(group)}
        for metric in metrics:
            values = group[metric].dropna().to_numpy(float)
            record[f"{metric}_median"] = float(np.median(values)) if len(values) else None
            record[f"{metric}_q25"] = float(np.quantile(values, .25)) if len(values) else None
            record[f"{metric}_q75"] = float(np.quantile(values, .75)) if len(values) else None
        rows.append(record)
    summary_table = pd.DataFrame(rows)
    summary_table.to_csv(output / "candidate_background_summary.csv", index=False)
    summary = {
        "protocol": "fixed WT-selected representative windows versus non-overlapping unannotated background in independent mutant replicates",
        "conditions": list(conditions), "replicates_per_condition": 2,
        "normalizations": ["raw_counts", "library_cpm"],
        "num_candidate_representatives": int(selected.window_id.isin(
            candidates.representative_window_id.astype(str),
        ).sum()),
        "num_unannotated_background": int((~selected.window_id.isin(
            candidates.representative_window_id.astype(str),
        )).sum()),
        "bin_size_bp": 160, "minimum_offset_bins": minimum_offset,
        "minimum_distance_bp": 160 * minimum_offset,
        "limitations": [
            "WT was used to select candidates, but the mutation-condition replicates were not used for that selection",
            "Unannotated background is not a confirmed biological negative",
            "Background windows may overlap one another and are not independent samples",
            "Only five candidate representatives and two replicates per condition are available",
            "Descriptive comparison only; no p-values, causal claim, or new-structure claim",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
