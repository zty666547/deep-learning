"""Fixed-coordinate transfer audit of candidate contact-map texture across conditions."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from .io import open_cooler
from .replicate_audit import distance_controlled_correlations


def audit_candidate_condition_transfer(
    conditions: dict[str, list[str]], candidates: pd.DataFrame, output_dir: str, *,
    reference_condition: str = "WT", minimum_offset: int = 5,
    normalization: str = "raw_counts",
) -> dict:
    """Compare fixed candidates within and between conditions, without reselection.

    Each condition must have exactly two independent replicate files. For each
    candidate, emits the within-condition replicate pair and all four
    reference-to-other-condition replicate pairs. The four cross-condition
    pairs share samples and are descriptive, not independent observations.
    """
    required = {"candidate_region_id", "chrom", "start", "end", "stable_candidate_key"}
    if required.difference(candidates.columns):
        raise ValueError("candidate table is missing required columns")
    if candidates.empty or candidates.candidate_region_id.duplicated().any():
        raise ValueError("candidate regions must be non-empty and uniquely identified")
    if reference_condition not in conditions or len(conditions) < 2:
        raise ValueError("reference and at least one comparison condition are required")
    if any(len(paths) != 2 for paths in conditions.values()):
        raise ValueError("each condition must have exactly two biological replicates")
    paths = [str(Path(path).resolve()) for group in conditions.values() for path in group]
    if len(paths) != len(set(paths)):
        raise ValueError("a replicate file cannot count as multiple conditions")
    if minimum_offset < 1:
        raise ValueError("minimum_offset must be positive")
    if normalization not in {"raw_counts", "library_cpm"}:
        raise ValueError("normalization must be 'raw_counts' or 'library_cpm'")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with ExitStack() as stack:
        coolers = {
            condition: [stack.enter_context(open_cooler(path)) for path in group]
            for condition, group in conditions.items()
        }
        first = next(iter(coolers.values()))[0]
        bin_size = int(first.binsize)
        chromsizes = first.chromsizes
        library_sums = {
            condition: [float(cool.info["sum"]) for cool in group]
            for condition, group in coolers.items()
        }
        if any(not np.isfinite(total) or total <= 0
               for group in library_sums.values() for total in group):
            raise ValueError("all libraries must have positive finite total contact counts")
        for group in coolers.values():
            for cool in group:
                if (int(cool.binsize) != bin_size or not cool.chromsizes.equals(chromsizes)
                        or cool.chromnames != first.chromnames):
                    raise ValueError("all conditions must share resolution and chromosome coordinates")

        for candidate in candidates.itertuples(index=False):
            start, end = int(candidate.start), int(candidate.end)
            if start < 0 or end <= start or start % bin_size or end % bin_size:
                raise ValueError("candidate intervals must be positive and bin-aligned")
            if candidate.chrom not in chromsizes.index or end > int(chromsizes[candidate.chrom]):
                raise ValueError("candidate interval exceeds reference chromosome coordinates")
            expected_size = (end - start) // bin_size
            if expected_size <= minimum_offset + 2:
                raise ValueError("candidate interval is too short for the selected distance offset")
            region = f"{candidate.chrom}:{start}-{end}"
            matrices = {
                condition: [
                    np.asarray(cool.matrix(balance=False).fetch(region), dtype=float)
                    * (1_000_000 / library_sums[condition][index]
                       if normalization == "library_cpm" else 1.0)
                    for index, cool in enumerate(group)
                ]
                for condition, group in coolers.items()
            }
            if any(matrix.shape != (expected_size, expected_size)
                   for group in matrices.values() for matrix in group):
                raise ValueError("candidate fetch shape does not match aligned coordinates")

            pairs: list[tuple[str, str, np.ndarray, np.ndarray]] = []
            for condition, group in matrices.items():
                pairs.append((f"within:{condition}", "rep1_vs_rep2", group[0], group[1]))
            for condition, group in matrices.items():
                if condition == reference_condition:
                    continue
                for reference_rep, reference_matrix in enumerate(matrices[reference_condition], 1):
                    for condition_rep, condition_matrix in enumerate(group, 1):
                        pairs.append((f"{reference_condition}_vs_{condition}",
                                      f"{reference_condition}_rep{reference_rep}_vs_{condition}_rep{condition_rep}",
                                      reference_matrix, condition_matrix))
            for contrast, replicate_pair, left, right in pairs:
                rows.append({
                    "candidate_region_id": candidate.candidate_region_id,
                    "stable_candidate_key": candidate.stable_candidate_key,
                    "chrom": candidate.chrom, "start": start, "end": end,
                    "condition_contrast": contrast, "replicate_pair": replicate_pair,
                    **distance_controlled_correlations(left, right, minimum_offset=minimum_offset),
                })

    table = pd.DataFrame(rows)
    table.to_csv(output / "candidate_condition_pairwise.csv", index=False)
    metric_names = ("full_log_correlation", "offdiagonal_log_correlation",
                    "distance_centered_correlation")
    summaries = []
    for (candidate_id, contrast), group in table.groupby(
            ["candidate_region_id", "condition_contrast"], sort=False):
        record = {"candidate_region_id": candidate_id, "condition_contrast": contrast,
                  "num_pairs": len(group)}
        for metric in metric_names:
            values = group[metric].dropna().to_numpy(float)
            record[f"{metric}_median"] = float(np.median(values)) if len(values) else None
            record[f"{metric}_q25"] = float(np.quantile(values, .25)) if len(values) else None
            record[f"{metric}_q75"] = float(np.quantile(values, .75)) if len(values) else None
        summaries.append(record)
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(output / "candidate_condition_summary.csv", index=False)
    summary = {
        "protocol": "fixed task2-main-v1 candidate coordinates; within-condition replicate and all reference cross-replicate pairs",
        "normalization": normalization,
        "library_total_contact_counts": library_sums,
        "reference_condition": reference_condition,
        "conditions": list(conditions), "replicates_per_condition": 2,
        "bin_size_bp": bin_size, "minimum_offset_bins": minimum_offset,
        "minimum_distance_bp": minimum_offset * bin_size,
        "num_candidate_regions": len(candidates),
        "candidate_keys": candidates.stable_candidate_key.astype(str).tolist(),
        "limitations": [
            "Candidates were originally selected using WT data; WT results retain selection bias",
            "Cross-condition replicate pairs share samples and are not independent",
            "Only five fixed candidates are assessed; this is descriptive transfer, not differential testing",
            "No p-values, causal genotype effects, or new-structure claims are produced",
            "Distance centering is within-window and is not genome-wide observed/expected normalization",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
