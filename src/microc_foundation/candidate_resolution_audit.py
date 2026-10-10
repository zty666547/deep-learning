"""Resolution sensitivity of fixed candidate-window contact texture."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import pool_square_matrix
from .io import open_cooler
from .replicate_audit import distance_controlled_correlations


def audit_candidate_resolution_robustness(
    conditions: dict[str, list[str]], candidates: pd.DataFrame, output_dir: str, *,
    resolutions: tuple[int, ...] = (80, 160, 320), distance_threshold_bp: int = 800,
    reference_condition: str = "WT",
) -> dict:
    """Pool fixed 10 bp candidate submatrices locally and compare resolutions.

    The same genomic interval and biological source files are reused at every
    resolution. This tests binning sensitivity, not independent biological
    replication; windows must align exactly to each requested bin grid.
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
    if not resolutions or any(value <= 0 for value in resolutions) or len(set(resolutions)) != len(resolutions):
        raise ValueError("resolutions must be unique positive integers")
    if distance_threshold_bp <= 0:
        raise ValueError("distance threshold must be positive")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with ExitStack() as stack:
        coolers = {
            condition: [stack.enter_context(open_cooler(path)) for path in group]
            for condition, group in conditions.items()
        }
        first = next(iter(coolers.values()))[0]
        source_resolution = int(first.binsize)
        chromsizes = first.chromsizes
        for group in coolers.values():
            for cool in group:
                if (int(cool.binsize) != source_resolution
                        or not cool.chromsizes.equals(chromsizes)
                        or cool.chromnames != first.chromnames):
                    raise ValueError("source maps must share resolution and chromosome coordinates")
        if any(value < source_resolution or value % source_resolution for value in resolutions):
            raise ValueError("requested resolutions must be integer multiples of source resolution")

        for candidate in candidates.itertuples(index=False):
            start, end = int(candidate.start), int(candidate.end)
            if start < 0 or end <= start or start % source_resolution or end % source_resolution:
                raise ValueError("candidate intervals must be positive and source-bin aligned")
            if candidate.chrom not in chromsizes.index or end > int(chromsizes[candidate.chrom]):
                raise ValueError("candidate interval exceeds reference chromosome coordinates")
            for resolution in resolutions:
                if start % resolution or end % resolution:
                    raise ValueError("candidate intervals must align to every requested resolution")
                minimum_offset = int(np.ceil(distance_threshold_bp / resolution))
                matrix_size = (end - start) // resolution
                if matrix_size <= minimum_offset + 2:
                    raise ValueError("candidate interval is too short at a requested resolution")
                region = f"{candidate.chrom}:{start}-{end}"
                factor = resolution // source_resolution
                matrices = {}
                for condition, group in coolers.items():
                    pooled = []
                    for cool in group:
                        dense = np.asarray(cool.matrix(balance=False).fetch(region), dtype=float)
                        source_size = (end - start) // source_resolution
                        if dense.shape != (source_size, source_size):
                            raise ValueError("source fetch shape does not match candidate coordinates")
                        matrix = pool_square_matrix(dense, factor, method="sum")
                        if matrix.shape != (matrix_size, matrix_size):
                            raise ValueError("pooled matrix shape does not match requested resolution")
                        pooled.append(matrix)
                    matrices[condition] = pooled

                pairs = []
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
                        "resolution_bp": resolution,
                        "distance_threshold_bp": distance_threshold_bp,
                        "minimum_offset_bins": minimum_offset,
                        "actual_minimum_distance_bp": minimum_offset * resolution,
                        "condition_contrast": contrast, "replicate_pair": replicate_pair,
                        **distance_controlled_correlations(
                            left, right, minimum_offset=minimum_offset,
                        ),
                    })

    table = pd.DataFrame(rows)
    table.to_csv(output / "candidate_resolution_pairwise.csv", index=False)
    metric_names = ("full_log_correlation", "offdiagonal_log_correlation",
                    "distance_centered_correlation")
    summaries = []
    for (resolution, candidate_id, contrast), group in table.groupby(
            ["resolution_bp", "candidate_region_id", "condition_contrast"], sort=False):
        record = {"resolution_bp": int(resolution), "candidate_region_id": candidate_id,
                  "condition_contrast": contrast, "num_pairs": len(group)}
        for metric in metric_names:
            values = group[metric].dropna().to_numpy(float)
            record[f"{metric}_median"] = float(np.median(values)) if len(values) else None
            record[f"{metric}_q25"] = float(np.quantile(values, .25)) if len(values) else None
            record[f"{metric}_q75"] = float(np.quantile(values, .75)) if len(values) else None
        summaries.append(record)
    pd.DataFrame(summaries).to_csv(output / "candidate_resolution_summary.csv", index=False)
    summary = {
        "protocol": "fixed task2-main-v1 candidate coordinates; local sum-pooling of source maps",
        "source_resolution_bp": source_resolution,
        "requested_resolutions_bp": list(resolutions),
        "distance_threshold_bp": distance_threshold_bp,
        "minimum_offsets_by_resolution": {
            str(resolution): int(np.ceil(distance_threshold_bp / resolution))
            for resolution in resolutions
        },
        "conditions": list(conditions), "replicates_per_condition": 2,
        "num_candidate_regions": len(candidates),
        "candidate_keys": candidates.stable_candidate_key.astype(str).tolist(),
        "limitations": [
            "Same source experiment is pooled at each resolution; bins are not independent experiments",
            "Candidate intervals were selected using WT and retain selection bias",
            "Distance thresholds are rounded up to whole bins at each resolution",
            "Five candidates and two replicates per condition support descriptive checks only",
            "No differential testing, causal inference, or new-structure claim",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
