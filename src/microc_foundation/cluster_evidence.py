"""Export and cross-check frozen clustering memberships, not new cluster discovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from microc_foundation.clustering import build_model_features, robust_scale


def build_cluster_evidence(
    windows: pd.DataFrame, candidates: pd.DataFrame, summary: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Reconcile existing labels/centers/support IDs without refitting anything."""
    k, dimensions = int(summary["selected_num_clusters"]), int(summary["pca_components"])
    pcs = [f"pc{i + 1}" for i in range(dimensions)]
    required = {"window_id", "chrom", "start", "end", "center", "quality_pass", "cluster",
                "known_overlap_bp", "known_structure_types", "candidate_pass", "novelty_score", *pcs}
    if (not required.issubset(windows.columns) or windows.empty or
            windows.window_id.duplicated().any() or len(windows) != summary["num_quality_windows"]):
        raise ValueError("invalid or incomplete frozen window table")
    if (not windows.quality_pass.isin([True]).all() or
            not windows.candidate_pass.isin([True, False]).all() or
            set(windows.cluster) != set(range(k))):
        raise ValueError("invalid QC, selection flags or cluster labels")
    if ((windows.start < 0).any() or (windows.end <= windows.start).any() or
            (windows.known_overlap_bp < 0).any() or
            (windows.known_overlap_bp > windows.end - windows.start).any()):
        raise ValueError("invalid window/overlap intervals")
    required_regions = {"candidate_region_id", "chrom", "start", "end", "cluster",
                        "representative_window_id", "num_supporting_windows", "supporting_window_ids"}
    if (not required_regions.issubset(candidates.columns) or candidates.empty or
            candidates.candidate_region_id.duplicated().any() or
            len(candidates) != summary["num_candidate_regions"]):
        raise ValueError("invalid frozen candidate regions")
    raw, names = build_model_features(windows)
    scaled, median, scale = robust_scale(raw)
    if (names != summary["feature_names"] or
            not np.allclose(median, summary["feature_median"], rtol=1e-7, atol=1e-7) or
            not np.allclose(scale, summary["feature_scale"], rtol=1e-7, atol=1e-7)):
        raise ValueError("feature scaling disagrees with frozen summary")
    embedding = windows[pcs].to_numpy(float)
    saved_centers = np.asarray(summary["cluster_centers"], dtype=float)
    if (saved_centers.shape != (k, dimensions) or not np.isfinite(embedding).all() or
            not np.isfinite(saved_centers).all()):
        raise ValueError("invalid PCA coordinates or saved centers")
    labels = windows.cluster.to_numpy(int)
    empirical = np.stack([embedding[labels == label].mean(axis=0) for label in range(k)])
    if not np.allclose(empirical, saved_centers, rtol=1e-7, atol=1e-7):
        raise ValueError("member centroid disagrees with saved cluster center")
    distances = np.linalg.norm(embedding[:, None, :] - saved_centers[None, :, :], axis=2)
    own = distances[np.arange(len(windows)), labels]
    if not np.allclose(own, distances.min(axis=1), rtol=1e-7, atol=1e-7):
        raise ValueError("stored assignment is not nearest to its saved center")
    members = windows[["window_id", "chrom", "start", "end", "center", "cluster",
                       "known_overlap_bp", "known_structure_types", "novelty_score", "candidate_pass", *pcs]].copy()
    members["distance_to_centroid"] = own
    members["candidate_region_id"] = ""
    members["selection_stage"] = np.where(windows.candidate_pass, "selected_unmerged", "not_selected")
    region_records = []
    indexed = members.set_index("window_id")
    for region in candidates.itertuples(index=False):
        ids = str(region.supporting_window_ids).split(";")
        if (len(set(ids)) != len(ids) or len(ids) != region.num_supporting_windows or
                not set(ids).issubset(indexed.index) or region.representative_window_id not in ids):
            raise ValueError("candidate support IDs/count/representative do not match")
        support = indexed.loc[ids]
        if (not support.candidate_pass.all() or (support.known_overlap_bp > 0).any() or
                set(support.chrom) != {region.chrom} or int(support.start.min()) != region.start or
                int(support.end.max()) != region.end or
                int(support.loc[region.representative_window_id, "cluster"]) != region.cluster):
            raise ValueError("candidate support coordinates/labels disagree with region")
        mask = members.window_id.isin(ids)
        if (members.loc[mask, "candidate_region_id"] != "").any():
            raise ValueError("a supporting window cannot belong to multiple final regions")
        members.loc[mask, "candidate_region_id"] = region.candidate_region_id
        members.loc[mask, "selection_stage"] = "final_region_support"
        for label, group in support.groupby("cluster", observed=True):
            region_records.append({
                "candidate_region_id": region.candidate_region_id,
                "chrom": region.chrom, "start": int(region.start), "end": int(region.end),
                "stable_candidate_key": f"task2-main-v1:{region.chrom}:{region.start}-{region.end}",
                "representative_window_id": region.representative_window_id,
                "representative_cluster": int(region.cluster), "support_cluster": int(label),
                "support_count": len(group), "supporting_window_ids": ";".join(group.index),
            })
    centers, profiles = [], []
    for label in range(k):
        selected = labels == label
        group = members.loc[selected]
        known_n = int((group.known_overlap_bp > 0).sum())
        if (len(group) != summary["cluster_counts"][str(label)] or
                not np.isclose(known_n / len(group), summary["cluster_known_overlap_fractions"][str(label)])):
            raise ValueError("cluster counts/known composition disagree with summary")
        nearest = group.sort_values(["distance_to_centroid", "window_id"]).iloc[0]
        centers.append({
            "cluster": label, "members": len(group), "known_overlap_members": known_n,
            "unannotated_members": len(group) - known_n, "known_overlap_fraction": known_n / len(group),
            "selected_windows": int(group.candidate_pass.sum()),
            "final_region_support_windows": int((group.selection_stage == "final_region_support").sum()),
            "nearest_window_id": nearest.window_id, "nearest_chrom": nearest.chrom,
            "nearest_start": int(nearest.start), "nearest_end": int(nearest.end),
            "nearest_known_structure_types": nearest.known_structure_types,
            **{name: float(saved_centers[label, index]) for index, name in enumerate(pcs)},
        })
        for index, name in enumerate(names):
            profiles.append({
                "cluster": label, "feature": name, "member_n": len(group),
                "feature_mean": float(raw[selected, index].mean()),
                "feature_median": float(np.median(raw[selected, index])),
                "feature_q25": float(np.percentile(raw[selected, index], 25)),
                "feature_q75": float(np.percentile(raw[selected, index], 75)),
                "scaled_mean": float(scaled[selected, index].mean()),
                "scaled_median": float(np.median(scaled[selected, index])),
            })
    selected_n = int(members.candidate_pass.sum())
    if selected_n != summary["num_candidate_windows_before_merge"]:
        raise ValueError("selected window count disagrees with frozen summary")
    final_n = int((members.selection_stage == "final_region_support").sum())
    return members, pd.DataFrame(profiles), pd.DataFrame(centers), pd.DataFrame(region_records), {
        "protocol": "task2-main-v1-cluster-evidence-v1", "refitted": False,
        "quality_windows": len(members), "clusters": k, "selected_windows": selected_n,
        "final_region_support_windows": final_n, "selected_unmerged_windows": selected_n - final_n,
        "candidate_regions": len(candidates),
        "max_centroid_absolute_error": float(np.max(np.abs(empirical - saved_centers))),
        "all_clusters_contain_known": bool(all(row["known_overlap_members"] > 0 for row in centers)),
        "limitations": ["whole-genome WT fit; not independent cluster validation",
                        "overlapping windows are not independent structures",
                        "nearest actual window is an example, not a mean contact-matrix shape",
                        "candidate regions are not novel biological classes"],
    }
