"""Exploratory label controls with unchanged background and spatial boundaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from microc_foundation.clustering import annotate_known_structures, build_model_features
from microc_foundation.discovery_benchmark import (
    binary_metrics,
    fit_logistic,
    predict_logistic,
    spatial_splits,
)

RULES = ("any_overlap", "coverage_25", "coverage_50", "coverage_75", "center_inside")


def label_roles(windows: pd.DataFrame, structures: pd.DataFrame) -> pd.DataFrame:
    """Keep zero-overlap background; EXCLUDE, never relabel, weaker positives."""
    required = {"window_id", "chrom", "start", "end", "center", "fold"}
    if (not required.issubset(windows.columns) or windows.empty or
            windows.window_id.duplicated().any() or windows[list(required)].isna().any().any()):
        raise ValueError("expected nonempty unique windows with coordinates and folds")
    if ((windows.start < 0).any() or (windows.end <= windows.start).any() or
            not np.allclose(windows.center, (windows.start + windows.end) / 2, atol=0.5)):
        raise ValueError("invalid window intervals or centers")
    annotated = annotate_known_structures(windows, structures)
    center_inside = np.array([
        bool(((structures.chrom.astype(str) == str(row.chrom)) &
              (structures.start <= row.center) & (row.center < structures.end)).any())
        for row in annotated.itertuples(index=False)
    ])
    fractions = annotated.known_overlap_fraction.to_numpy(float)
    positive_masks = {
        "any_overlap": fractions > 0,
        "coverage_25": fractions >= 0.25,
        "coverage_50": fractions >= 0.50,
        "coverage_75": fractions >= 0.75,
        "center_inside": center_inside,
    }
    for rule, positive in positive_masks.items():
        annotated[rule] = np.where(positive, "positive", np.where(
            fractions == 0, "background", "excluded"
        ))
    return annotated


def audit_label_sensitivity(
    windows: pd.DataFrame, structures: pd.DataFrame, *, purge_bp: int = 20_480,
    permutations: int = 100, seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Evaluate each fixed rule or explicitly mark the whole rule unsupported.

    Splits are made ONCE from the original any-overlap grid. Removing ambiguous
    windows cannot shrink held-out envelopes or re-admit purged training windows.
    No partial-fold averages or post-hoc selection of label rules are reported.
    """
    if permutations < 0:
        raise ValueError("permutations must be nonnegative")
    annotated = label_roles(windows, structures)
    annotated["label"] = (annotated.known_overlap_bp > 0).astype(int)
    splits = list(spatial_splits(annotated, purge_bp=purge_bp))
    matrix, feature_names = build_model_features(annotated)
    feature_sets = {"density_only": [0, 1], "shape_repeat": list(range(2, 8)),
                    "all_features": list(range(8))}
    memberships, count_records, results = [], [], {}
    for rule in RULES:
        roles = annotated[rule].to_numpy()
        eligible = roles != "excluded"
        labels = (roles == "positive").astype(int)
        rule_splits, unsupported = [], []
        for fold, reference_train, reference_test in splits:
            train, test = reference_train & eligible, reference_test & eligible
            record = {"rule": rule, "fold": fold}
            for partition, reference, mask in [
                ("train", reference_train, train), ("test", reference_test, test)
            ]:
                record[f"{partition}_positive"] = int((mask & (labels == 1)).sum())
                record[f"{partition}_background"] = int((mask & (roles == "background")).sum())
                record[f"{partition}_excluded"] = int((reference & ~eligible).sum())
                if len(np.unique(labels[mask])) != 2:
                    unsupported.append(f"fold {fold} {partition} lacks both classes")
            count_records.append(record)
            rule_splits.append((fold, train, test))
        selected = annotated.drop(columns=list(RULES) + ["label"]).copy()
        selected["rule"] = rule
        selected["role"] = roles
        selected["evaluation_label"] = np.where(eligible, labels, -1)
        result = {
            "status": "unsupported" if unsupported else "evaluated",
            "reasons": unsupported, "positive": int(labels.sum()),
            "background": int((roles == "background").sum()),
            "excluded": int((~eligible).sum()), "models": {},
            "shuffle_control": None,
        }
        for name in feature_sets:
            selected[f"{name}_probability"] = np.nan
        if not unsupported:
            for name, indices in feature_sets.items():
                features = matrix[:, indices]
                scores = np.full(len(windows), np.nan)
                folds = []
                for fold, train, test in rule_splits:
                    _, fitted = fit_logistic(features[train], labels[train])
                    scores[test] = predict_logistic(features[test], fitted)
                    folds.append({
                        "fold": fold, "train_window_ids": annotated.loc[train, "window_id"].tolist(),
                        "test_window_ids": annotated.loc[test, "window_id"].tolist(),
                        "metrics": binary_metrics(labels[test], scores[test]), "fit": fitted,
                    })
                selected[f"{name}_probability"] = scores
                aucs = [fold["metrics"]["roc_auc"] for fold in folds]
                result["models"][name] = {
                    "features": [feature_names[index] for index in indices], "folds": folds,
                    "mean_fold_roc_auc": float(np.mean(aucs)),
                    "sd_fold_roc_auc": float(np.std(aucs, ddof=1)),
                    "pooled_oof_metrics": binary_metrics(labels[eligible], scores[eligible]),
                }
            rng = np.random.default_rng(seed)
            null = []
            for _ in range(permutations):
                shuffled = labels.copy()
                for _, _, test in rule_splits:
                    shuffled[test] = rng.permutation(labels[test])
                aucs = []
                for _, train, test in rule_splits:
                    _, fitted = fit_logistic(matrix[train], shuffled[train])
                    aucs.append(binary_metrics(shuffled[test], predict_logistic(
                        matrix[test], fitted
                    ))["roc_auc"])
                null.append(float(np.mean(aucs)))
            result["shuffle_control"] = {
                "seed": seed, "permutations": permutations, "mean_fold_roc_auc": null,
                "null_2_5_percentile": float(np.percentile(null, 2.5)) if null else None,
                "null_97_5_percentile": float(np.percentile(null, 97.5)) if null else None,
                "interpretation": "descriptive control, not biological significance",
            }
        memberships.append(selected)
        results[rule] = result
    return pd.concat(memberships, ignore_index=True), pd.DataFrame(count_records), {
        "protocol": "task2-label-sensitivity-v1", "reference_n": len(annotated),
        "purge_bp": purge_bp, "threshold": 0.5, "penalty": 0.01,
        "background_rule": "zero union overlap only; other weak positives excluded",
        "reference_splits": [{
            "fold": fold, "test_start": int(annotated.loc[test, "start"].min()),
            "test_end": int(annotated.loc[test, "end"].max()),
            "train_window_ids": annotated.loc[train, "window_id"].tolist(),
            "test_window_ids": annotated.loc[test, "window_id"].tolist(),
        } for fold, train, test in splits],
        "rules": results,
        "limitations": [
            "exploratory reused regions, not an untouched test set",
            "coverage rules bias toward long annotations or annotation-dense windows",
            "different rules have different sample difficulty; no best-rule claim",
            "unannotated background is not verified biological negative",
        ],
    }
