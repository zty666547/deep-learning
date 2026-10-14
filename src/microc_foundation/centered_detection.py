"""Exploratory structure-centred detection with genomic purging and controls.

The background label means unannotated, not a verified biological negative.
This task is distinct from labelling any-overlap scan windows.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .contrastive_texture import TextureEncoder


def centered_registry(structures, scan, *, chrom_length=4_641_652):
    """Deduplicate grid-aligned annotation-centred windows; keep fixed background grid."""
    if (structures.empty or scan.empty or chrom_length < 20_480 or
            structures.structure_id.duplicated().any() or scan.window_id.duplicated().any()):
        raise ValueError("need nonempty, unique annotations and scan IDs")
    for frame in (structures, scan):
        if (set(frame.chrom) != {"NC_000913.3"} or (frame.start < 0).any() or
                (frame.end <= frame.start).any() or (frame.end > chrom_length).any()):
            raise ValueError("invalid chromosome or intervals")
    if not scan.quality_pass.isin([True, False]).all():
        raise ValueError("scan quality must be boolean")
    records = {}
    # Only full bins are used; a truncated chromosome-tail bin is not padded.
    last_start = (chrom_length // 160 - 128) * 160
    for row in structures.sort_values("structure_id").itertuples(index=False):
        if not row.start <= row.center < row.end:
            raise ValueError("annotation centre outside annotation")
        start = max(0, min((int(row.center) // 160 - 64) * 160, last_start))
        record = records.setdefault(start, {"sample_id": f"known_{start:07d}",
            "chrom": "NC_000913.3", "start": start, "end": start+20_480,
            "center": start+10_240, "label": 1, "structure_ids": [], "types": []})
        record["structure_ids"].append(str(row.structure_id))
        record["types"].append(str(row.structure_type))
    result = []
    for record in records.values():
        record["structure_ids"] = ";".join(record["structure_ids"])
        record["types"] = ";".join(sorted(set(record["types"])))
        result.append(record)
    grid = scan.loc[scan.quality_pass & (scan.start % 20_480 == 0) &
                    (scan.end-scan.start == 20_480)].sort_values("start")
    for row in grid.itertuples(index=False):
        if ((structures.end > row.start) & (structures.start < row.end)).any():
            continue
        result.append({"sample_id": str(row.window_id), "chrom": row.chrom,
            "start": int(row.start), "end": int(row.end), "center": int(row.center),
            "label": 0, "structure_ids": "", "types": "background"})
    frame = pd.DataFrame(result).sort_values(["start", "label"]).reset_index(drop=True)
    frame["fold"] = np.minimum(4, np.floor(frame.center/chrom_length*5).astype(int))
    if frame.duplicated(["start", "end"]).any() or frame.label.nunique() != 2:
        raise ValueError("conflicting or missing classes")
    return frame


def purged_centered_splits(registry, *, chrom_length=4_641_652, purge_bp=20_480):
    """Fixed five genomic blocks, actual-interval guard, and circular-end guard."""
    if (purge_bp < 0 or registry.empty or set(registry.fold) != set(range(5)) or
            (registry.start < 0).any() or (registry.end > chrom_length).any() or
            (registry.end <= registry.start).any() or
            not registry.label.isin([0, 1]).all()):
        raise ValueError("invalid registry or purge")
    expected = np.minimum(4, np.floor(registry.center/chrom_length*5).astype(int))
    if not np.array_equal(expected, registry.fold):
        raise ValueError("fold must match the fixed genomic block")
    for fold in range(5):
        test = registry.fold.to_numpy() == fold
        # A scan window centred in the held-out block can extend half a window
        # beyond its nominal edge, even if no centred annotation lies there.
        left = min(chrom_length*fold/5-10_240,
                   registry.loc[test, "start"].min())-purge_bp
        right = max(chrom_length*(fold+1)/5+10_240,
                    registry.loc[test, "end"].max())+purge_bp
        train = (~test) & ((registry.end.to_numpy() <= left) |
                          (registry.start.to_numpy() >= right))
        # E. coli is circular: the two coordinate ends are neighbours.
        if fold == 0:
            train &= registry.end.to_numpy() <= chrom_length-purge_bp
        if fold == 4:
            train &= registry.start.to_numpy() >= purge_bp
        for mask in (train, test):
            if registry.loc[mask, "label"].nunique() != 2:
                raise ValueError("each split needs both classes")
        yield fold, train, test


def component_weights(registry):
    """Class-balanced weights; each overlapping component has equal mass per class.

Compute this separately on the supplied training or evaluation subset. It
does not turn correlated samples into independent biological replicates.
"""
    if (registry.empty or registry.label.nunique() != 2 or
            not registry.label.isin([0, 1]).all() or
            (registry.end <= registry.start).any()):
        raise ValueError("need valid intervals and both classes")
    order = np.argsort(registry.start.to_numpy(), kind="stable")
    groups = np.empty(len(registry), dtype=int)
    group, end = -1, -1
    for index in order:
        if registry.start.iloc[index] >= end:
            group += 1
        end = max(end, int(registry.end.iloc[index]))
        groups[index] = group
    labels = registry.label.to_numpy(int)
    weights = np.zeros(len(registry), dtype=float)
    for label in (0, 1):
        members = np.unique(groups[labels == label])
        for identity in members:
            mask = (labels == label) & (groups == identity)
            weights[mask] = .5 / len(members) / mask.sum()
    return weights, groups


def component_binary_metrics(registry, scores):
    """Weighted AUC/recall/specificity; preserve raw-window metrics separately."""
    scores = np.asarray(scores, dtype=float)
    if (scores.shape != (len(registry),) or not np.isfinite(scores).all() or
            ((scores < 0) | (scores > 1)).any()):
        raise ValueError("invalid probabilities")
    weights, groups = component_weights(registry)
    labels = registry.label.to_numpy(int)
    p, n = labels == 1, labels == 0
    difference = scores[p, None] - scores[None, n]
    pair_weights = weights[p, None] * weights[None, n]
    auc = np.sum(pair_weights*((difference > 0)+.5*(difference == 0)))/pair_weights.sum()
    recall = np.sum(weights[p]*(scores[p] >= .5))/weights[p].sum()
    specificity = np.sum(weights[n]*(scores[n] < .5))/weights[n].sum()
    return {"roc_auc": float(auc), "recall": float(recall),
            "specificity": float(specificity),
            "balanced_accuracy": float((recall+specificity)/2),
            "positive_components": len(np.unique(groups[p])),
            "background_components": len(np.unique(groups[n])),
            "all_components": len(np.unique(groups))}


class CenteredDetector(nn.Module):
    """One fixed convolutional encoder and a binary supervision head."""

    def __init__(self):
        super().__init__()
        self.features = TextureEncoder().features
        self.classifier = nn.Linear(128, 1)

    def encode(self, values):
        return self.features(values).flatten(1)

    def forward(self, values):
        return self.classifier(self.encode(values)).flatten()


def train_centered_detector(maps, registry, *, seed, epochs=20):
    """Only WT1 training maps/labels; fixed final epoch, no holdout checkpoint choice."""
    x = np.asarray(maps, dtype="float32")
    if (x.shape != (len(registry), 64, 64) or not np.isfinite(x).all() or epochs < 1):
        raise ValueError("invalid training maps or epochs")
    weights, _ = component_weights(registry)
    torch.manual_seed(seed)
    model = CenteredDetector()
    initial = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(seed)
    history = []
    labels = torch.from_numpy(registry.label.to_numpy(dtype="float32"))
    weight_tensor = torch.from_numpy(weights.astype("float32"))
    all_maps = torch.from_numpy(x[:, None])
    for epoch in range(epochs):
        model.train()
        for indices in np.array_split(rng.permutation(len(x)), int(np.ceil(len(x)/32))):
            optimizer.zero_grad()
            logits = model(all_maps[indices])
            # Unbiased random-minibatch estimator of the global weighted loss.
            loss = (F.binary_cross_entropy_with_logits(logits, labels[indices], reduction="none") *
                    weight_tensor[indices]).sum() * len(x)/len(indices)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = float((F.binary_cross_entropy_with_logits(model(all_maps), labels,
                reduction="none")*weight_tensor).sum().item())
        history.append({"epoch": epoch+1, "weighted_training_bce": loss})
    return model.eval(), initial, history


def detector_outputs(model, maps):
    x = np.asarray(maps, dtype="float32")
    if x.ndim != 3 or x.shape[1:] != (64, 64) or not np.isfinite(x).all() or not len(x):
        raise ValueError("need finite nonempty maps")
    probabilities, embeddings = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 32):
            batch = torch.from_numpy(x[start:start+32, None])
            probabilities.append(torch.sigmoid(model(batch)).numpy())
            embeddings.append(model.encode(batch).numpy())
    return np.concatenate(probabilities), np.concatenate(embeddings)
