"""Boundary-conditioned known-structure recognition, not annotation-free detection."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .metrics import classification_metrics


def region_grid(window_start, window_end, annotation_start, annotation_end, *, size=64):
    """Class-blind 1.5x annotation scope, clipped to the original extraction window.

Sampling at pixel centres with align_corners=False avoids integer ROI rounding.
This is bilinear ROI resampling, not the complete Mask R-CNN RoIAlign algorithm.
"""
    if (not window_start <= annotation_start < annotation_end <= window_end or
            size < 8 or window_end <= window_start):
        raise ValueError("annotation must lie within the extraction window")
    length = annotation_end-annotation_start
    left = max(window_start, annotation_start-.25*length)
    right = min(window_end, annotation_end+.25*length)
    centers = left+(np.arange(size)+.5)*(right-left)/size
    coordinates = 2*(centers-window_start)/(window_end-window_start)-1
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    inside = (centers >= annotation_start) & (centers < annotation_end)
    mask = inside[:, None] & inside[None, :]
    return np.stack((xx, yy), axis=-1).astype("float32"), mask, (left, right)


def roi_shape_input(log_maps, grid):
    """Remove native log-distance means, resample, then unit-RMS ROI contrast.

No label, coordinate ID, annotation length scalar, or fitted test statistics
enter the classifier. ROI resampling can still expose annotation scale, hence
the mandatory length/density controls and conditional-use limitation.
"""
    x = np.asarray(log_maps, dtype=float)
    g = np.asarray(grid, dtype="float32")
    if (x.shape != (2, 256, 256) or not np.isfinite(x).all() or (x < 0).any() or
            g.shape != (64, 64, 2) or not np.isfinite(g).all() or (np.abs(g) > 1).any()):
        raise ValueError("need two finite nonnegative 256-bin maps and a 64-bin grid")
    residual = np.zeros_like(x)
    for offset in range(2, 254):
        rows = np.arange(256-offset)
        values = x[:, rows, rows+offset]
        centered = values-values.mean(axis=1, keepdims=True)
        residual[:, rows, rows+offset] = centered
        residual[:, rows+offset, rows] = centered
    sampled = F.grid_sample(torch.from_numpy(residual.astype("float32"))[None],
        torch.from_numpy(g)[None], mode="bilinear", padding_mode="border",
        align_corners=False)[0].numpy()
    # Bilinear accumulation order can introduce sub-ulp asymmetry.
    sampled = (sampled+sampled.transpose(0, 2, 1))/2
    row, col = np.triu_indices(64, k=2)
    rms = np.sqrt((sampled[:, row, col]**2).mean(axis=1))
    sampled /= np.where(rms > 1e-6, rms, 1)[:, None, None]
    return np.clip(sampled, -5, 5).astype("float32")


def density_preserving_shuffle(maps, mask, *, seed):
    """Shuffle transformed values within each offset and selected region.

Preserves each repeat's offset histogram (including ties and zeros) exactly;
does not claim to simulate a biological sample or preserve native read counts.
"""
    x = np.asarray(maps, dtype="float32")
    mask = np.asarray(mask, dtype=bool)
    if (x.shape != (2, 64, 64) or mask.shape != (64, 64) or
            not np.isfinite(x).all() or not np.array_equal(mask, mask.T)):
        raise ValueError("need finite paired symmetric-region maps")
    rng = np.random.default_rng(seed)
    output = x.copy()
    for offset in range(2, 64):
        rows = np.arange(64-offset)
        rows = rows[mask[rows, rows+offset]]
        columns = rows+offset
        order = rng.permutation(len(rows))
        output[:, rows, columns] = x[:, rows[order], columns[order]]
        output[:, columns, rows] = output[:, rows, columns]
    return output


class RegionShapeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, 64), nn.SiLU(),
                                        nn.Dropout(.15), nn.Linear(64, 3))

    def forward(self, x):
        return self.classifier(self.features(x))


def paired_logits(model, maps):
    x = np.asarray(maps, dtype="float32")
    if (x.ndim != 4 or x.shape[1:] != (2, 64, 64) or not len(x) or
            not np.isfinite(x).all()):
        raise ValueError("invalid ROI maps")
    model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(x), 32):
            batch = torch.from_numpy(x[start:start+32].reshape(-1, 1, 64, 64))
            results.append(model(batch).numpy().reshape(-1, 2, 3))
    return np.concatenate(results)


def background_gradient_ratio(logits, values, labels, masks):
    """Train-only target-log-probability gradient penalty outside annotated ROI.

Unlike the published unnormalised gradient penalty this uses a scale-normalised
ratio. Held-out masks are NOT supplied to the model or applied to explanations.
"""
    selected = F.log_softmax(logits, dim=1).gather(1, labels[:, None]).sum()
    gradients = torch.autograd.grad(selected, values, create_graph=True)[0]
    energy = gradients.square()
    outside = (~masks).to(values.dtype)[:, None]
    return ((energy*outside).sum((1, 2, 3)) /
            energy.sum((1, 2, 3)).clamp_min(1e-12)).mean()


def train_roi_classifier(train_maps, train_labels, train_masks, val_maps, val_labels,
                         *, seed, gradient_penalty=0., epochs=100, patience=20):
    """Test data is absent from this API; choose on validation worst-class recall/F1."""
    train_maps = np.asarray(train_maps, dtype="float32")
    train_labels = np.asarray(train_labels, dtype=int)
    train_masks = np.asarray(train_masks, dtype=bool)
    val_labels = np.asarray(val_labels, dtype=int)
    if (train_maps.shape != (len(train_labels), 2, 64, 64) or
            train_masks.shape != (len(train_labels), 64, 64) or
            set(train_labels) != {0, 1, 2} or set(val_labels) != {0, 1, 2} or
            gradient_penalty < 0 or epochs < 1 or patience < 1 or
            not np.isfinite(train_maps).all()):
        raise ValueError("invalid maps, labels, masks or training configuration")
    torch.manual_seed(seed)
    model = RegionShapeCNN()
    initial = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                          eta_min=.00001)
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(train_labels == i) for i in range(3)]
    history, best_key, best_state, best_epoch, stale = [], (-1., -1.), None, 0, 0
    for epoch in range(epochs):
        model.train()
        ce_sum, penalty_sum = 0., 0.
        steps = int(np.ceil(len(train_labels)/36))
        for _ in range(steps):
            indices = np.concatenate([rng.choice(pool, 12, replace=True) for pool in pools])
            rng.shuffle(indices)
            # Random repeat view and simultaneous genomic reversal, not fake samples.
            reps = rng.integers(0, 2, len(indices))
            values = train_maps[indices, reps].copy()
            masks = train_masks[indices].copy()
            reverse = rng.random(len(indices)) < .5
            values[reverse] = values[reverse, ::-1, ::-1]
            masks[reverse] = masks[reverse, ::-1, ::-1]
            inputs = torch.from_numpy(values[:, None]).requires_grad_(gradient_penalty > 0)
            target = torch.from_numpy(train_labels[indices]).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            ce = F.cross_entropy(logits, target)
            penalty = (background_gradient_ratio(logits, inputs, target, torch.from_numpy(masks))
                       if gradient_penalty else torch.zeros(()))
            loss = ce+gradient_penalty*penalty
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite ROI training loss")
            loss.backward()
            optimizer.step()
            ce_sum += float(ce.item())
            penalty_sum += float(penalty.item())
        scheduler.step()
        logits = paired_logits(model, val_maps).mean(axis=1)
        metrics = classification_metrics(val_labels, logits.argmax(axis=1), ["CHID", "CHIN", "OPCID"])
        minimum = min(v["recall"] for v in metrics["per_class"].values())
        key = (minimum, metrics["macro_f1"])
        history.append({"epoch": epoch+1, "train_ce": ce_sum/steps,
            "train_background_gradient_ratio": penalty_sum/steps,
            "validation_minimum_recall": minimum, "validation_macro_f1": metrics["macro_f1"],
            "validation_accuracy": metrics["accuracy"]})
        if key > best_key:
            best_key, best_state, best_epoch, stale = key, copy.deepcopy(model.state_dict()), epoch+1, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model.eval(), initial, history, {"best_epoch": best_epoch,
        "selection_key": list(best_key), "gradient_penalty": gradient_penalty, "seed": seed}


def input_gradient_map(model, maps, label):
    """Actual-class, unmasked gradient times input, averaged across repeat views."""
    model.eval()
    values = torch.from_numpy(np.asarray(maps, dtype="float32")[:, None].copy()).requires_grad_(True)
    logits = model(values)
    gradient = torch.autograd.grad(logits[:, label].mean(), values)[0]
    attribution = (gradient*values).abs().mean(dim=(0, 1)).detach().numpy()
    return (attribution+attribution.T)/2
