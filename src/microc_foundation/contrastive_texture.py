"""Train-fold-only replicate-paired texture representations for exploratory Task 2."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .structure_fidelity import distance_residual


def representation_diagnostics(features: np.ndarray) -> dict[str, object]:
    """Finite paired-feature summaries without platform-dependent BLAS matmul."""
    h = np.asarray(features, dtype=float)
    if (h.ndim != 3 or h.shape[1] != 2 or h.shape[2] < 1 or len(h) < 2
            or not np.isfinite(h).all()):
        raise ValueError("need at least two finite paired embeddings")
    normalized = h / np.maximum(np.linalg.norm(h, axis=2, keepdims=True), 1e-12)
    similar = np.einsum("ik,jk->ij", normalized[:, 0], normalized[:, 1], optimize=False)
    offdiag = ~np.eye(len(h), dtype=bool)
    return {"windows": len(h), "mean_embedding_std": float(h.mean(axis=1).std(axis=0).mean()),
            "mean_positive_cosine": float(np.diag(similar).mean()),
            "mean_nonmatching_cosine": float(similar[offdiag].mean()),
            "rep1_to_rep2_top1_retrieval": float((similar.argmax(axis=1) == np.arange(len(h))).mean()),
            "chance_retrieval": 1 / len(h)}


def texture_input(counts: np.ndarray) -> np.ndarray:
    """128-bin counts -> clipped unit-RMS distance residual -> 64-bin average."""
    x = np.asarray(counts, dtype=float)
    if x.shape != (128, 128) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("requires finite nonnegative 128x128 counts")
    residual = distance_residual(np.log1p(x), minimum_offset=5)
    row, column = np.triu_indices(128, k=5)
    keep = column - row <= 125
    rms = float(np.sqrt(np.mean(residual[row[keep], column[keep]]**2)))
    normalized = np.clip(residual / rms, -5, 5) if rms > 1e-12 else np.zeros_like(residual)
    return normalized.reshape(64, 2, 64, 2).mean(axis=(1, 3)).astype("float32")


class TextureEncoder(nn.Module):
    """128-dimensional spatial embedding; 32-dimensional contrastive projection."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=2), nn.GroupNorm(4, 8), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.projection = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))

    def encode(self, values):
        return self.features(values).flatten(1)

    def forward(self, values):
        return self.projection(self.encode(values))


def paired_contrastive_loss(first: torch.Tensor, second: torch.Tensor,
                            temperature: float = .2) -> torch.Tensor:
    """Mean NT-Xent over two-view anchors; self pairs masked, exact replicate pair positive."""
    if (first.ndim != 2 or first.shape != second.shape or len(first) < 2
            or temperature <= 0 or not torch.isfinite(first).all()
            or not torch.isfinite(second).all()):
        raise ValueError("need at least two finite matching pairs and positive temperature")
    count = len(first)
    joined = F.normalize(torch.cat((first, second)), dim=1)
    scores = joined @ joined.T / temperature
    scores = scores.masked_fill(torch.eye(2*count, dtype=torch.bool, device=scores.device), -torch.inf)
    positives = (torch.arange(2*count, device=scores.device) + count) % (2*count)
    return F.cross_entropy(scores, positives)


def train_pair_encoder(training_maps: np.ndarray, *, seed: int,
                       epochs: int = 20, batch_size: int = 32):
    """Only supplied training windows are accessible here; labels are not an argument."""
    x = np.asarray(training_maps, dtype="float32")
    if (x.ndim != 4 or x.shape[1:] != (2, 64, 64) or len(x) < 2
            or not np.isfinite(x).all() or epochs < 1 or batch_size < 2):
        raise ValueError("invalid training maps or optimizer configuration")
    torch.manual_seed(seed)
    model = TextureEncoder()
    initial = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(seed)
    history = []
    model.train()
    # Equal-sized chunks keep all windows and avoid a singleton last batch.
    batches = min(int(np.ceil(len(x) / batch_size)), len(x) // 2)
    for epoch in range(epochs):
        loss_sum = 0.
        for indices in np.array_split(rng.permutation(len(x)), batches):
            first = torch.from_numpy(x[indices, 0, None])
            second = torch.from_numpy(x[indices, 1, None])
            optimizer.zero_grad()
            projected = model(torch.cat((first, second)))
            loss = paired_contrastive_loss(projected[:len(indices)], projected[len(indices):])
            if not torch.isfinite(loss):
                raise RuntimeError("contrastive training diverged")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(indices)
        history.append({"epoch": epoch + 1, "train_contrastive_loss": loss_sum / len(x),
                        "training_windows": len(x)})
    return model.eval(), initial.eval(), history


def encode_replicates(model: TextureEncoder, maps: np.ndarray) -> np.ndarray:
    x = np.asarray(maps, dtype="float32")
    if x.ndim != 4 or x.shape[1:] != (2, 64, 64) or not np.isfinite(x).all():
        raise ValueError("invalid evaluation texture maps")
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(x), 32):
            chunk = x[start:start+32].reshape(-1, 1, 64, 64)
            result.append(model.encode(torch.from_numpy(chunk)).numpy().reshape(-1, 2, 128))
    return np.concatenate(result)
