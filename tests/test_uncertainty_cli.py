import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microc_foundation.super_resolution import local_ssim, psnr


def fixture(tmp_path):
    root = tmp_path / "runs"
    target = np.random.default_rng(6).uniform(0, 1, size=(2, 2, 256, 256)).astype("float32")
    for seed in (2026, 2027, 2028):
        directory = root / f"seed_{seed}"
        directory.mkdir(parents=True)
        arrays = {"target": target, "bicubic": target + .2, "cnn": target + .1}
        np.savez(directory / "test_reconstruction.npz", **arrays, structure_id=["a", "b"],
                 label=["CHIN", "CHID"], chrom=["x", "x"], window_start=[0, 200000],
                 window_end=[20480, 220480], annotation_start=[0, 200000], annotation_end=[20000, 220000],
                 replicate=["rep1", "rep2"])
        records = []
        for i, sid in enumerate(("a", "b")):
            for rep in range(2):
                for method in ("bicubic", "cnn"):
                    t, p = target[i, rep], arrays[method][i, rep]
                    records.append({"structure_id": sid, "replicate_index": rep + 1, "method": method,
                                    "psnr": psnr(t, p), "ssim": local_ssim(t, p),
                                    "mse": float(np.mean((t-p)**2))})
        pd.DataFrame(records).to_csv(directory / "per_sample_metrics.csv", index=False)
    return root


def run(root, out, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["audit", "--reconstruction-dir", str(root), "--output-dir", str(out), "--draws", "100"])
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_super_resolution_uncertainty.py"), run_name="__main__")


def test_cli_fixture_outputs_and_never_overwrites(tmp_path, monkeypatch):
    root, out = fixture(tmp_path), tmp_path / "out"
    with pytest.raises(SystemExit) as result:
        run(root, out, monkeypatch)
    assert result.value.code == 0
    assert len(pd.read_csv(out / "per_channel_metrics.csv")) == 24
    paired = pd.read_csv(out / "paired_structure_metrics.csv")
    assert paired.cnn_residual_valid_observations.tolist() == [6, 6]
    before = (out / "summary.json").read_bytes()
    with pytest.raises(SystemExit) as result:
        run(root, out, monkeypatch)
    assert result.value.code == 2 and (out / "summary.json").read_bytes() == before


def test_cli_rejects_stale_metric_before_creating_outputs(tmp_path, monkeypatch):
    root, out = fixture(tmp_path), tmp_path / "out"
    p = root / "seed_2026/per_sample_metrics.csv"
    frame = pd.read_csv(p)
    frame.loc[0, "psnr"] += .1
    frame.to_csv(p, index=False)
    with pytest.raises(ValueError, match="psnr disagrees"):
        run(root, out, monkeypatch)
    assert not out.exists()
