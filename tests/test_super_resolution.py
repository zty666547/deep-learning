import numpy as np
import pytest
import torch

from microc_foundation.super_resolution import (
    ContactSuperResolutionCNN,
    correct_count_scale,
    global_ssim,
    load_aligned_super_resolution_data,
    local_ssim,
    psnr,
)
from microc_foundation.super_resolution_evaluation import evaluate_reconstructions


def test_super_resolution_model_doubles_spatial_shape():
    result = ContactSuperResolutionCNN(2)(torch.zeros(3, 2, 8, 8))
    assert result.shape == (3, 2, 16, 16)


def test_image_metrics_are_perfect_for_identical_maps():
    values = np.eye(8)
    assert psnr(values, values) == float("inf")
    assert global_ssim(values, values) == pytest.approx(1.0)


def test_image_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        psnr(np.zeros((2, 2)), np.zeros((3, 3)))


def test_count_scale_corrects_sum_pooling_before_log_transform():
    coarse_counts = np.array([[4.0, 40.0], [0.0, 400.0]])
    assert np.allclose(correct_count_scale(np.log1p(coarse_counts), 2), np.log1p(coarse_counts / 4))
    assert not np.allclose(np.log1p(coarse_counts) / 4, np.log1p(coarse_counts / 4))


def test_local_ssim_matches_direct_window_moments():
    truth = np.arange(81).reshape(9, 9).astype(float)
    predicted = truth.copy()
    predicted[4, 4] = 0
    scores = []
    c1, c2 = (0.01 * 80)**2, (0.03 * 80)**2
    for row in range(3):
        for column in range(3):
            x = truth[row:row + 7, column:column + 7]
            y = predicted[row:row + 7, column:column + 7]
            covariance = ((x - x.mean()) * (y - y.mean())).mean()
            scores.append(((2*x.mean()*y.mean()+c1)*(2*covariance+c2)) /
                          ((x.mean()**2+y.mean()**2+c1)*(x.var()+y.var()+c2)))
    assert local_ssim(truth, predicted) == pytest.approx(np.mean(scores))
    assert local_ssim(truth, truth) == pytest.approx(1)


def _paired_fixture(tmp_path, *, high_start=None, high_split=None, common_starts=None):
    ids = np.array(["A", "B", "C"])
    for name, size, resolution in (("low", 8, 160), ("high", 16, 80)):
        starts = np.array([0, 2000, 4000] if common_starts is None else common_starts)
        splits = np.array(["train", "validation", "test"])
        if name == "high":
            starts = starts if high_start is None else np.asarray(high_start)
            splits = splits if high_split is None else np.asarray(high_split)
        np.savez(tmp_path / f"{name}.npz", matrices=np.ones((3, 2, size, size)),
                 structure_id=ids, chrom=np.array(["chr1"]*3), window_start=starts,
                 window_end=starts + 1280, replicate=np.array(["rep1", "rep2"]),
                 split=splits, pooled_bin_size=resolution, normalization="log1p", pooling="sum")
    return str(tmp_path / "low.npz"), str(tmp_path / "high.npz")


def test_paired_loader_excludes_if_either_resolution_excludes(tmp_path):
    low, high = _paired_fixture(tmp_path, high_split=["train", "excluded_boundary", "test"])
    result = load_aligned_super_resolution_data(low, high)
    assert result["split"].tolist() == ["train", "excluded", "test"]


def test_paired_loader_rejects_coordinate_disagreement(tmp_path):
    low, high = _paired_fixture(tmp_path, high_start=[0, 2100, 4000])
    with pytest.raises(ValueError, match="coordinates differ"):
        load_aligned_super_resolution_data(low, high)


def test_paired_loader_rejects_retained_split_disagreement(tmp_path):
    low, high = _paired_fixture(tmp_path, high_split=["test", "validation", "test"])
    with pytest.raises(ValueError, match="inconsistent retained splits"):
        load_aligned_super_resolution_data(low, high)


def test_cnn_output_obeys_contact_symmetry():
    result = ContactSuperResolutionCNN(2)(torch.randn(1, 2, 8, 8))
    assert torch.allclose(result, result.transpose(-1, -2))


def test_paired_loader_rejects_genomic_overlap_across_splits(tmp_path):
    low, high = _paired_fixture(tmp_path, common_starts=[0, 500, 4000])
    with pytest.raises(ValueError, match="overlap across data splits"):
        load_aligned_super_resolution_data(low, high)


def test_local_ssim_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        local_ssim(np.zeros((8, 8)), np.full((8, 8), np.nan))


def test_structure_and_distance_evaluation_aggregates_repeated_seeds(tmp_path):
    paths = []
    target = np.zeros((2, 2, 256, 256), dtype=np.float32)
    target[:, :, np.arange(256), np.arange(256)] = 1
    for seed in (2026, 2027):
        path = tmp_path / f"seed_{seed}" / "test_reconstruction.npz"
        path.parent.mkdir()
        np.savez_compressed(
            path,
            structure_id=np.array(["A", "B"]),
            label=np.array(["CHIN", "OPCID"]),
            chrom=np.array(["chr1", "chr1"]),
            window_start=np.array([0, 20480]),
            window_end=np.array([20480, 40960]),
            annotation_start=np.array([0, 20480]),
            annotation_end=np.array([20480, 40960]),
            replicate=np.array(["rep1", "rep2"]),
            target=target,
            bicubic=np.zeros_like(target),
            cnn=np.full_like(target, 0.5),
        )
        paths.append(str(path))

    summary = evaluate_reconstructions(paths, str(tmp_path / "evaluation"))
    assert summary["test_structure_count"] == 2
    assert len(summary["distance_summary"]) == 10
    assert len(summary["known_structure_summary"]) == 4
    assert (tmp_path / "evaluation" / "summary.json").is_file()
    assert (tmp_path / "evaluation" / "known_structure_metrics_by_sample.csv").is_file()
