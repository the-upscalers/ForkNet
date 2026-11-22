import math

import numpy as np
import pytest
import torch

from conftest import smooth_texture
from forknet.metrics import frame_metrics, legacy_clip_metrics, psnr, rgb_to_y, ssim


@pytest.fixture
def pair():
    gt = smooth_texture(64, 80, seed=1)
    noise = np.random.default_rng(0).normal(0, 8, gt.shape)
    sr = np.clip(gt.astype(np.float64) + noise, 0, 255).round().astype(np.uint8)
    return sr, gt


def test_psnr_identical_is_inf(pair):
    assert math.isinf(psnr(pair[1], pair[1]))
    assert ssim(pair[1], pair[1]) == pytest.approx(1.0)


def test_psnr_matches_skimage(pair):
    skm = pytest.importorskip("skimage.metrics")
    assert psnr(*pair) == pytest.approx(skm.peak_signal_noise_ratio(pair[1], pair[0], data_range=255), abs=1e-10)


def test_psnr_known_value():
    a = np.zeros((4, 4, 3), np.uint8)
    b = np.full((4, 4, 3), 10, np.uint8)
    assert psnr(a, b) == pytest.approx(10 * math.log10(255**2 / 100))


def test_ssim_matches_skimage_gaussian(pair):
    """BasicSR's SSIM == skimage SSIM with Gaussian weights, sigma 1.5, population covariance."""
    skm = pytest.importorskip("skimage.metrics")
    ref = skm.structural_similarity(
        pair[1].astype(np.float64), pair[0].astype(np.float64), channel_axis=2, data_range=255,
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    )
    assert ssim(*pair) == pytest.approx(ref, abs=1e-6)


def test_rgb_to_y_range():
    assert rgb_to_y(np.zeros((1, 1, 3)))[0, 0] == pytest.approx(16.0)
    assert rgb_to_y(np.full((1, 1, 3), 255.0))[0, 0] == pytest.approx(235.0)


def test_crop_border_and_keys(pair):
    m = frame_metrics(*pair, crop_border=4)
    assert set(m) == {"psnr_rgb", "ssim_rgb", "psnr_y", "ssim_y"}
    assert m["psnr_y"] > m["psnr_rgb"]  # luma averages out independent channel noise


def test_legacy_metrics_use_target_range():
    pytest.importorskip("torchmetrics")
    hr = torch.rand(1, 3, 3, 32, 32) * 0.5 + 0.25  # dynamic range 0.5 instead of 1.0
    sr = hr + 0.01
    legacy = legacy_clip_metrics(sr, hr)["psnr"]
    true_range = hr.max() - hr.min()
    assert legacy == pytest.approx(10 * math.log10(true_range**2 / 1e-4), abs=1e-3)
    assert legacy < 10 * math.log10(1 / 1e-4)  # lower than PSNR with data_range=1
