"""Image quality metrics.

Standard protocol (used by BasicVSR, BasicVSR++, IconVSR, MIA-VSR):
    outputs are clamped to [0, 1] and quantised to uint8, PSNR / SSIM are computed per
    frame on 0-255 images (RGB for REDS4, optionally Y of BT.601 YCbCr), averaged over the
    frames of a clip and then over clips. SSIM uses an 11x11 Gaussian window (sigma 1.5)
    with 'valid' filtering, matching BasicSR's ``calculate_ssim``.

Legacy protocol (``legacy_clip_metrics``): what the original Fork-Net training code
reported, kept for backward compatibility. See docs/FINDINGS.md, F6.
"""

import numpy as np
import torch
import torch.nn.functional as F


def rgb_to_y(img):
    """(H, W, 3) RGB array in [0, 255] -> Y channel of BT.601 YCbCr in [16, 235] (MATLAB rgb2ycbcr)."""
    img = img.astype(np.float64)
    return 16.0 + (img @ np.array([65.481, 128.553, 24.966])) / 255.0


def _prepare(img, crop_border, y_channel):
    img = img.astype(np.float64)
    if y_channel:
        img = rgb_to_y(img)[..., None]
    if crop_border:
        img = img[crop_border:-crop_border, crop_border:-crop_border]
    return img


def psnr(img1, img2, crop_border=0, y_channel=False):
    """PSNR between two (H, W, 3) uint8 images."""
    assert img1.shape == img2.shape, (img1.shape, img2.shape)
    a, b = _prepare(img1, crop_border, y_channel), _prepare(img2, crop_border, y_channel)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else float(10.0 * np.log10(255.0**2 / mse))


def _gaussian_window(size=11, sigma=1.5):
    x = np.arange(size, dtype=np.float64) - size // 2
    g = np.exp(-(x**2) / (2 * sigma**2))
    g /= g.sum()
    return torch.from_numpy(np.outer(g, g))[None, None]


_WINDOW = _gaussian_window()
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ssim(img1, img2, crop_border=0, y_channel=False):
    """SSIM between two (H, W, 3) uint8 images (mean over channels). Float64, on GPU when available."""
    assert img1.shape == img2.shape, (img1.shape, img2.shape)
    a, b = _prepare(img1, crop_border, y_channel), _prepare(img2, crop_border, y_channel)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    # channels as a batch of single-channel images: (C, 1, H, W)
    a = torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[:, None].to(_DEVICE)
    b = torch.from_numpy(np.ascontiguousarray(b.transpose(2, 0, 1)))[:, None].to(_DEVICE)
    window = _WINDOW.to(_DEVICE)
    filt = lambda x: F.conv2d(x, window)  # noqa: E731 - 'valid' filtering
    mu1, mu2 = filt(a), filt(b)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2
    s1, s2, s12 = filt(a * a) - mu1_sq, filt(b * b) - mu2_sq, filt(a * b) - mu12
    ssim_map = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return float(ssim_map.mean(dim=(1, 2, 3)).mean())


def frame_metrics(sr, gt, crop_border=0):
    """All standard metrics for one frame pair of (H, W, 3) uint8 images."""
    return {
        "psnr_rgb": psnr(sr, gt, crop_border),
        "ssim_rgb": ssim(sr, gt, crop_border),
        "psnr_y": psnr(sr, gt, crop_border, y_channel=True),
        "ssim_y": ssim(sr, gt, crop_border, y_channel=True),
    }


def legacy_clip_metrics(sr, hr):
    """Metrics exactly as computed by the original training code, for one batch of clips.

    Differences from the standard protocol: no clamping or quantisation; PSNR is one value
    over the whole (N, T, C, H, W) tensor with ``data_range = hr.max() - hr.min()``
    (the torchmetrics < 1.0 default when ``data_range`` was unset; implemented directly since
    newer torchmetrics require it); SSIM is torchmetrics' SSIM on float RGB averaged over
    time steps. Requires ``torchmetrics``.
    """
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    sr, hr = sr.double(), hr.double()
    data_range = hr.max() - hr.min()
    psnr_val = 10 * torch.log10(data_range**2 / torch.mean((sr - hr) ** 2))
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(sr.device)
    ssim_val = torch.stack([ssim_fn(sr[:, t], hr[:, t]) for t in range(sr.size(1))]).mean()
    return {"psnr": float(psnr_val), "ssim": float(ssim_val)}
