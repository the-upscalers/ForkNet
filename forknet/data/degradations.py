"""Synthetic video degradations.

* ``degrade_clip``: one fixed, reproducible degradation chain (blur -> x4 downsampling -> noise ->
  JPEG -> H.264), used for the robustness sweep (``scripts/degradation_sweep.py``).
* ``RealVideoDegradation``: randomly sampled two-stage chains in the spirit of RealBasicVSR
  (Chan et al., CVPR 2022) / Real-ESRGAN, used to train real-world models. Parameters are sampled
  once per clip and shared by all its frames (except JPEG quality), so degradations are temporally
  consistent like real camera pipelines. This is a simplified re-implementation (no sinc filters),
  documented as such in the paper.
"""

import io
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .imresize import imresize


def _ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def gaussian_kernel(size, sigma_x, sigma_y=None, theta=0.0):
    """Normalised (an)isotropic Gaussian kernel of odd ``size``."""
    sigma_y = sigma_x if sigma_y is None else sigma_y
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    c, s = math.cos(theta), math.sin(theta)
    xr, yr = c * xx + s * yy, -s * xx + c * yy
    k = np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2))
    return k / k.sum()


def filter2d(img, kernel):
    """Convolves an (H, W, C) float image with a 2-D kernel (reflect padding)."""
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:  # ~40x faster than the torch float64 path; same correlation and border (reflect-101)
        out = cv2.filter2D(img.astype(np.float32), -1, kernel.astype(np.float32), borderType=cv2.BORDER_REFLECT_101)
        return out.reshape(img.shape).astype(np.float64)
    import torch
    import torch.nn.functional as F

    pad = kernel.shape[0] // 2
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[:, None].double()
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    k = torch.from_numpy(kernel)[None, None].double()
    return F.conv2d(x, k)[:, 0].numpy().transpose(1, 2, 0)


def jpeg(img_uint8, quality):
    buf = io.BytesIO()
    Image.fromarray(img_uint8).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def video_compress(frames_uint8, codec="libx264", crf=None, bitrate=None, fps=25):
    """Encodes and decodes a clip with ffmpeg. Frames must share an even-sized shape."""
    exe = _ffmpeg_exe()
    if exe is None:
        raise RuntimeError("Video compression needs ffmpeg (system binary or `pip install imageio-ffmpeg`)")
    h, w = frames_uint8[0].shape[:2]
    pad_h, pad_w = h % 2, w % 2
    frames = [np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge") for f in frames_uint8]
    rate = ["-crf", str(crf)] if crf is not None else ["-b:v", str(int(bitrate))]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.mp4"
        enc = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w + pad_w}x{h + pad_h}",
               "-r", str(fps), "-i", "-", "-c:v", codec, *rate, "-pix_fmt", "yuv420p", str(path)]
        subprocess.run(enc, input=b"".join(f.tobytes() for f in frames), check=True)
        dec = [exe, "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(dec, capture_output=True, check=True).stdout
    out = np.frombuffer(raw, np.uint8).reshape(-1, h + pad_h, w + pad_w, 3)[: len(frames)]
    return [f[:h, :w].copy() for f in out]


def _resize(img, size, mode):
    """Resizes a float (H, W, C) image in [0, 255] to ``size`` = (h, w)."""
    if mode == "matlab_bicubic":
        return imresize(img, size[0] / img.shape[0])[: size[0], : size[1]]
    if mode == "cv2_bicubic":  # no antialiasing when downsampling (aliased); used to build the LP dataset
        import cv2

        return cv2.resize(img.astype(np.float32), size[::-1], interpolation=cv2.INTER_CUBIC).astype(np.float64)
    resample = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "area": Image.BOX, "lanczos": Image.LANCZOS}[mode]
    chans = [np.asarray(Image.fromarray(img[..., c].astype(np.float32), mode="F").resize(size[::-1], resample)) for c in range(3)]
    return np.stack(chans, -1).astype(np.float64)


def degrade_clip(hr_frames, scale=4, blur_sigma=0.0, downsample="matlab_bicubic", noise_sigma=0.0, jpeg_quality=None,
                 crf=None, seed=0):
    """Fixed degradation chain for evaluation. ``hr_frames``: list of (H, W, 3) uint8.

    blur (isotropic Gaussian, sigma in HR pixels) -> x``scale`` downsampling -> additive Gaussian
    noise (sigma in 0-255 units, fixed seed) -> JPEG -> H.264 (libx264, constant ``crf``).
    With all options at their defaults this is the standard bicubic (BI) degradation.
    """
    rng = np.random.default_rng(seed)
    h, w = hr_frames[0].shape[:2]
    lr_size = (h // scale, w // scale)
    kernel = gaussian_kernel(2 * math.ceil(3 * blur_sigma) + 1, blur_sigma) if blur_sigma > 0 else None
    out = []
    for f in hr_frames:
        x = f.astype(np.float64)
        if kernel is not None:
            x = filter2d(x, kernel)
        x = _resize(x, lr_size, downsample)
        if noise_sigma > 0:
            x = x + rng.normal(0, noise_sigma, x.shape)
        x = np.clip(np.round(x), 0, 255).astype(np.uint8)
        if jpeg_quality is not None:
            x = jpeg(x, jpeg_quality)
        out.append(x)
    if crf is not None:
        out = video_compress(out, crf=crf)
    return out


class RealVideoDegradation:
    """Random two-stage degradation (RealBasicVSR-style) mapping HR clips to LR clips.

    ``cfg`` keys (all optional, defaults follow RealBasicVSR's REDS configuration):
        blur_prob, blur_sigma, blur_kernel_size, resize_range, noise_prob, noise_range, gray_noise_prob,
        jpeg_range, video_prob, video_codecs, video_bitrate, second_order_prob
    """

    DEFAULTS = {
        "blur_prob": [1.0, 0.8],
        "blur_sigma": [[0.2, 3.0], [0.2, 1.5]],
        "blur_kernel_size": [7, 21],
        "resize_range": [[0.15, 1.5], [0.3, 1.2]],
        "noise_prob": [1.0, 1.0],
        "noise_range": [[1, 30], [1, 25]],
        "gray_noise_prob": 0.4,
        "jpeg_range": [[30, 95], [35, 95]],
        "video_prob": 1.0,
        "video_codecs": ["libx264", "mpeg4"],
        "video_bitrate": [1e4, 1e5],
        "second_order_prob": 1.0,
    }

    def __init__(self, cfg=None, scale=4):
        self.cfg = {**self.DEFAULTS, **(cfg or {})}
        self.scale = scale
        self.has_ffmpeg = _ffmpeg_exe() is not None

    def _stage(self, frames, rng, i):
        c = self.cfg
        if rng.random() < c["blur_prob"][i]:
            size = int(rng.integers(c["blur_kernel_size"][0] // 2, c["blur_kernel_size"][1] // 2 + 1)) * 2 + 1
            sx, sy = rng.uniform(*c["blur_sigma"][i], size=2)
            k = gaussian_kernel(size, sx, sy, rng.uniform(0, math.pi))
            frames = [filter2d(f, k) for f in frames]
        s = rng.uniform(*c["resize_range"][i])
        h, w = frames[0].shape[:2]
        mode = rng.choice(["area", "bilinear", "bicubic"])
        new = (max(8, int(round(h * s))), max(8, int(round(w * s))))
        frames = [_resize(f, new, mode) for f in frames]
        if rng.random() < c["noise_prob"][i]:
            sigma = rng.uniform(*c["noise_range"][i])
            gray = rng.random() < c["gray_noise_prob"]
            frames = [f + (rng.normal(0, sigma, f.shape[:2])[..., None] if gray else rng.normal(0, sigma, f.shape)) for f in frames]
        lo, hi = c["jpeg_range"][i]
        return [jpeg(np.clip(np.round(f), 0, 255).astype(np.uint8), rng.uniform(lo, hi)).astype(np.float64) for f in frames]

    def __call__(self, hr_frames, rng=None):
        """hr_frames: list of (H, W, 3) uint8 with H, W divisible by ``scale``. Returns uint8 LR frames."""
        rng = rng or np.random.default_rng()
        h, w = hr_frames[0].shape[:2]
        frames = [f.astype(np.float64) for f in hr_frames]
        frames = self._stage(frames, rng, 0)
        if rng.random() < self.cfg["second_order_prob"]:
            frames = self._stage(frames, rng, 1)
        mode = rng.choice(["area", "bilinear", "bicubic"])
        lr = [np.clip(np.round(_resize(f, (h // self.scale, w // self.scale), mode)), 0, 255).astype(np.uint8) for f in frames]
        if self.has_ffmpeg and rng.random() < self.cfg["video_prob"]:
            codec = str(rng.choice(self.cfg["video_codecs"]))
            lr = video_compress(lr, codec=codec, bitrate=rng.uniform(*self.cfg["video_bitrate"]))
        return lr


class KernelMixDegradation:
    """Bicubic-family degradation with the downsampling kernel sampled per clip (pilot P1, research/12).

    Each clip is downscaled x``scale`` with one kernel drawn from ``kernels`` (weights ``kernel_probs``,
    uniform by default), then H.264-compressed with probability ``video_prob`` at a CRF drawn from
    ``crf_range``. Unlike ``RealVideoDegradation`` there is no blur or noise: it isolates the kernel shift
    (MATLAB bicubic vs aliased OpenCV cubic etc.) measured in the robustness sweep.
    """

    DEFAULTS = {
        "kernels": ["matlab_bicubic", "cv2_bicubic", "area", "bilinear", "lanczos"],
        "kernel_probs": None,
        "video_prob": 0.3,
        "crf_range": [18, 30],
    }

    def __init__(self, cfg=None, scale=4):
        self.cfg = {**self.DEFAULTS, **(cfg or {})}
        self.scale = scale
        self.has_ffmpeg = _ffmpeg_exe() is not None

    def __call__(self, hr_frames, rng=None):
        rng = rng or np.random.default_rng()
        c = self.cfg
        kernel = str(rng.choice(c["kernels"], p=c["kernel_probs"]))
        h, w = hr_frames[0].shape[:2]
        size = (h // self.scale, w // self.scale)
        lr = [np.clip(np.round(_resize(f.astype(np.float64), size, kernel)), 0, 255).astype(np.uint8) for f in hr_frames]
        if self.has_ffmpeg and rng.random() < c["video_prob"]:
            lr = video_compress(lr, crf=int(rng.integers(c["crf_range"][0], c["crf_range"][1] + 1)))
        return lr


class MixedDegradation:
    """Per clip, the RealBasicVSR-style random pipeline with probability ``real_prob``, else the kernel mix: the
    kernel diversity of the kernel mix (incl. aliased OpenCV cubic) plus blur / noise / compression coverage.

    ``cfg`` keys: ``real_prob`` (default 0.5), ``real`` and ``kernel_mix`` (sub-configurations, defaults as in
    ``RealVideoDegradation`` / ``KernelMixDegradation``).
    """

    def __init__(self, cfg=None, scale=4):
        cfg = dict(cfg or {})
        self.real_prob = float(cfg.get("real_prob", 0.5))
        self.real = RealVideoDegradation(cfg.get("real"), scale)
        self.kernel_mix = KernelMixDegradation(cfg.get("kernel_mix"), scale)

    def __call__(self, hr_frames, rng=None):
        rng = rng or np.random.default_rng()
        return (self.real if rng.random() < self.real_prob else self.kernel_mix)(hr_frames, rng)


DEGRADATIONS = {"real": RealVideoDegradation, "kernel_mix": KernelMixDegradation, "mixed": MixedDegradation}


def build_degradation(cfg, scale=4):
    """``cfg`` dict with an optional ``type`` key (``real`` by default, or ``kernel_mix``); the rest is the
    degradation's own configuration."""
    cfg = dict(cfg or {})
    kind = cfg.pop("type", "real")
    if kind not in DEGRADATIONS:
        raise ValueError(f"Unknown degradation type {kind!r}; choose from {list(DEGRADATIONS)}")
    return DEGRADATIONS[kind](cfg, scale)
