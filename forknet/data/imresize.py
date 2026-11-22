"""MATLAB-compatible ``imresize`` (bicubic, antialiased), used to synthesise LR frames.

REDS and Vid4 (BI) LR frames were produced with MATLAB's bicubic ``imresize``; using the
same kernel for the license-plate dataset keeps the degradation consistent.
"""

import math

import numpy as np


def _cubic(x):
    ax = np.abs(x)
    ax2, ax3 = ax**2, ax**3
    return (1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1) + (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((ax > 1) & (ax <= 2))


def _contributions(in_len, out_len, scale, antialias=True):
    kernel_width = 4.0
    if scale < 1 and antialias:
        kernel = lambda x: scale * _cubic(scale * x)  # noqa: E731
        kernel_width /= scale
    else:
        kernel = _cubic
    x = np.arange(1, out_len + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    taps = int(math.ceil(kernel_width)) + 2
    indices = left[:, None] + np.arange(taps)[None, :]
    weights = kernel(u[:, None] - indices)
    weights /= weights.sum(axis=1, keepdims=True)
    # Symmetric boundary handling, as in MATLAB.
    mirror = np.concatenate([np.arange(in_len), np.arange(in_len - 1, -1, -1)])
    indices = mirror[np.mod(indices.astype(np.int64) - 1, mirror.size)]
    keep = np.any(weights != 0, axis=0)
    return weights[:, keep], indices[:, keep]


def _resize_axis(img, weights, indices, axis):
    img = np.moveaxis(img, axis, 0)
    out = np.einsum("ij,ij...->i...", weights, img[indices])
    return np.moveaxis(out, 0, axis)


def imresize(img, scale, antialias=True):
    """Resizes an (H, W[, C]) image by ``scale``.

    uint8 input returns uint8 (rounded, like MATLAB); float input returns float64.
    """
    is_uint8 = img.dtype == np.uint8
    out = img.astype(np.float64)
    h, w = img.shape[:2]
    for axis, (n_in, n_out) in enumerate(((h, math.ceil(h * scale)), (w, math.ceil(w * scale)))):
        weights, indices = _contributions(n_in, n_out, scale, antialias)
        out = _resize_axis(out, weights, indices, axis)
    if is_uint8:
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out
