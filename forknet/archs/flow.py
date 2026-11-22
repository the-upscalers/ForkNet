"""Optical-flow utilities."""

import torch
import torch.nn.functional as F


def flow_warp(x, flow, interp_mode="bilinear", padding_mode="zeros", align_corners=True, clamp=None):
    """Backward-warp ``x`` with ``flow``: ``out(p) = x(p + flow(p))``.

    Args:
        x: (N, C, H, W) tensor.
        flow: (N, H, W, 2) tensor of (dx, dy) displacements in pixels.
        clamp: optional bound on the absolute flow value.
    """
    assert x.shape[-2:] == flow.shape[1:3], (x.shape, flow.shape)
    _, _, h, w = x.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=x.dtype, device=x.device),
        torch.arange(w, dtype=x.dtype, device=x.device),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), 2)
    if clamp is not None:
        flow = flow.clamp(-clamp, clamp)
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[..., 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(h - 1, 1) - 1.0
    vgrid = torch.stack((vgrid_x, vgrid_y), dim=3)
    return F.grid_sample(x, vgrid, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)


def resize_flow(flow, size, rescale_values=True):
    """Bilinearly resize a (N, T, 2, H, W) flow field to ``size`` = (H', W').

    When ``rescale_values`` is True the displacement vectors are scaled by the
    resize ratio, which is required for the flow to stay geometrically valid.
    The original Fork-Net code did not rescale.
    """
    n, t, _, h, w = flow.shape
    out = F.interpolate(flow.reshape(n * t, 2, h, w), size=size, mode="bilinear", align_corners=False)
    if rescale_values:
        out = out * torch.tensor([size[1] / w, size[0] / h], dtype=out.dtype, device=out.device).view(1, 2, 1, 1)
    return out.view(n, t, 2, *size)
