"""Convolutional building blocks of Fork-Net.

Attribute names (``conv1``, ``pw``/``dw``, ``c1_d_3`` ...) intentionally mirror the
original research code so that checkpoints trained with it load with ``strict=True``.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class CheckpointMixin:
    """Optional activation checkpointing: when ``grad_checkpointing`` is set, the block's activations are
    recomputed in the backward pass instead of stored (same result, less memory, ~30 % slower)."""

    grad_checkpointing = False

    def forward(self, x):
        if self.grad_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class BSConvU(nn.Sequential):
    """Unconstrained blueprint-separable convolution (Haase & Amthor, CVPR 2020).

    A bias-free 1x1 pointwise convolution followed by a KxK depthwise convolution.
    Numerically and structurally identical to ``bsconv.pytorch.BSConvU(with_bn=False)``.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True):
        super().__init__()
        self.add_module("pw", nn.Conv2d(in_channels, out_channels, 1, bias=False))
        self.add_module(
            "dw",
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                stride,
                padding=kernel_size // 2,
                groups=out_channels,
                bias=bias,
            ),
        )


def conv3x3(in_channels, out_channels, bsconv=True):
    """3x3 convolution, either blueprint-separable or standard."""
    if bsconv:
        return BSConvU(in_channels, out_channels, 3)
    return nn.Conv2d(in_channels, out_channels, 3, padding=1)


class UNetBlock(CheckpointMixin, nn.Module):
    """Encoder block: conv-ReLU-conv with an identity shortcut when shapes allow.

    Adapted from SR-UNet (Vaccaro et al., 2021).
    """

    def __init__(self, in_channels, out_channels, bsconv=True):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, bsconv)
        self.conv2 = conv3x3(out_channels, out_channels, bsconv)
        self.act = nn.ReLU(inplace=True)
        self.residual = in_channels == out_channels

    def _forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))
        return out + x if self.residual else out


def make_unet_layer(in_channels, out_channels, num_blocks, bsconv=True):
    blocks = [UNetBlock(in_channels, out_channels, bsconv)]
    blocks += [UNetBlock(out_channels, out_channels, bsconv) for _ in range(num_blocks - 1)]
    return nn.Sequential(*blocks)


class ConvNeXtBlock(CheckpointMixin, nn.Module):
    """ConvNeXt-style inverted bottleneck used for temporal aggregation.

    depthwise 3x3 -> pointwise (4x expansion) -> GELU -> pointwise, plus identity.
    Differs from the original ConvNeXt block (Liu et al., CVPR 2022) in using a 3x3
    depthwise kernel and no LayerNorm / LayerScale.
    """

    def __init__(self, num_feat, expansion=4):
        super().__init__()
        self.c1_d_3 = nn.Conv2d(num_feat, num_feat, 3, padding=1, groups=num_feat)
        self.c1_r_1 = nn.Conv2d(num_feat, num_feat * expansion, 1)
        self.c1_r_2 = nn.Conv2d(num_feat * expansion, num_feat, 1)
        self.act = nn.GELU()

    def _forward(self, x):
        return x + self.c1_r_2(self.act(self.c1_r_1(self.c1_d_3(x))))


class ResidualBlock(CheckpointMixin, nn.Module):
    """Plain residual block without BatchNorm (conv-ReLU-conv + identity), as in BasicVSR."""

    def __init__(self, num_feat):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.act = nn.ReLU()

    def _forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


PROP_BLOCKS = {"convnext": ConvNeXtBlock, "resblock": ResidualBlock}


class FeatureConstructor(CheckpointMixin, nn.Module):
    """Fuses upsampled temporal features with encoder skip features (FC in Eq. 4)."""

    def __init__(self, in_channels, out_channels, bsconv=True):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, bsconv)
        self.conv2 = conv3x3(out_channels, out_channels, bsconv)
        self.act = nn.ReLU(inplace=True)

    def _forward(self, x):
        return self.act(self.conv2(self.act(self.conv1(x))))


class RepDWConv(nn.Module):
    """Training-time re-parameterisable depthwise 3x3 conv (RepVGG-style, pilot P2 in research/12).

    ``conv3(x) + conv1(x) + x`` with ``conv3`` the original depthwise 3x3 and ``conv1`` a depthwise 1x1;
    all three branches are linear, so :meth:`fuse` folds them into one depthwise 3x3 conv with an identical
    output. Inference cost and parameter count after folding equal the plain model's.
    """

    def __init__(self, conv3: nn.Conv2d):
        super().__init__()
        c = conv3.out_channels
        self.conv3 = conv3
        self.conv1 = nn.Conv2d(c, c, 1, groups=c, bias=True)
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

    def forward(self, x):
        return self.conv3(x) + self.conv1(x) + x

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        c3 = self.conv3
        fused = nn.Conv2d(c3.in_channels, c3.out_channels, 3, padding=1, groups=c3.groups, bias=True)
        w = c3.weight.clone()
        w[:, :, 1, 1] += self.conv1.weight[:, :, 0, 0] + 1.0  # 1x1 branch and identity sit at the kernel centre
        b = self.conv1.bias.clone() + (c3.bias if c3.bias is not None else 0)
        fused.weight.copy_(w)
        fused.bias.copy_(b)
        return fused.to(c3.weight.device, c3.weight.dtype)


def _is_dw3x3(m):
    return (isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3) and m.stride == (1, 1) and m.padding == (1, 1)
            and m.groups == m.in_channels == m.out_channels)


def convert_rep_dw(module, skip=("spynet",)):
    """Wraps every depthwise 3x3 conv under ``module`` (except in children named in ``skip``) in RepDWConv."""
    for name, child in module.named_children():
        if name in skip:
            continue
        if _is_dw3x3(child):
            setattr(module, name, RepDWConv(child))
        else:
            convert_rep_dw(child, skip)
    return module


def fuse_rep_dw(module):
    """Replaces every RepDWConv under ``module`` by its folded plain conv (in place)."""
    for name, child in module.named_children():
        if isinstance(child, RepDWConv):
            setattr(module, name, child.fuse())
        else:
            fuse_rep_dw(child)
    return module
