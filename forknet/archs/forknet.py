"""Fork-Net: a U-Net video super-resolution network with second-order
bidirectional propagation (x4 upscaling).

Paper notation <-> code:
    S^1..S^4 (encoder skips)  -> Encoder outputs ``s1`` (4x LR), ``s2`` (2x), ``s3`` (1x), ``s4`` (1/2x)
    Tempo Fuser (TF_j)        -> ``TempoFuser`` (``u_up.tempo_fuser_32`` at 1/2x, ``u_up.tempo_fuser_64`` at 1x)
    A_fwd / A_bwd (Eq. 1, 2)  -> ``TempoFuser.res_forward`` / ``TempoFuser.res_backward``
    Feature Constructor (FC)  -> ``FeatureConstructor`` (``u_up.skip_fuse_64``, ``u_up.skip_fuse_128``)

The level names ``*_32`` / ``*_64`` / ``*_128`` refer to the spatial size for the
64x64 LR training patches, i.e. 1/2x, 1x and 2x the LR resolution.
"""

from dataclasses import asdict, dataclass, fields
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import PROP_BLOCKS, FeatureConstructor, conv3x3, convert_rep_dw, fuse_rep_dw, make_unet_layer
from .flow import flow_warp, resize_flow
from .spynet import SpyNet


@dataclass
class ForkNetConfig:
    num_feat: int = 64
    num_encoder_blocks: int = 3
    # Blocks per propagation branch: one entry conv followed by (num_prop_blocks - 1) aggregation blocks.
    num_prop_blocks: int = 5
    prop_order: int = 2  # 0: no temporal propagation, 1: t-1 / t+1 only, 2: also t-2 / t+2
    prop_direction: str = "bidirectional"  # "bidirectional" | "forward"
    use_flow: bool = True  # False: neighbours are concatenated without motion compensation
    lr_injection: bool = True  # inject features computed from the LR frame into the encoder
    prop_block: str = "convnext"  # "convnext" | "resblock"
    use_bsconv: bool = True
    # BSConv for the fusion / upsampling convs inside the Tempo Fuser (None -> follow use_bsconv).
    bsconv_fusion: Optional[bool] = None
    # Concatenate the un-propagated input feature into the bidirectional fusion (an earlier variant).
    fuse_with_input: bool = False
    # Channels at the deepest encoder level = bottleneck_mult * num_feat.
    bottleneck_mult: int = 2
    # Behaviour of the original research code; see docs/FINDINGS.md.
    legacy_flow_direction: bool = True  # F1: forward/backward flows swapped
    legacy_backward_index: bool = True  # F2: backward branch reads wrong time steps
    legacy_flow_scaling: bool = True  # F3: flow magnitudes not rescaled at 1/2x level
    # Training-time RepVGG branches on every depthwise 3x3 conv (research/12, P2); fold with reparameterize().
    rep_dw: bool = False

    def __post_init__(self):
        if self.prop_order not in (0, 1, 2):
            raise ValueError(f"prop_order must be 0, 1 or 2, got {self.prop_order}")
        if self.prop_direction not in ("bidirectional", "forward"):
            raise ValueError(f"prop_direction must be 'bidirectional' or 'forward', got {self.prop_direction}")
        if self.prop_block not in PROP_BLOCKS:
            raise ValueError(f"prop_block must be one of {list(PROP_BLOCKS)}, got {self.prop_block}")
        if self.bsconv_fusion is None:
            self.bsconv_fusion = self.use_bsconv

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown ForkNetConfig keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self):
        return asdict(self)

    @property
    def is_legacy(self):
        return self.legacy_flow_direction or self.legacy_backward_index or self.legacy_flow_scaling


class Encoder(nn.Module):
    """U-Net encoder operating on the x4 bicubic-upsampled frame, with LR features
    injected at the LR-resolution level (Fig. 1)."""

    def __init__(self, cfg: ForkNetConfig):
        super().__init__()
        nf, nb, bs = cfg.num_feat, cfg.num_encoder_blocks, cfg.use_bsconv
        self.conv_1 = make_unet_layer(3, nf, 2, bs)
        self.conv_2 = make_unet_layer(nf, nf, 2, bs)
        self.lr_injection = cfg.lr_injection
        if cfg.lr_injection:
            self.conv_3_pre_net = make_unet_layer(3, nf, nb, bs)
        # Without LR injection conv_3 maps C -> 2C, so all later shapes are unchanged.
        self.conv_3 = make_unet_layer(nf * 2 if cfg.lr_injection else nf, nf * 2, nb, bs)
        self.conv_4 = make_unet_layer(nf * 2, nf * cfg.bottleneck_mult, nb, bs)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, lrs):
        n, t, c, h, w = lrs.shape
        lr = lrs.reshape(n * t, c, h, w)
        s1 = self.conv_1(F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False))
        s2 = self.conv_2(self.maxpool(s1))
        x3 = self.maxpool(s2)
        s3 = self.conv_3(torch.cat([self.conv_3_pre_net(lr), x3], dim=1) if self.lr_injection else x3)
        s4 = self.conv_4(self.maxpool(s3))
        return tuple(s.view(n, t, *s.shape[1:]) for s in (s1, s2, s3, s4))


class TempoFuser(nn.Module):
    """Bidirectional (second-order) temporal propagation, fusion and x2 upsampling.

    Implements Eq. 1-3: forward and backward recurrent aggregation over flow-warped
    features of t-1, t-2 (resp. t+1, t+2), fusion of both directions, then
    conv + PixelShuffle(2). ``up_mult`` sets the output channels to
    ``num_feat * up_mult / 4``.
    """

    def __init__(self, num_feat, cfg: ForkNetConfig, up_mult=4):
        super().__init__()
        self.order = cfg.prop_order
        self.legacy_backward_index = cfg.legacy_backward_index
        block = PROP_BLOCKS[cfg.prop_block]
        n_in = (self.order + 1) * num_feat

        def branch():
            return nn.Sequential(
                conv3x3(n_in, num_feat, cfg.use_bsconv),
                nn.LeakyReLU(0.1),
                *[block(num_feat) for _ in range(cfg.num_prop_blocks - 1)],
            )

        self.bidirectional = cfg.prop_direction == "bidirectional"
        self.res_forward = branch()
        if self.bidirectional:
            self.res_backward = branch()
        self.fuse_with_input = cfg.fuse_with_input
        n_fuse = (2 if self.bidirectional else 1) + int(cfg.fuse_with_input)
        self.fuse_net = conv3x3(n_fuse * num_feat, num_feat, cfg.bsconv_fusion)
        self.up_net = conv3x3(num_feat, num_feat * up_mult, cfg.bsconv_fusion)
        self.pixel_shuffle = nn.PixelShuffle(2)

    @staticmethod
    def _warp(feat, flow):
        return flow_warp(feat, flow.permute(0, 2, 3, 1), clamp=100)

    def forward(self, feats, flows):
        """feats: (N, T, C, H, W); flows: dict of (N, T-k, 2, H, W) tensors, see ForkNet.compute_flows."""
        t_len = feats.size(1)

        fwd = []
        for t in range(t_len):
            cur = feats[:, t]
            prev1, prev2 = cur.new_zeros(cur.shape), cur.new_zeros(cur.shape)
            if self.order >= 1 and t > 0:
                prev1 = self._warp(fwd[t - 1], flows["forward_1"][:, t - 1])
            if self.order == 2 and t > 1:
                prev2 = self._warp(fwd[t - 2], flows["forward_2"][:, t - 2])
            inputs = [[cur], [prev1, cur], [prev2, prev1, cur]][self.order]
            fwd.append(self.res_forward(torch.cat(inputs, dim=1)))

        bwd = [None] * t_len
        if self.bidirectional:
            done = []  # filled from t = T-1 down to 0, so done[-1] is time t+1
            for t in range(t_len - 1, -1, -1):
                cur = feats[:, t]
                next1, next2 = cur.new_zeros(cur.shape), cur.new_zeros(cur.shape)
                if self.order >= 1 and t < t_len - 1:
                    src = done[t - t_len + 2] if self.legacy_backward_index else done[-1]
                    next1 = self._warp(src, flows["backward_1"][:, t])
                if self.order == 2 and t < t_len - 2:
                    src = done[t - t_len + 3] if self.legacy_backward_index else done[-2]
                    next2 = self._warp(src, flows["backward_2"][:, t])
                inputs = [[cur], [cur, next1], [cur, next1, next2]][self.order]
                done.append(self.res_backward(torch.cat(inputs, dim=1)))
            bwd = done[::-1]

        out = []
        for t in range(t_len):
            parts = [fwd[t]] + ([feats[:, t]] if self.fuse_with_input else []) + ([bwd[t]] if self.bidirectional else [])
            out.append(self.pixel_shuffle(self.up_net(self.fuse_net(torch.cat(parts, dim=1)))))
        return torch.stack(out, dim=1)


class Decoder(nn.Module):
    def __init__(self, cfg: ForkNetConfig):
        super().__init__()
        nf = cfg.num_feat
        nb = nf * cfg.bottleneck_mult
        self.legacy_flow_scaling = cfg.legacy_flow_scaling
        self.tempo_fuser_32 = TempoFuser(nb, cfg, up_mult=4)  # 1/2x -> 1x, nb channels
        self.skip_fuse_64 = FeatureConstructor(nb + nf * 2, nf * 2, cfg.use_bsconv)
        self.tempo_fuser_64 = TempoFuser(nf * 2, cfg, up_mult=2)  # 1x -> 2x, nf channels
        self.skip_fuse_128 = FeatureConstructor(nf * 2, nf * 4, cfg.use_bsconv)
        self.pixel_shuffle = nn.PixelShuffle(2)  # 2x -> 4x, nf channels
        self.final_conv = nn.Conv2d(nf * 2, 3, 3, padding=1)

    @staticmethod
    def _per_frame(module, x):
        n, t = x.shape[:2]
        y = module(x.flatten(0, 1))
        return y.view(n, t, *y.shape[1:])

    def forward(self, lrs, flows, s1, s2, s3, s4):
        size = s4.shape[-2:]
        flows_half = {
            k: resize_flow(v, size, rescale_values=not self.legacy_flow_scaling) if v is not None else None
            for k, v in flows.items()
        }
        h4 = self.tempo_fuser_32(s4, flows_half)
        f3 = self._per_frame(self.skip_fuse_64, torch.cat([h4, s3], dim=2))
        h3 = self.tempo_fuser_64(f3, flows)
        f2 = self._per_frame(self.skip_fuse_128, torch.cat([h3, s2], dim=2))
        f1 = self._per_frame(self.pixel_shuffle, f2)
        residual = self._per_frame(self.final_conv, torch.cat([f1, s1], dim=2))
        n, t, c, h, w = lrs.shape
        base = F.interpolate(lrs.reshape(n * t, c, h, w), scale_factor=4, mode="bicubic", align_corners=False)
        return base.view_as(residual) + residual


class ForkNet(nn.Module):
    """Fork-Net for x4 VSR. Input (N, T, 3, H, W) in [0, 1]; output (N, T, 3, 4H, 4W)."""

    scale = 4

    def __init__(self, cfg: Optional[ForkNetConfig] = None, spynet_path=None):
        super().__init__()
        self.cfg = cfg or ForkNetConfig()
        self.spynet = SpyNet(spynet_path)
        self.u_down = Encoder(self.cfg)
        self.u_up = Decoder(self.cfg)
        if self.cfg.rep_dw:
            convert_rep_dw(self)

    def reparameterize(self):
        """Folds the rep_dw training branches into plain convs (identical output); returns self, now a
        standard model whose config has rep_dw=False."""
        if self.cfg.rep_dw:
            fuse_rep_dw(self)
            self.cfg = ForkNetConfig.from_dict({**self.cfg.to_dict(), "rep_dw": False})
        return self

    def _spynet_pairs(self, lrs, k):
        """Returns (a, b) with a[:, i] = spynet(x_i, x_{i+k}) and b[:, i] = spynet(x_{i+k}, x_i)."""
        n, t, c, h, w = lrs.shape
        x0 = lrs[:, :-k].reshape(-1, c, h, w)
        x1 = lrs[:, k:].reshape(-1, c, h, w)
        a = self.spynet(x0, x1).view(n, t - k, 2, h, w)
        b = self.spynet(x1, x0).view(n, t - k, 2, h, w)
        return a, b

    def compute_flows(self, lrs):
        """Flows used for propagation.

        ``forward_k[:, i]`` warps features of frame i to frame i+k and
        ``backward_k[:, i]`` warps features of frame i+k to frame i. With
        ``legacy_flow_direction`` the two are swapped, reproducing the original code.
        """
        n, t, _, h, w = lrs.shape
        flows = {"forward_1": None, "backward_1": None, "forward_2": None, "backward_2": None}
        for k in (1, 2):
            if k > self.cfg.prop_order or t <= k:
                continue
            if not self.cfg.use_flow:
                zeros = lrs.new_zeros(n, t - k, 2, h, w)
                flows[f"forward_{k}"], flows[f"backward_{k}"] = zeros, zeros
                continue
            a, b = self._spynet_pairs(lrs, k)
            fwd, bwd = (a, b) if self.cfg.legacy_flow_direction else (b, a)
            flows[f"forward_{k}"], flows[f"backward_{k}"] = fwd, bwd
        return flows

    def forward(self, lrs):
        h, w = lrs.shape[-2:]
        pad_h, pad_w = h % 2, w % 2  # the 1/2x level needs even LR sizes
        if pad_h or pad_w:
            n, t = lrs.shape[:2]
            lrs = F.pad(lrs.flatten(0, 1), (0, pad_w, 0, pad_h), mode="replicate").view(n, t, 3, h + pad_h, w + pad_w)
        flows = self.compute_flows(lrs)
        s1, s2, s3, s4 = self.u_down(lrs)
        out = self.u_up(lrs, flows, s1, s2, s3, s4)
        return out[..., : 4 * h, : 4 * w]

    def set_grad_checkpointing(self, enabled=True):
        """Recompute block activations in the backward pass (training memory roughly halved, ~30 % slower)."""
        from .blocks import CheckpointMixin

        for m in self.modules():
            if isinstance(m, CheckpointMixin):
                m.grad_checkpointing = enabled
        return self

    def num_params(self, include_spynet=True):
        return sum(p.numel() for n, p in self.named_parameters() if include_spynet or not n.startswith("spynet."))
