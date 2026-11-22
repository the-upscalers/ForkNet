"""Checkpoint I/O, including checkpoints written by the original research code.

Release format: ``{"params": state_dict, "config": ForkNetConfig dict, "iter": int, ...}``.
Original format: ``{"model_state_dict": state_dict, "epoch": int, ...}``; its architecture is
inferred from tensor shapes, and the three ``legacy_*`` behaviours are assumed to be on,
because every original variant had them (docs/FINDINGS.md).
"""

import re
from pathlib import Path

import torch

from ..archs.forknet import ForkNet, ForkNetConfig


def _strip_prefix(state_dict, prefix="module."):
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def _conv_in_out(sd, key):
    """(in, out) channels of a conv or BSConv stored under ``key``."""
    if f"{key}.pw.weight" in sd:
        w = sd[f"{key}.pw.weight"]
    else:
        w = sd[f"{key}.weight"]
    return w.shape[1], w.shape[0]


def infer_config(sd):
    """Infers a ForkNetConfig from a state dict produced by any Fork-Net variant."""
    use_bsconv = "u_down.conv_1.0.conv1.pw.weight" in sd
    _, nf = _conv_in_out(sd, "u_down.conv_1.0.conv1")
    enc_blocks = {int(m.group(1)) for k in sd if (m := re.match(r"u_down\.conv_3_pre_net\.(\d+)\.", k))}
    _, c4 = _conv_in_out(sd, "u_down.conv_4.0.conv1")
    tf = "u_up.tempo_fuser_32"
    prop_blocks = {int(m.group(1)) for k in sd if (m := re.match(rf"{tf}\.res_forward\.(\d+)\.", k))} - {0}
    prop_block = "convnext" if f"{tf}.res_forward.2.c1_d_3.weight" in sd else "resblock"
    res_in, mid = _conv_in_out(sd, f"{tf}.res_forward.0")
    fuse_in, _ = _conv_in_out(sd, f"{tf}.fuse_net")
    bidirectional = any(k.startswith(f"{tf}.res_backward.") for k in sd)
    lr_injection = any(k.startswith("u_down.conv_3_pre_net.") for k in sd)
    return ForkNetConfig(
        num_feat=nf,
        num_encoder_blocks=len(enc_blocks) if lr_injection else
        len({int(m.group(1)) for k in sd if (m := re.match(r"u_down\.conv_3\.(\d+)\.", k))}),
        num_prop_blocks=len(prop_blocks) + 1,
        prop_order=res_in // mid - 1,
        prop_direction="bidirectional" if bidirectional else "forward",
        lr_injection=lr_injection,
        prop_block=prop_block,
        use_bsconv=use_bsconv,
        bsconv_fusion=f"{tf}.fuse_net.pw.weight" in sd,
        fuse_with_input=fuse_in == ((2 if bidirectional else 1) + 1) * mid,
        bottleneck_mult=c4 // nf,
        legacy_flow_direction=True,
        legacy_backward_index=True,
        legacy_flow_scaling=True,
    )


def load_checkpoint(path, map_location="cpu"):
    """Returns ``(state_dict, ForkNetConfig, meta)`` for a release or original checkpoint."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if "params" in ckpt:
        sd, cfg = ckpt["params"], ForkNetConfig.from_dict(ckpt["config"]) if "config" in ckpt else None
        fmt = "release"
    elif "model_state_dict" in ckpt:
        sd, cfg, fmt = ckpt["model_state_dict"], None, "original"
    elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        sd, cfg, fmt = ckpt, None, "state_dict"
    else:
        raise ValueError(f"Unrecognised checkpoint format: keys {list(ckpt)[:10]}")
    sd = _strip_prefix(sd)
    if cfg is None:
        cfg = infer_config(sd)
    meta = {k: v for k, v in ckpt.items() if k in ("iter", "epoch", "psnr", "ssim", "loss")}
    meta["format"] = fmt
    return sd, cfg, meta


def build_model_from_checkpoint(path, config_overrides=None, map_location="cpu"):
    """Builds a ForkNet and loads weights with ``strict=True``.

    ``config_overrides`` may only change behaviour flags (``legacy_*``), never shapes.
    """
    sd, cfg, meta = load_checkpoint(path, map_location)
    if config_overrides:
        allowed = {"legacy_flow_direction", "legacy_backward_index", "legacy_flow_scaling"}
        bad = set(config_overrides) - allowed
        if bad:
            raise ValueError(f"Only {sorted(allowed)} can be overridden for a trained checkpoint, got {sorted(bad)}")
        cfg = ForkNetConfig.from_dict({**cfg.to_dict(), **config_overrides})
    model = ForkNet(cfg)
    model.load_state_dict(sd, strict=True)
    return model, meta


def save_checkpoint(path, model, cfg, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.module if hasattr(model, "module") else model
    torch.save({"params": model.state_dict(), "config": cfg.to_dict(), **extra}, path)
