"""Dataset-level evaluation shared by ``scripts/test.py``, ``scripts/evaluate_outputs.py`` and training."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import imresize, list_frames, read_image, tensor_to_uint8, write_image
from .inference import upscale_sequence
from .metrics import frame_metrics, legacy_clip_metrics

METRIC_KEYS = ("psnr_rgb", "ssim_rgb", "psnr_y", "ssim_y")


def _summarise(per_clip):
    """Mean over frames within each clip, then mean over clips (BasicSR video protocol)."""
    clips = {}
    for clip, frames in per_clip.items():
        clips[clip] = {k: float(np.mean([f[k] for f in frames])) for k in METRIC_KEYS}
        clips[clip]["num_frames"] = len(frames)
    average = {k: float(np.mean([c[k] for c in clips.values()])) for k in METRIC_KEYS}
    return {"average": average, "clips": clips}


def bicubic_upscale(lrs, scale=4, method="torch"):
    """x4 bicubic upsampling of (N, T, C, h, w). ``"matlab"`` uses MATLAB's kernel (a = -0.5) and
    uint8 rounding, which is how the 'Bicubic' rows of VSR papers are computed; ``"torch"`` is
    PyTorch's kernel (a = -0.75), the one Fork-Net uses internally."""
    n, t, c, h, w = lrs.shape
    if method == "matlab":
        frames = [imresize(tensor_to_uint8(f), scale) for f in lrs.reshape(n * t, c, h, w)]
        out = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(255.0)
        return out.view(n, t, c, h * scale, w * scale)
    out = F.interpolate(lrs.reshape(n * t, c, h, w), scale_factor=scale, mode="bicubic", align_corners=False)
    return out.view(n, t, c, h * scale, w * scale)


@torch.no_grad()
def evaluate(model, dataset, device, chunk_size=15, overlap=0, amp=False, crop_border=0, save_dir=None, logger=None,
             bicubic="matlab"):
    """Standard protocol over every frame of every clip. ``model=None`` evaluates bicubic upsampling."""
    per_clip = {}
    for i in range(len(dataset)):
        item = dataset[i]
        lrs = item["lr"][None].to(device)
        if model is None:
            srs = bicubic_upscale(lrs.cpu(), method=bicubic)
        else:
            srs = upscale_sequence(model, lrs, chunk_size, overlap, amp)
        frames = []
        for t, name in enumerate(item["frames"]):
            sr = tensor_to_uint8(srs[0, t])
            if save_dir is not None:
                write_image(Path(save_dir) / item["clip"] / f"{name}.png", sr)
            if item["hr"].numel():
                frames.append({"frame": name, **frame_metrics(sr, tensor_to_uint8(item["hr"][t]), crop_border)})
        per_clip[item["clip"]] = frames
        if logger and frames:
            logger.info(
                f"{item['clip']}: PSNR {np.mean([f['psnr_rgb'] for f in frames]):.4f} dB, "
                f"SSIM {np.mean([f['ssim_rgb'] for f in frames]):.4f} ({len(frames)} frames)"
            )
    if not any(per_clip.values()):
        return {"average": {}, "clips": {}}
    result = _summarise(per_clip)
    result["frames"] = per_clip
    return result


@torch.no_grad()
def evaluate_legacy(model, dataset, device, chunk_size=15, amp=False, logger=None):
    """Re-creates the original validation loop: each clip is cut into disjoint ``chunk_size``-frame
    chunks (an incomplete last chunk is dropped), each chunk is a batch, and the original
    torchmetrics-based PSNR / SSIM are averaged over chunks."""
    scores = []
    for i in range(len(dataset)):
        item = dataset[i]
        lrs, hrs = item["lr"][None].to(device), item["hr"][None].to(device)
        for s in range(0, lrs.size(1) - chunk_size + 1, chunk_size):
            with torch.autocast(device_type=lrs.device.type, dtype=torch.float16, enabled=amp and lrs.device.type == "cuda"):
                sr = model(lrs[:, s : s + chunk_size]) if model is not None else bicubic_upscale(lrs[:, s : s + chunk_size])
            scores.append(legacy_clip_metrics(sr.float(), hrs[:, s : s + chunk_size]))
        if logger:
            logger.info(f"{item['clip']}: {len(scores)} chunks evaluated so far")
    return {
        "average": {"psnr": float(np.mean([s["psnr"] for s in scores])), "ssim": float(np.mean([s["ssim"] for s in scores]))},
        "num_chunks": len(scores),
    }


def evaluate_folders(sr_root, gt_root, clips=None, crop_border=0, logger=None, extra=(), device="cpu"):
    """Scores pre-computed outputs (e.g. from BasicVSR++ / MIA-VSR) with exactly the same metric code.

    Frames are matched by sorted order within each clip folder. ``extra`` may contain ``"lpips"``
    and/or ``"tof"`` (see ``forknet.metrics_extra``); they are reported per clip and averaged.
    """
    from .metrics_extra import LPIPSMetric, TemporalFlowError

    lpips_fn = LPIPSMetric(device) if "lpips" in extra else None
    tof_fn = TemporalFlowError(device) if "tof" in extra else None
    sr_root, gt_root = Path(sr_root), Path(gt_root)
    clips = clips or sorted(p.name for p in gt_root.iterdir() if p.is_dir())
    per_clip, clip_extra = {}, {}
    for clip in clips:
        sr_paths, gt_paths = list_frames(sr_root / clip), list_frames(gt_root / clip)
        if len(sr_paths) != len(gt_paths):
            raise ValueError(f"{clip}: {len(sr_paths)} SR frames vs {len(gt_paths)} GT frames")
        frames, srs, gts = [], [], []
        for sp, gp in zip(sr_paths, gt_paths):
            sr, gt = read_image(sp), read_image(gp)
            if sr.shape != gt.shape:
                raise ValueError(f"{sp}: SR shape {sr.shape} != GT shape {gt.shape}")
            frames.append({"frame": gp.stem, **frame_metrics(sr, gt, crop_border)})
            if extra:
                srs.append(sr)
                gts.append(gt)
        per_clip[clip] = frames
        if lpips_fn:
            clip_extra.setdefault(clip, {})["lpips"] = float(np.mean(lpips_fn(srs, gts)))
        if tof_fn:
            clip_extra.setdefault(clip, {})["tof"] = tof_fn(srs, gts)
        if logger:
            logger.info(f"{clip}: PSNR {np.mean([f['psnr_rgb'] for f in frames]):.4f} dB {clip_extra.get(clip, '')}")
    result = _summarise(per_clip)
    for clip, vals in clip_extra.items():
        result["clips"][clip].update(vals)
    for key in ("lpips", "tof"):
        if key in extra:
            result["average"][key] = float(np.mean([v[key] for v in clip_extra.values()]))
    result["frames"] = per_clip
    return result
