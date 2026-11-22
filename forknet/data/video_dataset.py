"""Paired LR/HR video datasets laid out as ``<root>/<clip>/<frame>.png``.

Used for REDS, Vid4 and the license-plate (LP) video dataset alike.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .degradations import build_degradation
from .io import list_clips, list_frames, read_image

REDS4 = ("000", "011", "015", "020")


def resolve_clips(lr_root, clips=None, exclude=None, split_file=None, split=None):
    """Selects clip names from an explicit list, a JSON split file, or all sub-folders of ``lr_root``."""
    if split_file is not None:
        with open(split_file) as f:
            clips = json.load(f)[split]
    elif clips is None:
        clips = list_clips(lr_root)
    clips = [str(c) for c in clips]
    if exclude:
        excluded = {str(c) for c in exclude}
        clips = [c for c in clips if c not in excluded]
    if not clips:
        raise ValueError(f"No clips selected under {lr_root}")
    return clips


def _to_tensor(frames):
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div_(255.0)


class VideoTrainDataset(Dataset):
    """Random spatio-temporal crops for training.

    Args:
        num_frames: frames per training sample (15 for REDS).
        patch_size: LR patch size (64).
        temporal_sampling: ``"fixed_chunks"`` reproduces the original code: each clip is split
            into non-overlapping ``num_frames`` windows (frames past the last full window are
            never used). ``"random"`` draws a random window start per sample (BasicVSR style).
        augment: ``"flips"`` applies random horizontal and vertical flips shared by all frames.
            ``"legacy"`` reproduces the original code, whose flips acted on a (T, C, H, W)
            tensor along dims 2 and 1, i.e. a vertical flip and an RGB<->BGR channel
            reversal (docs/FINDINGS.md, F5). ``"none"`` disables augmentation.
        rot90, temporal_reverse: optional extra augmentations (off in the paper setting).
        degradation: if given (a dict, possibly empty), LR frames are synthesised on the fly from the
            HR crop with ``build_degradation(degradation)`` (``type: real`` by default, or ``kernel_mix``)
            and ``lr_root`` may be None.
        plate_boxes: optional JSON ``{clip: {frame_stem: [x1, y1, x2, y2]}}`` of plate boxes on the HR frames
            (``scripts/lp/plate_boxes.py``). Items are then ``(lr, hr, mask)`` with a (T, 1, H, W) plate mask that
            follows the same crop and flips, for plate-region losses.
    """

    def __init__(
        self,
        lr_root,
        hr_root,
        scale=4,
        num_frames=15,
        patch_size=64,
        temporal_sampling="fixed_chunks",
        clips=None,
        exclude=None,
        split_file=None,
        split=None,
        augment="flips",
        rot90=False,
        temporal_reverse=False,
        degradation=None,
        plate_boxes=None,
    ):
        self.plate_boxes = json.loads(Path(plate_boxes).read_text()) if plate_boxes else None
        if temporal_sampling not in ("fixed_chunks", "random"):
            raise ValueError(temporal_sampling)
        if augment not in ("flips", "legacy", "none"):
            raise ValueError(augment)
        self.hr_root = Path(hr_root)
        self.degrade = build_degradation(degradation, scale) if degradation is not None else None
        self.lr_root = Path(lr_root) if lr_root else None
        if self.degrade is None and self.lr_root is None:
            raise ValueError("lr_root is required unless a degradation is configured")
        self.scale, self.num_frames, self.patch_size = scale, num_frames, patch_size
        self.temporal_sampling = temporal_sampling
        self.augment, self.rot90, self.temporal_reverse = augment, rot90, temporal_reverse
        self.clips = resolve_clips(self.lr_root or self.hr_root, clips, exclude, split_file, split)
        self.hr_frames = {c: list_frames(self.hr_root / c) for c in self.clips}
        self.lr_frames = {c: list_frames(self.lr_root / c) for c in self.clips} if self.lr_root else self.hr_frames
        self.index = []
        for c in self.clips:
            n = len(self.lr_frames[c])
            if n != len(self.hr_frames[c]):
                raise ValueError(f"Clip {c}: {n} LR frames vs {len(self.hr_frames[c])} HR frames")
            if n < num_frames:
                raise ValueError(f"Clip {c} has {n} frames, fewer than num_frames={num_frames}")
            self.index += [(c, k * num_frames) for k in range(n // num_frames)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        clip, start = self.index[idx]
        if self.temporal_sampling == "random":
            start = random.randint(0, len(self.lr_frames[clip]) - self.num_frames)
        names = range(start, start + self.num_frames)
        hrs = [read_image(self.hr_frames[clip][i]) for i in names]
        masks = None
        if self.plate_boxes is not None:
            boxes = self.plate_boxes.get(clip, {})
            masks = []
            for i in names:
                m = np.zeros((*hrs[0].shape[:2], 1), np.uint8)
                b = boxes.get(self.hr_frames[clip][i].stem)
                if b:
                    m[max(0, int(b[1])) : int(b[3]), max(0, int(b[0])) : int(b[2])] = 255
                masks.append(m)
        ps, s = self.patch_size, self.scale
        if self.degrade is not None:
            hh, ww = hrs[0].shape[:2]
            if ps is not None:
                y, x = random.randint(0, hh - ps * s), random.randint(0, ww - ps * s)
                hrs = [f[y : y + ps * s, x : x + ps * s] for f in hrs]
                masks = masks and [f[y : y + ps * s, x : x + ps * s] for f in masks]
            else:
                hrs = [f[: hh - hh % s, : ww - ww % s] for f in hrs]
                masks = masks and [f[: hh - hh % s, : ww - ww % s] for f in masks]
            lrs = self.degrade(hrs, np.random.default_rng(random.getrandbits(64)))
            ps = None  # already cropped
        else:
            lrs = [read_image(self.lr_frames[clip][i]) for i in names]

        h, w = lrs[0].shape[:2]
        if hrs[0].shape[0] != h * s or hrs[0].shape[1] != w * s:
            raise ValueError(f"Clip {clip}: HR size {hrs[0].shape[:2]} is not {s}x LR size {(h, w)}")
        if ps is not None:
            if h < ps or w < ps:
                raise ValueError(f"Clip {clip}: LR size {(h, w)} smaller than patch {ps}")
            y, x = random.randint(0, h - ps), random.randint(0, w - ps)
            lrs = [f[y : y + ps, x : x + ps] for f in lrs]
            hrs = [f[y * s : (y + ps) * s, x * s : (x + ps) * s] for f in hrs]
            masks = masks and [f[y * s : (y + ps) * s, x * s : (x + ps) * s] for f in masks]

        lr, hr = _to_tensor(lrs), _to_tensor(hrs)  # (T, C, H, W)
        if masks is not None:  # the mask travels as a 4th HR channel through the shared augmentations
            hr = torch.cat([hr, _to_tensor(masks)], dim=1)
        if self.augment != "none":
            # Same draw order as the original code: first flip, then second flip.
            first_dim, second_dim = (-2, 1) if self.augment == "legacy" else (-1, -2)
            if random.random() < 0.5:
                lr, hr = lr.flip(first_dim), hr.flip(first_dim)
            if random.random() < 0.5:
                lr, hr = lr.flip(second_dim), hr.flip(second_dim)
        if self.rot90 and random.random() < 0.5:
            lr, hr = lr.transpose(-1, -2), hr.transpose(-1, -2)
        if self.temporal_reverse and random.random() < 0.5:
            lr, hr = lr.flip(0), hr.flip(0)
        if masks is not None:
            return lr.contiguous(), hr[:, :3].contiguous(), hr[:, 3:].contiguous()
        return lr.contiguous(), hr.contiguous()


class VideoTestDataset(Dataset):
    """Whole sequences for evaluation. Each item is one clip.

    If ``hr_root`` is None (inference on unpaired LR video), ``hr`` is an empty tensor.
    """

    def __init__(self, lr_root, hr_root=None, clips=None, exclude=None, split_file=None, split=None, max_frames=None):
        self.lr_root = Path(lr_root)
        self.hr_root = Path(hr_root) if hr_root else None
        self.clips = resolve_clips(lr_root, clips, exclude, split_file, split)
        self.max_frames = max_frames

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        lr_paths = list_frames(self.lr_root / clip)[: self.max_frames]
        lr = _to_tensor([read_image(p) for p in lr_paths])
        hr = torch.empty(0)
        if self.hr_root is not None:
            hr_paths = list_frames(self.hr_root / clip)[: self.max_frames]
            if len(hr_paths) != len(lr_paths):
                raise ValueError(f"Clip {clip}: {len(lr_paths)} LR vs {len(hr_paths)} HR frames")
            hr = _to_tensor([read_image(p) for p in hr_paths])
        return {"lr": lr, "hr": hr, "clip": clip, "frames": [p.stem for p in lr_paths]}
