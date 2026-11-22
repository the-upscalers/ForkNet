import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPYNET_WEIGHTS = ROOT / "weights" / "spynet_sintel_final-3d2a1287.pth"


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def smooth_texture(h, w, seed=0):
    """Smooth random RGB texture (uint8) on which optical flow is well defined."""
    rng = np.random.default_rng(seed)
    small = torch.from_numpy(rng.random((1, 3, h // 8 + 2, w // 8 + 2), dtype=np.float32))
    big = torch.nn.functional.interpolate(small, size=(h, w), mode="bicubic", align_corners=False)
    return (big[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)


@pytest.fixture
def make_video_dataset(tmp_path):
    """Writes a synthetic paired dataset <tmp>/HR/<clip>/*.png, <tmp>/LR/<clip>/*.png with moving texture."""
    from forknet.data import imresize, write_image

    def _make(num_clips=3, num_frames=8, lr_size=(64, 64), scale=4):
        hr_h, hr_w = lr_size[0] * scale, lr_size[1] * scale
        for c in range(num_clips):
            base = smooth_texture(hr_h + 64, hr_w + 64, seed=c)
            for t in range(num_frames):
                hr = base[2 * t : 2 * t + hr_h, 3 * t : 3 * t + hr_w]
                write_image(tmp_path / "HR" / f"{c:03d}" / f"{t:08d}.png", hr)
                write_image(tmp_path / "LR" / f"{c:03d}" / f"{t:08d}.png", imresize(hr, 1 / scale))
        return tmp_path / "LR", tmp_path / "HR"

    return _make
