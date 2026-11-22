"""Optional metrics beyond PSNR / SSIM.

* LPIPS (Zhang et al., 2018), AlexNet variant - perceptual distance, lower is better. Needs ``lpips``.
* tOF - temporal optical-flow error: mean |flow(SR_t, SR_t+1) - flow(GT_t, GT_t+1)| in pixels, computed
  with the bundled SpyNet; measures flicker / temporal inconsistency, lower is better.
* No-reference NIQE / BRISQUE for real-world videos without ground truth (VideoLQ protocol of
  RealBasicVSR), lower is better. Needs ``pyiqa``.
"""

from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _to_tensor(frames, device):
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(255).to(device)


class LPIPSMetric:
    def __init__(self, device="cpu"):
        import lpips

        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, sr_frames, gt_frames):
        sr, gt = _to_tensor(sr_frames, self.device) * 2 - 1, _to_tensor(gt_frames, self.device) * 2 - 1
        return [float(v) for v in self.net(sr, gt).flatten()]


class TemporalFlowError:
    def __init__(self, device="cpu", spynet_path=ROOT / "weights" / "spynet_sintel_final-3d2a1287.pth"):
        from .archs.spynet import SpyNet

        self.net = SpyNet(str(spynet_path)).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, sr_frames, gt_frames):
        if len(sr_frames) < 2:
            return float("nan")
        sr, gt = _to_tensor(sr_frames, self.device), _to_tensor(gt_frames, self.device)
        errs = []
        for t in range(len(sr_frames) - 1):  # one pair at a time to bound memory at HR resolution
            f_sr = self.net(sr[t + 1 : t + 2], sr[t : t + 1])
            f_gt = self.net(gt[t + 1 : t + 2], gt[t : t + 1])
            errs.append(float((f_sr - f_gt).norm(dim=1).mean()))
        return float(np.mean(errs))


class NoReferenceMetrics:
    def __init__(self, names=("niqe", "brisque"), device="cpu"):
        import pyiqa

        self.metrics = {n: pyiqa.create_metric(n, device=device) for n in names}
        self.device = device

    @torch.no_grad()
    def __call__(self, frames):
        x = _to_tensor(frames, self.device)
        return {n: float(np.mean([float(m(x[i : i + 1])) for i in range(len(x))])) for n, m in self.metrics.items()}
