"""SpyNet optical-flow estimator (Ranjan & Black, CVPR 2017).

Port of the BasicSR implementation. Pretrained weights:
``weights/spynet_sintel_final-3d2a1287.pth`` (SHA-256 prefix 3d2a1287).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .flow import flow_warp


class BasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            nn.Conv2d(8, 32, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 32, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 16, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 2, 7, padding=3),
        )

    def forward(self, x):
        return self.basic_module(x)


class SpyNet(nn.Module):
    """Estimates flow ``f`` such that ``flow_warp(supp, f) ~= ref``."""

    def __init__(self, load_path=None):
        super().__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            state = torch.load(load_path, map_location="cpu", weights_only=True)
            self.load_state_dict(state.get("params", state))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def process(self, ref, supp):
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref.insert(0, F.avg_pool2d(ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(supp[0], kernel_size=2, stride=2, count_include_pad=False))

        flow = ref[0].new_zeros(ref[0].size(0), 2, ref[0].size(2) // 2, ref[0].size(3) // 2)
        for level in range(len(ref)):
            up = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            if up.size(2) != ref[level].size(2):
                up = F.pad(up, [0, 0, 0, 1], mode="replicate")
            if up.size(3) != ref[level].size(3):
                up = F.pad(up, [0, 1, 0, 0], mode="replicate")
            warped = flow_warp(supp[level], up.permute(0, 2, 3, 1), padding_mode="border")
            flow = self.basic_module[level](torch.cat([ref[level], warped, up], 1)) + up
        return flow

    def forward(self, ref, supp):
        assert ref.size() == supp.size()
        h, w = ref.shape[-2:]
        # Multiple of 32 for the 6-level pyramid; at least 64 so the coarsest level is non-empty
        # (sizes >= 64 behave exactly as in BasicSR).
        h_up, w_up = max(math.ceil(h / 32) * 32, 64), max(math.ceil(w / 32) * 32, 64)
        ref = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(self.process(ref, supp), size=(h, w), mode="bilinear", align_corners=False)
        flow[:, 0] *= w / w_up
        flow[:, 1] *= h / h_up
        return flow
