"""Run the official pretrained BasicVSR (Chan et al., CVPR 2021) and save its outputs as PNGs.

Pure-PyTorch port of MMEditing's ``BasicVSRNet`` (Apache-2.0) that loads the official
REDS-trained weights, so baselines can be scored with ``scripts/evaluate_outputs.py`` under
exactly the same protocol as Fork-Net. On REDS4 it reproduces the published 31.42 dB / 0.8909.

    python scripts/baselines/basicvsr.py --lr-root data/REDS/LR_bicubic_X4 --clips 000 011 015 020 --out outputs/basicvsr/reds4
"""

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from forknet.archs.flow import flow_warp  # noqa: E402
from forknet.archs.spynet import SpyNet  # noqa: E402
from forknet.data import VideoTestDataset, tensor_to_uint8, write_image  # noqa: E402
from forknet.inference import upscale_sequence  # noqa: E402

WEIGHTS_URL = "https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth"


class ResidualBlockNoBN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1, 1)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x)))


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, cin, c=64, n=30):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(cin, c, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
                                  nn.Sequential(*[ResidualBlockNoBN(c) for _ in range(n)]))

    def forward(self, x):
        return self.main(x)


class PixelShufflePack(nn.Module):
    def __init__(self, cin, cout, scale):
        super().__init__()
        self.scale = scale
        self.upsample_conv = nn.Conv2d(cin, cout * scale * scale, 3, 1, 1)

    def forward(self, x):
        return F.pixel_shuffle(self.upsample_conv(x), self.scale)


class BasicVSR(nn.Module):
    def __init__(self, c=64, n=30):
        super().__init__()
        self.c = c
        self.spynet = SpyNet()
        self.backward_resblocks = ResidualBlocksWithInputConv(c + 3, c, n)
        self.forward_resblocks = ResidualBlocksWithInputConv(c + 3, c, n)
        self.fusion = nn.Conv2d(c * 2, c, 1)
        self.upsample1 = PixelShufflePack(c, c, 2)
        self.upsample2 = PixelShufflePack(c, 64, 2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, lrs):
        n, t, c, h, w = lrs.shape
        x1, x2 = lrs[:, :-1].reshape(-1, c, h, w), lrs[:, 1:].reshape(-1, c, h, w)
        flows_backward = self.spynet(x1, x2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(x2, x1).view(n, t - 1, 2, h, w)
        feats = []
        prop = lrs.new_zeros(n, self.c, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                prop = flow_warp(prop, flows_backward[:, i].permute(0, 2, 3, 1))
            prop = self.backward_resblocks(torch.cat([lrs[:, i], prop], 1))
            feats.append(prop)
        feats = feats[::-1]
        prop = torch.zeros_like(prop)
        outs = []
        for i in range(t):
            if i > 0:
                prop = flow_warp(prop, flows_forward[:, i - 1].permute(0, 2, 3, 1))
            prop = self.forward_resblocks(torch.cat([lrs[:, i], prop], 1))
            out = self.lrelu(self.fusion(torch.cat([feats[i], prop], 1)))
            out = self.lrelu(self.upsample1(out))
            out = self.lrelu(self.upsample2(out))
            out = self.conv_last(self.lrelu(self.conv_hr(out)))
            outs.append(out + F.interpolate(lrs[:, i], scale_factor=4, mode="bilinear", align_corners=False))
        return torch.stack(outs, 1)


def load_official(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return remap_official(sd.get("state_dict", sd))


def remap_official(sd):
    """MMEditing BasicVSRNet keys -> this port's keys."""
    out = {}
    for k, v in sd.items():
        k = k.removeprefix("generator.")
        parts = k.split(".")
        # mmcv ConvModule: spynet.basic_module.<i>.basic_module.<j>.conv.<w|b> -> Sequential index 2j
        if parts[:2] == ["spynet", "basic_module"] and len(parts) == 7 and parts[5] == "conv":
            k = ".".join(parts[:4] + [str(2 * int(parts[4])), parts[6]])
        out[k] = v
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lr-root", required=True)
    p.add_argument("--clips", nargs="*")
    p.add_argument("--split-file")
    p.add_argument("--split")
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default=str(ROOT / "weights" / "basicvsr_reds4_20120409-0e599677.pth"))
    p.add_argument("--chunk-size", type=int, default=None, help="frames per forward pass (default: whole clip, as published)")
    p.add_argument("--max-frames", type=int, help="only the first N frames of each clip")
    args = p.parse_args()

    if not Path(args.weights).exists():
        Path(args.weights).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WEIGHTS_URL, args.weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicVSR()
    model.load_state_dict(load_official(args.weights), strict=True)
    model.to(device).eval()
    ds = VideoTestDataset(args.lr_root, None, args.clips, None, args.split_file, args.split, args.max_frames)
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            sr = upscale_sequence(model, item["lr"][None].to(device), args.chunk_size)[0]
            for t, name in enumerate(item["frames"]):
                write_image(Path(args.out) / item["clip"] / f"{name}.png", tensor_to_uint8(sr[t]))
            print(f"{item['clip']}: {sr.shape[0]} frames", flush=True)


if __name__ == "__main__":
    main()
