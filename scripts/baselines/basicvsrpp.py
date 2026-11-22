"""Run the official pretrained BasicVSR++ (Chan et al., CVPR 2022) and save its outputs as PNGs.

Pure-PyTorch port of MMEditing's ``BasicVSRPlusPlus`` (Apache-2.0): second-order grid propagation with flow-guided
deformable alignment (modulated deformable convolution via ``torchvision.ops.deform_conv2d``, the offset layout of
mmcv's DCNv2). Loads the official REDS-trained weights; the port is validated by reproducing the published REDS4 score
(32.39 dB, RGB) with ``scripts/evaluate_outputs.py``.

    python scripts/baselines/basicvsrpp.py --lr-root data/REDS/LR_bicubic_X4 --clips 000 011 015 020 --out outputs/basicvsrpp/reds4
"""

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from basicvsr import PixelShufflePack, ResidualBlocksWithInputConv, remap_official  # noqa: E402
from torch import nn  # noqa: E402
from torchvision.ops import deform_conv2d  # noqa: E402

from forknet.archs.flow import flow_warp  # noqa: E402
from forknet.archs.spynet import SpyNet  # noqa: E402
from forknet.data import VideoTestDataset, tensor_to_uint8, write_image  # noqa: E402
from forknet.inference import upscale_sequence  # noqa: E402

WEIGHTS_URL = ("https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/"
               "basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth")


class SecondOrderDeformableAlignment(nn.Module):
    def __init__(self, cin, cout, deform_groups=16, max_residue_magnitude=10):
        super().__init__()
        self.deform_groups = deform_groups
        self.max_residue_magnitude = max_residue_magnitude
        self.weight = nn.Parameter(torch.zeros(cout, cin, 3, 3))
        self.bias = nn.Parameter(torch.zeros(cout))
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * cout + 4, cout, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(cout, 27 * deform_groups, 3, 1, 1),
        )

    def forward(self, x, extra_feat, flow_1, flow_2):
        out = self.conv_offset(torch.cat([extra_feat, flow_1, flow_2], dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)
        return deform_conv2d(x, offset, self.weight, self.bias, padding=(1, 1), mask=torch.sigmoid(mask))


class BasicVSRPlusPlus(nn.Module):
    MODULES = ("backward_1", "forward_1", "backward_2", "forward_2")

    def __init__(self, c=64, num_blocks=7):
        super().__init__()
        self.c = c
        self.spynet = SpyNet()
        self.feat_extract = ResidualBlocksWithInputConv(3, c, 5)
        self.deform_align = nn.ModuleDict({m: SecondOrderDeformableAlignment(2 * c, c) for m in self.MODULES})
        self.backbone = nn.ModuleDict({m: ResidualBlocksWithInputConv((2 + i) * c, c, num_blocks) for i, m in enumerate(self.MODULES)})
        self.reconstruction = ResidualBlocksWithInputConv(5 * c, c, 5)
        self.upsample1 = PixelShufflePack(c, c, 2)
        self.upsample2 = PixelShufflePack(c, 64, 2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def propagate(self, feats, flows, name):
        n, t, _, h, w = flows.shape
        frame_idx = list(range(t + 1))
        flow_idx = list(range(-1, t))
        mapping_idx = list(range(len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]
        if name.startswith("backward"):
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx
        feat_prop = flows.new_zeros(n, self.c, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[idx]]
            if i > 0:
                flow_n1 = flows[:, flow_idx[i]]
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                feat_n2, flow_n2, cond_n2 = torch.zeros_like(feat_prop), torch.zeros_like(flow_n1), torch.zeros_like(cond_n1)
                if i > 1:
                    feat_n2 = feats[name][-2]
                    flow_n2 = flows[:, flow_idx[i - 1]]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = self.deform_align[name](torch.cat([feat_prop, feat_n2], dim=1), cond, flow_n1, flow_n2)
            feat = [feat_current] + [feats[k][idx] for k in feats if k not in ("spatial", name)] + [feat_prop]
            feat_prop = feat_prop + self.backbone[name](torch.cat(feat, dim=1))
            feats[name].append(feat_prop)
        if name.startswith("backward"):
            feats[name] = feats[name][::-1]
        return feats

    def forward(self, lrs):
        n, t, c, h, w = lrs.shape
        x1, x2 = lrs[:, :-1].reshape(-1, c, h, w), lrs[:, 1:].reshape(-1, c, h, w)
        flows_backward = self.spynet(x1, x2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(x2, x1).view(n, t - 1, 2, h, w)
        f = self.feat_extract(lrs.reshape(-1, c, h, w)).view(n, t, -1, h, w)
        feats = {"spatial": [f[:, i] for i in range(t)]}
        for name in self.MODULES:
            feats[name] = []
            feats = self.propagate(feats, flows_backward if name.startswith("backward") else flows_forward, name)
        outs = []
        for i in range(t):
            hr = torch.cat([feats["spatial"][i]] + [feats[k][i] for k in self.MODULES], dim=1)
            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.conv_last(self.lrelu(self.conv_hr(hr)))
            outs.append(hr + F.interpolate(lrs[:, i], scale_factor=4, mode="bilinear", align_corners=False))
        return torch.stack(outs, 1)


def load_official_pp(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    gen = {k.removeprefix("generator."): v for k, v in sd.items()
           if not k.startswith(("discriminator", "generator_ema")) and k != "step_counter"}  # step_counter: training buffer
    return remap_official(gen)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lr-root", required=True)
    p.add_argument("--clips", nargs="*")
    p.add_argument("--split-file")
    p.add_argument("--split")
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default=str(ROOT / "weights" / Path(WEIGHTS_URL).name))
    p.add_argument("--chunk-size", type=int, default=None, help="frames per forward pass (default: whole clip)")
    p.add_argument("--max-frames", type=int, help="only the first N frames of each clip")
    args = p.parse_args()

    if not Path(args.weights).exists():
        Path(args.weights).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WEIGHTS_URL, args.weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicVSRPlusPlus()
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    # Our own fine-tunes (train_basicvsrpp.py) store {"params": ...}; official MMEditing files need remapping.
    model.load_state_dict(ckpt["params"] if "params" in ckpt else load_official_pp(args.weights), strict=True)
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
