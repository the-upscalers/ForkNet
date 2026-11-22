"""Run RVRT (Liang et al., NeurIPS 2022) or MIA-VSR (Zhou et al., CVPR 2024) from their official repositories with the
official REDS-trained weights, and save PNGs in this repo's layout (``<out>/<clip>/<frame>.png``) so they are scored by
``scripts/evaluate_outputs.py`` / ``scripts/lp/ocr_eval.py`` exactly like Fork-Net.

Frames are padded (replicate) to multiples of 8 (both models use 8x8 attention windows) and, for RVRT, to an even
number (clip size 2); outputs are cropped back. Long clips are processed in chunks of ``--chunk-size`` frames (GPU
memory); the published numbers use longer windows, so REDS4 is re-scored here as the port check.

    git clone https://github.com/JingyunLiang/RVRT  &&  python scripts/baselines/sota_repo.py --model rvrt --repo RVRT \\
        --weights 001_RVRT_videosr_bi_REDS_30frames.pth --lr-root data/REDS4/LR --out outputs/rvrt/reds4 --chunk-size 30
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from forknet.data import VideoTestDataset, tensor_to_uint8, write_image  # noqa: E402
from forknet.inference import upscale_sequence  # noqa: E402


def build(model, repo, weights):
    sys.path.insert(0, str(Path(repo).resolve()))
    if model == "rvrt":
        from models.network_rvrt import RVRT

        net = RVRT(upscale=4, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                   depths=[2, 2, 2], embed_dims=[144, 144, 144], num_heads=[6, 6, 6], inputconv_groups=[1, 1, 1, 1, 1, 1],
                   deformable_groups=12, attention_heads=12, attention_window=[3, 3], cpu_cache_length=100)
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        net.load_state_dict(sd.get("params", sd), strict=True)
        return net, 2
    if model == "miavsr":
        # BasicSR does not build on recent Python: a minimal stand-in with the same modules and parameter names,
        # and two imports of the repo removed from recent torch (torch._six, tensorboardX) patched out.
        sys.path.insert(0, str(Path(__file__).resolve().parent / "basicsr_shim"))
        fn = Path(repo) / "archs" / "functions.py"
        fn.write_text(fn.read_text().replace("from torch._six import inf", "from math import inf")
                      .replace("from tensorboardX import SummaryWriter", "SummaryWriter = None"))
        from archs.mia_vsr_arch import MIAVSR

        net = MIAVSR(mid_channels=64, embed_dim=120, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6], window_size=[3, 8, 8],
                     num_frames=3, cpu_cache_length=100, is_low_res_input=True, use_mask=True,
                     spynet_path=str(Path(repo) / "experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth"))
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        missing, unexpected = net.load_state_dict(sd.get("params", sd), strict=False)  # as in the official script
        print(f"MIA-VSR weights: {len(missing)} missing, {len(unexpected)} unexpected keys", flush=True)
        return net, 1
    raise ValueError(model)


class Padded(torch.nn.Module):
    """Pads (n, t, c, h, w) inputs to at least 64 px and multiples of 8 spatially (tiny plate crops, attention
    windows) and to multiples of ``t_mult`` temporally; crops the output."""

    def __init__(self, net, t_mult):
        super().__init__()
        self.net, self.t_mult = net, t_mult

    def forward(self, x):
        n, t, c, h, w = x.shape
        ph, pw, pt = max(64 - h, 0) + (-max(h, 64)) % 8, max(64 - w, 0) + (-max(w, 64)) % 8, (-t) % self.t_mult  # >= 64 px
        if ph or pw:
            x = F.pad(x.flatten(0, 1), (0, pw, 0, ph), mode="replicate").view(n, t, c, h + ph, w + pw)
        if pt:
            x = torch.cat([x, x[:, -1:].expand(n, pt, c, h + ph, w + pw)], 1)
        y = self.net(x)
        y = y[0] if isinstance(y, (tuple, list)) else y  # MIA-VSR returns (output, masks)
        return y[:, :t, :, : 4 * h, : 4 * w]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["rvrt", "miavsr"])
    p.add_argument("--repo", required=True, help="clone of the official repository")
    p.add_argument("--weights", required=True)
    p.add_argument("--lr-root", required=True)
    p.add_argument("--clips", nargs="*")
    p.add_argument("--split-file")
    p.add_argument("--split")
    p.add_argument("--out", required=True)
    p.add_argument("--chunk-size", type=int, default=30)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--tile", type=int, help="process frames in overlapping TILE x TILE LR tiles (GPU memory on large inputs)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, t_mult = build(args.model, args.repo, args.weights)
    model = Padded(net, t_mult).to(device).eval()
    ds = VideoTestDataset(args.lr_root, None, args.clips, None, args.split_file, args.split, args.max_frames)
    with torch.no_grad(), torch.autocast("cuda", enabled=args.amp and device.type == "cuda"):
        for i in range(len(ds)):
            item = ds[i]
            sr = upscale_sequence(model, item["lr"][None].to(device), args.chunk_size, tile=args.tile)[0].float()
            for t, name in enumerate(item["frames"]):
                write_image(Path(args.out) / item["clip"] / f"{name}.png", tensor_to_uint8(sr[t]))
            print(f"{item['clip']}: {sr.shape[0]} frames", flush=True)


if __name__ == "__main__":
    main()
