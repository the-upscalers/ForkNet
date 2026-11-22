"""Run the official pretrained RealBasicVSR (Chan et al., CVPR 2022) and save its outputs as PNGs.

Pure-PyTorch port of MMagic's ``RealBasicVSRNet`` (Apache-2.0): an image-cleaning module (20 residual blocks, applied
up to three times while the mean residue exceeds ``--refine-thres`` / 255) followed by BasicVSR with 20 blocks per
branch. Loads the official REDS-trained weights (EMA generator when present), so the baseline is scored with
``scripts/evaluate_outputs.py`` / ``scripts/eval_noref.py`` under the same protocol as Fork-Net.

    python scripts/baselines/realbasicvsr.py --lr-root data/VideoLQ --out outputs/realbasicvsr/videolq --chunk-size 10
"""

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from basicvsr import BasicVSR, ResidualBlocksWithInputConv, remap_official  # noqa: E402
from torch import nn  # noqa: E402

from forknet.data import VideoTestDataset, tensor_to_uint8, write_image  # noqa: E402
from forknet.inference import upscale_sequence  # noqa: E402

WEIGHTS_URL = ("https://download.openmmlab.com/mmediting/restorers/real_basicvsr/"
               "realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth")


class RealBasicVSR(nn.Module):
    def __init__(self, c=64, n_prop=20, n_clean=20, refine_thres=5.0):
        super().__init__()
        self.refine_thres = refine_thres / 255.0
        self.image_cleaning = nn.Sequential(ResidualBlocksWithInputConv(3, c, n_clean), nn.Conv2d(c, 3, 3, 1, 1))
        self.basicvsr = BasicVSR(c, n_prop)

    def forward(self, lrs):
        n, t, c, h, w = lrs.shape
        for _ in range(3):  # at most three cleaning passes, stopped once the residue is small
            x = lrs.reshape(-1, c, h, w)
            residues = self.image_cleaning(x)
            lrs = (x + residues).view(n, t, c, h, w)
            if torch.mean(torch.abs(residues)) < self.refine_thres:
                break
        return self.basicvsr(lrs)


def load_official_real(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    prefix = "generator_ema." if any(k.startswith("generator_ema.") for k in sd) else "generator."
    gen = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    out = {k: v for k, v in gen.items() if not k.startswith("basicvsr.")}
    inner = remap_official({k[len("basicvsr."):]: v for k, v in gen.items() if k.startswith("basicvsr.")})
    out.update({f"basicvsr.{k}": v for k, v in inner.items()})
    return out


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
    p.add_argument("--refine-thres", type=float, default=5.0, help="dynamic cleaning threshold (x/255), as released")
    args = p.parse_args()

    if not Path(args.weights).exists():
        Path(args.weights).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WEIGHTS_URL, args.weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RealBasicVSR(refine_thres=args.refine_thres)
    model.load_state_dict(load_official_real(args.weights), strict=True)
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
