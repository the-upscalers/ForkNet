"""Evaluate "TheUpscalers", the team's earlier sliding-window model (pixel-level-vsr `main`,
models/u_net_architecture/{forknet.py,tempo_fuser.py}, Feb 2025; checkpoint in the Kaggle dataset
joelsathiyendra/fork-net-200-epochs-run-pod). Ported verbatim, quirks included (e.g. the T-2 neighbour is warped
from frame t-1), so the checkpoint loads with strict=True.

Per frame t: SpyNet-warped neighbours t-2..t+2, a two-stream U-Net (current frame / neighbours, neighbour
features averaged per side), two-step pixel shuffle, plus bilinear x4 of the input. No recurrent propagation.

    python scripts/baselines/the_upscalers.py --checkpoint best_checkpoint.pth --config configs/test_lp.yaml \\
        --protocol both --out results/the_upscalers_lp --set data.test.lr_root=... data.test.hr_root=...
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from forknet.archs.flow import flow_warp  # noqa: E402
from forknet.archs.spynet import SpyNet  # noqa: E402
from forknet.data import VideoTestDataset  # noqa: E402
from forknet.evaluation import evaluate, evaluate_legacy  # noqa: E402
from forknet.utils import environment_info, get_logger, load_config  # noqa: E402


class UnetBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.cin, self.cout = cin, cout
        self.conv_adapter = nn.Conv2d(cin, cout, 1)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, inp):
        x = self.conv1(inp)
        if self.cin == self.cout:
            x = x + inp + self.conv_adapter(inp)
        return self.act(x)


def layers(cin, cout, n=2):
    return nn.Sequential(UnetBlock(cin, cout), *[UnetBlock(cout, cout) for _ in range(n - 1)])


class TempoFuser(nn.Module):
    def __init__(self, nf=64, n=2):  # n = 2 * layer_multiplier blocks per stage (2 in the code, 6 in the RunPod run)
        super().__init__()
        for s in ("cur", "temp"):
            setattr(self, f"conv_down_1_{s}", layers(3, nf, n))
            setattr(self, f"conv_down_2_{s}", layers(nf, nf, n))
            setattr(self, f"conv_down_3_{s}", layers(nf, nf, n))
        self.conv_down_4_cur = layers(nf, nf * 2, n)
        self.conv_down_4_temp = layers(nf, nf, n)
        self.conv_up_1, self.conv_up_2, self.conv_up_3 = layers(nf, nf, n), layers(nf, nf, n), layers(nf, nf, n)
        self.pixel_shuffle, self.maxpool = nn.PixelShuffle(2), nn.MaxPool2d(2)
        self.refine_conv_1 = nn.Conv2d(nf, 64, 3, padding=1)
        self.refine_conv_2 = nn.Conv2d(16, 3, 3, padding=1)

    def _pipe(self, x, s):
        o1 = getattr(self, f"conv_down_1_{s}")(x)
        o2 = getattr(self, f"conv_down_2_{s}")(self.maxpool(o1))
        o3 = getattr(self, f"conv_down_3_{s}")(self.maxpool(o2))
        o4 = getattr(self, f"conv_down_4_{s}")(self.maxpool(o3))
        return o1, o2, o3, o4

    @staticmethod
    def _stack(*frames):  # channel interleaving exactly as the original stack_frames
        shape = frames[0].shape
        return torch.stack(frames, dim=2).view(shape[0], shape[1] * len(frames), *shape[2:])

    def forward(self, cur, p1, p2, n1, n2):
        c = self._pipe(cur, "cur")
        p = [(a + b) / 2 for a, b in zip(self._pipe(p1, "temp"), self._pipe(p2, "temp"))]
        n = [(a + b) / 2 for a, b in zip(self._pipe(n1, "temp"), self._pipe(n2, "temp"))]
        x = self.conv_up_1(self.pixel_shuffle(torch.cat([c[3], self._stack(p[3], n[3])], 1)))
        x = self.conv_up_2(self.pixel_shuffle(self._stack(p[2], c[2], x, n[2])))
        x = self.conv_up_3(self.pixel_shuffle(self._stack(p[1], c[1], x, n[1])))
        x = self._stack(p[0], c[0], x, n[0])
        x = self.refine_conv_1(self.pixel_shuffle(x))
        return self.refine_conv_2(self.pixel_shuffle(x))


class TheUpscalers(nn.Module):
    def __init__(self, blocks=2):
        super().__init__()
        self.spynet = SpyNet()
        self.tempo_fuser = TempoFuser(n=blocks)

    def _forward(self, lr):
        n, t, c, h, w = lr.shape
        a, b = lr[:, :-1].reshape(-1, c, h, w), lr[:, 1:].reshape(-1, c, h, w)
        ff1, bf1 = self.spynet(a, b).view(n, t - 1, 2, h, w), self.spynet(b, a).view(n, t - 1, 2, h, w)
        a2, b2 = lr[:, :-2].reshape(-1, c, h, w), lr[:, 2:].reshape(-1, c, h, w)
        ff2, bf2 = self.spynet(a2, b2).view(n, t - 2, 2, h, w), self.spynet(b2, a2).view(n, t - 2, 2, h, w)

        def warp(i, flow):
            return flow_warp(lr[:, i], flow.permute(0, 2, 3, 1))

        outs = []
        for i in range(t):
            cur = lr[:, i]
            if i == 0:
                p2 = p1 = cur
                n1, n2 = warp(i + 1, bf1[:, i]), warp(i + 2, bf2[:, i])
            elif i == 1:
                p2, p1 = warp(i - 1, ff1[:, i - 1]), cur
                n1, n2 = warp(i + 1, bf1[:, i]), warp(i + 2, bf2[:, i])
            elif i == t - 2:
                p2, p1 = warp(i - 1, ff2[:, i - 2]), warp(i - 1, ff1[:, i - 1])
                n1, n2 = cur, warp(i + 1, bf1[:, i])
            elif i == t - 1:
                p2, p1 = warp(i - 1, ff2[:, i - 2]), warp(i - 1, ff1[:, i - 1])
                n1 = n2 = cur
            else:
                p2, p1 = warp(i - 1, ff2[:, i - 2]), warp(i - 1, ff1[:, i - 1])
                n1, n2 = warp(i + 1, bf1[:, i]), warp(i + 2, bf2[:, i])
            sr = self.tempo_fuser(cur, p1, p2, n1, n2)
            outs.append(sr + F.interpolate(cur, scale_factor=4.0, mode="bilinear", align_corners=False))
        return torch.stack(outs, 1)

    def forward(self, lr):
        """(N, T, 3, h, w) -> (N, T, 3, 4h, 4w). The original needs T >= 4 and h, w divisible by 8 (three
        max-pools); shorter clips are padded by repeating the last frame and frames are reflect-padded, then cropped."""
        n, t, c, h, w = lr.shape
        x = torch.cat([lr, lr[:, -1:].expand(n, 4 - t, c, h, w)], 1) if t < 4 else lr
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            x = F.pad(x.flatten(0, 1), (0, pw, 0, ph), mode="reflect").view(n, x.size(1), c, h + ph, w + pw)
        return self._forward(x)[:, :t, :, : 4 * h, : 4 * w]


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("module."): v for k, v in ckpt.get("model_state_dict", ckpt).items()}
    blocks = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("tempo_fuser.conv_down_1_cur."))
    model = TheUpscalers(blocks)
    model.load_state_dict(sd, strict=True)
    meta = {k: (float(v) if torch.is_tensor(v) else v) for k, v in ckpt.items() if k in ("epoch", "psnr", "ssim", "loss")}
    return model, {**meta, "blocks_per_stage": blocks}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True, help="a test config, e.g. configs/test_lp.yaml")
    p.add_argument("--protocol", choices=("standard", "legacy", "both"), default="both")
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    tcfg = cfg.get("test", {})
    out = Path(args.out)
    log = get_logger(log_file=out / "test.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load(args.checkpoint)
    model.to(device).eval()
    log.info(f"Loaded {args.checkpoint} {meta}: {sum(q.numel() for q in model.parameters()) / 1e6:.3f}M params")
    dataset = VideoTestDataset(**cfg["data"]["test"])
    report = {"model": "TheUpscalers", "checkpoint": args.checkpoint, "checkpoint_meta": meta, "config": cfg,
              "environment": environment_info()}
    chunk, amp = tcfg.get("chunk_size", 15), tcfg.get("amp", False)
    if args.protocol in ("standard", "both"):
        res = evaluate(model, dataset, device, chunk, tcfg.get("overlap", 0), amp, tcfg.get("crop_border", 0),
                       out / "images" if args.save_images else None, log)
        report["standard"] = res
        a = res["average"]
        log.info(f"[standard] PSNR-RGB {a['psnr_rgb']:.4f}  SSIM-RGB {a['ssim_rgb']:.4f}  PSNR-Y {a['psnr_y']:.4f}  SSIM-Y {a['ssim_y']:.4f}")
    if args.protocol in ("legacy", "both"):
        res = evaluate_legacy(model, dataset, device, tcfg.get("legacy_chunk_size", 15), amp, log)
        report["legacy"] = res
        log.info(f"[legacy] PSNR {res['average']['psnr']:.4f}  SSIM {res['average']['ssim']:.4f} over {res['num_chunks']} chunks")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
    log.info(f"Wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
