"""Robustness of VSR models to degradations other than the bicubic one they were trained on.

For every level in configs/robustness.yaml, LR clips are generated once from the HR test clips
(deterministic seed per clip) and cached; every method is then evaluated with the standard protocol.

    python scripts/degradation_sweep.py --hr-root data/lp/test/sharp --cache cache/lp_sweep \
        --checkpoint weights/forknet_reds.pth weights/forknet_reds_corrected.pth --name ForkNet ForkNet-fixed \
        --bicubic --basicvsr --out results/robustness_lp.json
"""

import argparse
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

import torch  # noqa: E402

from forknet.data import list_frames, read_image, resolve_clips, write_image, VideoTestDataset  # noqa: E402
from forknet.data.degradations import degrade_clip  # noqa: E402
from forknet.evaluation import evaluate  # noqa: E402
from forknet.utils import build_model_from_checkpoint, get_logger, load_config  # noqa: E402


def build_level(hr_root, clips, out_dir, spec, scale, log):
    for clip in clips:
        dst = out_dir / clip
        frames = list_frames(Path(hr_root) / clip)
        if dst.exists() and len(list_frames(dst)) == len(frames):
            continue
        hr = [read_image(p) for p in frames]
        hr = [f[: f.shape[0] - f.shape[0] % scale, : f.shape[1] - f.shape[1] % scale] for f in hr]
        for p, lr in zip(frames, degrade_clip(hr, scale, seed=zlib.crc32(clip.encode()), **spec)):
            write_image(dst / f"{p.stem}.png", lr)
    log.info(f"LR set ready: {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hr-root", required=True)
    p.add_argument("--clips", nargs="*")
    p.add_argument("--split-file")
    p.add_argument("--split")
    p.add_argument("--levels", default=str(ROOT / "configs" / "robustness.yaml"))
    p.add_argument("--only", nargs="*", help="subset of level names")
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", nargs="*", default=[])
    p.add_argument("--name", nargs="*", default=[])
    p.add_argument("--bicubic", action="store_true")
    p.add_argument("--basicvsr", action="store_true", help="also run the official BasicVSR (REDS weights)")
    p.add_argument("--chunk-size", type=int, default=15)
    p.add_argument("--amp", action="store_true", help="fp16 inference for Fork-Net (needed for 720p outputs on a 16 GB GPU)")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    log = get_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clips = resolve_clips(args.hr_root, args.clips, None, args.split_file, args.split)
    levels = load_config(args.levels)["levels"]
    if args.only:
        levels = {k: v for k, v in levels.items() if k in args.only}

    methods = {}
    names = args.name or [Path(c).stem for c in args.checkpoint]
    for name, ckpt in zip(names, args.checkpoint):
        methods[name] = build_model_from_checkpoint(ckpt)[0].to(device).eval()
    if args.basicvsr:
        from basicvsr import BasicVSR, WEIGHTS_URL, load_official
        import urllib.request

        w = ROOT / "weights" / "basicvsr_reds4_20120409-0e599677.pth"
        if not w.exists():
            urllib.request.urlretrieve(WEIGHTS_URL, w)
        m = BasicVSR()
        m.load_state_dict(load_official(w), strict=True)
        methods["BasicVSR"] = m.to(device).eval()
    if args.bicubic:
        methods["Bicubic"] = None

    results = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    for level, spec in levels.items():
        lr_dir = Path(args.cache) / level
        build_level(args.hr_root, clips, lr_dir, spec or {}, 4, log)
        ds = VideoTestDataset(lr_dir, args.hr_root, clips=clips, max_frames=args.max_frames)
        for name, model in methods.items():
            if name in results.get(level, {}):
                continue
            res = evaluate(model, ds, device, args.chunk_size, amp=args.amp)
            results.setdefault(level, {})[name] = {**res["average"], "clips": res["clips"]}
            log.info(f"{level:14s} {name:16s} PSNR {res['average']['psnr_rgb']:.3f}  SSIM {res['average']['ssim_rgb']:.4f}")
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(results, indent=2))

    header = "| Level | " + " | ".join(methods) + " |"
    print("\n" + header + "\n|" + "---|" * (len(methods) + 1))
    for level in levels:
        cells = [f"{results[level][m]['psnr_rgb']:.2f} / {results[level][m]['ssim_rgb']:.4f}" for m in methods]
        print(f"| {level} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
