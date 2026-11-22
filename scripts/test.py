"""Evaluate Fork-Net (or bicubic) on a test set and write metrics to JSON.

    # standard protocol, REDS4, released/legacy checkpoint
    python scripts/test.py --config configs/test_reds4.yaml --checkpoint path/to/ckpt.pth
    # also the original (legacy) validation protocol
    python scripts/test.py --config configs/test_reds4.yaml --checkpoint ckpt.pth --protocol both
    # bicubic baseline
    python scripts/test.py --config configs/test_reds4.yaml --bicubic
"""

import argparse
import os

# Full-frame REDS inference fragments the CUDA cache (OOM with GBs reserved but unused); must precede `import torch`.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from forknet.data import VideoTestDataset  # noqa: E402
from forknet.evaluation import evaluate, evaluate_legacy  # noqa: E402
from forknet.utils import build_model_from_checkpoint, environment_info, get_logger, load_config, param_counts  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--bicubic", nargs="?", const="matlab", choices=("matlab", "torch"),
                       help="evaluate bicubic upsampling instead of a model (default kernel: matlab)")
    p.add_argument("--protocol", choices=("standard", "legacy", "both"), default="standard")
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--out", help="output directory (default results/<config name>/<checkpoint stem>)")
    p.add_argument("--legacy", choices=("auto", "on", "off"), default="auto",
                   help="force the legacy_* propagation flags; 'auto' uses the checkpoint's own config")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    tcfg = cfg.get("test", {})
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    tag = f"bicubic_{args.bicubic}" if args.bicubic else Path(args.checkpoint).stem
    out_dir = Path(args.out or Path("results") / cfg["name"] / tag)
    log = get_logger(log_file=out_dir / "test.log")

    model, meta = None, {}
    if not args.bicubic:
        overrides = None
        if args.legacy != "auto":
            v = args.legacy == "on"
            overrides = {"legacy_flow_direction": v, "legacy_backward_index": v, "legacy_flow_scaling": v}
        model, meta = build_model_from_checkpoint(args.checkpoint, overrides)
        model.to(device).eval()
        log.info(f"Loaded {args.checkpoint} ({meta}); config {model.cfg}")

    dataset = VideoTestDataset(**cfg["data"]["test"])
    report = {"config": cfg, "checkpoint": args.checkpoint, "checkpoint_meta": meta, "environment": environment_info()}
    if model is not None:
        report["model_config"] = model.cfg.to_dict()
        report["params"] = param_counts(model)

    chunk, overlap, amp = tcfg.get("chunk_size", 15), tcfg.get("overlap", 0), tcfg.get("amp", False)
    if args.protocol in ("standard", "both"):
        res = evaluate(model, dataset, device, chunk, overlap, amp, tcfg.get("crop_border", 0),
                       out_dir / "images" if args.save_images else None, log, bicubic=args.bicubic or "matlab")
        report["standard"] = res
        a = res["average"]
        log.info(f"[standard] PSNR-RGB {a['psnr_rgb']:.4f}  SSIM-RGB {a['ssim_rgb']:.4f}  "
                 f"PSNR-Y {a['psnr_y']:.4f}  SSIM-Y {a['ssim_y']:.4f}")
    if args.protocol in ("legacy", "both"):
        res = evaluate_legacy(model, dataset, device, tcfg.get("legacy_chunk_size", 15), amp, log)
        report["legacy"] = res
        log.info(f"[legacy] PSNR {res['average']['psnr']:.4f}  SSIM {res['average']['ssim']:.4f} over {res['num_chunks']} chunks")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
