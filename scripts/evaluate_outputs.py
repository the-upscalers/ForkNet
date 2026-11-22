"""Score pre-computed SR frames of any method against ground truth with Fork-Net's metric code.

Use this for every baseline (BasicVSR, IconVSR, BasicVSR++, MIA-VSR, ...) so that all numbers
in a table come from one implementation and one protocol.

    python scripts/evaluate_outputs.py --sr outputs/basicvsr_pp/lp_test --gt data/lp/HR \\
        --split-file data/lp/splits.json --split test --name BasicVSR++ --out results/lp/basicvsr_pp.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forknet.data import resolve_clips  # noqa: E402
from forknet.evaluation import evaluate_folders  # noqa: E402
from forknet.utils import get_logger  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sr", required=True, help="root with <clip>/<frame>.png SR outputs")
    p.add_argument("--gt", required=True, help="root with <clip>/<frame>.png ground truth")
    p.add_argument("--clips", nargs="*")
    p.add_argument("--split-file")
    p.add_argument("--split")
    p.add_argument("--crop-border", type=int, default=0)
    p.add_argument("--extra", nargs="*", default=[], choices=("lpips", "tof"), help="additional metrics")
    p.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    p.add_argument("--name", help="method name stored in the report")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    log = get_logger()
    clips = resolve_clips(args.gt, args.clips, None, args.split_file, args.split)
    res = evaluate_folders(args.sr, args.gt, clips, args.crop_border, log, tuple(args.extra), args.device)
    a = res["average"]
    log.info(f"{args.name or args.sr}: PSNR-RGB {a['psnr_rgb']:.4f}  SSIM-RGB {a['ssim_rgb']:.4f}  "
             f"PSNR-Y {a['psnr_y']:.4f}  SSIM-Y {a['ssim_y']:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"method": args.name, "sr": args.sr, "gt": args.gt, "standard": res}, f, indent=2)


if __name__ == "__main__":
    main()
