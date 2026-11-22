"""Parameters, FLOPs/MACs and runtime for the main model and the ablation variants.

    python scripts/complexity.py --out results/complexity.json            # all paper configs
    python scripts/complexity.py --config configs/ablation/first_order.yaml
    python scripts/complexity.py --runtime --device cuda                  # add GPU timing
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from forknet import ForkNet, ForkNetConfig  # noqa: E402
from forknet.utils import flops_per_frame, load_config, param_counts, runtime_per_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = {
    "Fork-Net (2nd order + BSConv + ConvNeXt)": "configs/train_reds.yaml",
    "1st order + BSConv + ConvNeXt": "configs/ablation/first_order.yaml",
    "2nd order + BSConv, ResBlock": "configs/ablation/no_convnext.yaml",
    "2nd order + ConvNeXt, no BSConv": "configs/ablation/no_bsconv.yaml",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", nargs="*", help="config files (default: main model + ablations)")
    p.add_argument("--lr-size", type=int, nargs=2, default=(180, 320))
    p.add_argument("--num-frames", type=int, default=7)
    p.add_argument("--no-flops", action="store_true")
    p.add_argument("--runtime", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out")
    args = p.parse_args()

    configs = {Path(c).stem: c for c in args.config} if args.config else {k: ROOT / v for k, v in DEFAULT_CONFIGS.items()}
    rows = {}
    for name, path in configs.items():
        cfg = ForkNetConfig.from_dict(load_config(path).get("model", {}))
        model = ForkNet(cfg).to(args.device).eval()
        row = param_counts(model)
        if not args.no_flops:
            row.update(flops_per_frame(model, tuple(args.lr_size), args.num_frames, args.device))
        if args.runtime:
            row.update(runtime_per_frame(model, tuple(args.lr_size), device=args.device))
        rows[name] = row
        msg = f"{name:45s} params {row['params_total'] / 1e6:6.3f}M (w/o SpyNet {row['params_without_spynet'] / 1e6:6.3f}M)"
        if "macs_per_frame" in row:
            msg += f"  MACs/frame {row['macs_per_frame'] / 1e12:.4f}T  FLOPs/frame {row['flops_per_frame'] / 1e12:.4f}T"
        if "seconds_per_frame" in row:
            msg += f"  {row['seconds_per_frame'] * 1000:.1f} ms/frame"
        print(msg, flush=True)
        del model
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
