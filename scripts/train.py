"""Train Fork-Net.

    python scripts/train.py --config configs/train_reds.yaml
    torchrun --nproc_per_node 2 scripts/train.py --config configs/train_reds.yaml
    python scripts/train.py --config configs/train_reds.yaml --set train.batch_size=2 name=debug
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forknet.trainer import Trainer  # noqa: E402
from forknet.utils import load_config  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--resume", help="checkpoint written by this trainer, or 'auto' for "
                   "<output_dir>/<name>/checkpoints/latest.pth if it exists")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    resume = args.resume
    if resume == "auto":
        latest = Path(cfg.get("output_dir", "experiments")) / cfg["name"] / "checkpoints" / "latest.pth"
        resume = str(latest) if latest.exists() else None
    Trainer(cfg, resume=resume).fit()


if __name__ == "__main__":
    main()
