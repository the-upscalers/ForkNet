"""Identify the architecture of a checkpoint (including ones written by the original research code),
and optionally export it in the release format.

    python scripts/inspect_checkpoint.py path/to/best_checkpoint.pth
    python scripts/inspect_checkpoint.py path/to/best_checkpoint.pth --export weights/forknet_reds.pth
"""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forknet.archs import ForkNet  # noqa: E402
from forknet.utils import load_checkpoint, param_counts, save_checkpoint  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--export", help="write a weights-only release checkpoint here")
    args = p.parse_args()

    sd, cfg, meta = load_checkpoint(args.checkpoint)
    model = ForkNet(cfg)
    model.load_state_dict(sd, strict=True)
    counts = param_counts(model)
    print(f"format      : {meta.pop('format')}")
    print(f"metadata    : {meta}")
    print("architecture:")
    for k, v in cfg.to_dict().items():
        print(f"  {k:24s} {v}")
    print(f"params      : {counts['params_total']:,} total, {counts['params_without_spynet']:,} without SpyNet")
    if args.export:
        save_checkpoint(args.export, model, cfg, source=str(args.checkpoint), source_meta=meta)
        digest = hashlib.sha256(Path(args.export).read_bytes()).hexdigest()
        print(f"exported    : {args.export} (sha256 {digest})")


if __name__ == "__main__":
    main()
