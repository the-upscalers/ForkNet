"""No-reference quality (NIQE, BRISQUE; lower is better) of SR outputs for real-world videos without
ground truth, e.g. VideoLQ (RealBasicVSR protocol). Requires ``pip install pyiqa``.

    python scripts/eval_noref.py --sr outputs/forknet/videolq --name Fork-Net --out results/videolq_forknet.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from forknet.data import list_clips, list_frames, read_image  # noqa: E402
from forknet.metrics_extra import NoReferenceMetrics  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sr", required=True)
    p.add_argument("--clips", nargs="*")
    p.add_argument("--metrics", nargs="*", default=["niqe", "brisque"])
    p.add_argument("--every", type=int, default=1, help="score every k-th frame (speed)")
    p.add_argument("--name")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fn = NoReferenceMetrics(args.metrics, device)
    clips = args.clips or list_clips(args.sr)
    per_clip = {}
    for c in clips:
        frames = [read_image(f) for f in list_frames(Path(args.sr) / c)[:: args.every]]
        per_clip[c] = fn(frames)
        print(c, per_clip[c], flush=True)
    avg = {m: float(np.mean([v[m] for v in per_clip.values()])) for m in args.metrics}
    print("average", avg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"method": args.name, "sr": args.sr, "average": avg, "clips": per_clip}, indent=2))


if __name__ == "__main__":
    main()
