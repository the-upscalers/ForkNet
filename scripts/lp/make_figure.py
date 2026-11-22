"""Qualitative LP figure: the same plate crop (box found on the HR frame) from every method, one row per clip.

    python scripts/lp/make_figure.py --boxes results/lp_test_boxes.json --clips 001 007 013 \\
        --method HR=data/lp/test/sharp Bicubic=... Fork-Net=... --out figures/lp_qualitative.pdf
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from forknet.data import list_frames, read_image  # noqa: E402
from forknet.lp.ocr import crop_plate  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--boxes", required=True)
    p.add_argument("--clips", nargs="+", required=True)
    p.add_argument("--method", nargs="+", required=True, metavar="NAME=DIR")
    p.add_argument("--frame", type=float, default=0.5, help="relative position of the frame within the clip")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    boxes = json.loads(Path(args.boxes).read_text())
    methods = [m.partition("=")[::2] for m in args.method]
    fig, axes = plt.subplots(len(args.clips), len(methods), figsize=(1.9 * len(methods), 1.0 * len(args.clips) + 0.4),
                             squeeze=False)
    for r, clip in enumerate(args.clips):
        frames = list_frames(Path(methods[0][1]) / clip)
        stem = frames[min(len(frames) - 1, int(args.frame * len(frames)))].stem
        box = boxes.get(clip, {}).get(stem)
        for c, (name, root) in enumerate(methods):
            ax = axes[r][c]
            path = next((f for f in list_frames(Path(root) / clip) if f.stem == stem), None)
            if path is not None:
                ax.imshow(crop_plate(read_image(path), box, margin=0.25, min_height=0))
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"clip {clip}", fontsize=7)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
