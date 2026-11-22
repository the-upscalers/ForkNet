"""Prepare the license-plate (LP) video dataset: synthesise LR frames and a reproducible split.

Input:  <src>/<clip>/<frame>.(png|jpg) HR frames (256x256 crops in the paper), or video files.
Output: <dst>/HR/<clip>/<frame:08d>.png, <dst>/LR/<clip>/<frame:08d>.png (x4 MATLAB bicubic),
        <dst>/splits.json  {"train": [...], "test": [...], "seed": ..., "test_ratio": ...}

    python scripts/prepare_lp_dataset.py --src raw_lp_clips --dst data/lp --test-ratio 0.2 --seed 0
    # use an existing split (e.g. the one used for the paper) instead of drawing a new one
    python scripts/prepare_lp_dataset.py --src raw_lp_clips --dst data/lp --split-file paper_split.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forknet.data import imresize, list_frames, read_image, write_image  # noqa: E402

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


def load_clip(path):
    if path.is_dir():
        return [read_image(p) for p in list_frames(path)]
    import cv2

    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-file", help="existing JSON split to copy instead of drawing a new one")
    p.add_argument("--min-frames", type=int, default=5)
    p.add_argument("--kernel", choices=("matlab", "cv2_bicubic"), default="matlab",
                   help="LR kernel: MATLAB bicubic (antialiased, as REDS) or OpenCV INTER_CUBIC "
                        "(aliased; how the released LP dataset was made)")
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    entries = sorted(e for e in src.iterdir() if e.is_dir() or e.suffix.lower() in VIDEO_EXT)
    kept = []
    for e in entries:
        frames = load_clip(e)
        if len(frames) < args.min_frames:
            print(f"skip {e.name}: {len(frames)} frames")
            continue
        h, w = frames[0].shape[:2]
        h, w = h - h % args.scale, w - w % args.scale  # crop so HR is divisible by the scale
        for i, f in enumerate(frames):
            hr = f[:h, :w]
            write_image(dst / "HR" / e.stem / f"{i:08d}.png", hr)
            if args.kernel == "matlab":
                lr = imresize(hr, 1 / args.scale)
            else:
                import cv2

                lr = cv2.resize(hr, (w // args.scale, h // args.scale), interpolation=cv2.INTER_CUBIC)
            write_image(dst / "LR" / e.stem / f"{i:08d}.png", lr)
        kept.append(e.stem)
        print(f"{e.stem}: {len(frames)} frames, HR {h}x{w}")

    if args.split_file:
        with open(args.split_file) as f:
            split = json.load(f)
        missing = (set(split["train"]) | set(split["test"])) - set(kept)
        if missing:
            raise ValueError(f"Split refers to clips not found: {sorted(missing)[:10]}")
    else:
        rng = random.Random(args.seed)
        shuffled = kept[:]
        rng.shuffle(shuffled)
        n_test = round(len(shuffled) * args.test_ratio)
        split = {"train": sorted(shuffled[n_test:]), "test": sorted(shuffled[:n_test]), "seed": args.seed, "test_ratio": args.test_ratio}
    with open(dst / "splits.json", "w") as f:
        json.dump(split, f, indent=2)
    print(f"{len(kept)} clips -> {len(split['train'])} train / {len(split['test'])} test; wrote {dst / 'splits.json'}")


if __name__ == "__main__":
    main()
