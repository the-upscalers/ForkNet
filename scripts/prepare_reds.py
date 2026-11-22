"""Arrange REDS into the layout the configs expect and verify it.

Official REDS (https://seungjunnah.github.io/Datasets/reds.html) ships train_sharp (240 clips),
val_sharp (30 clips) and their *_bicubic/X4 LR versions. Following BasicVSR, the 30 validation
clips are renamed 240-269 and merged with the training clips (270 total); REDS4
(000, 011, 015, 020) is held out for testing, leaving 266 training clips.

    python scripts/prepare_reds.py --reds-root /data/REDS --dst data/REDS        # links (default)
    python scripts/prepare_reds.py --reds-root /data/REDS --dst data/REDS --copy
    python scripts/prepare_reds.py --verify data/REDS

Expected input (any of the usual nestings is searched):
    <reds-root>/train_sharp/000..239, <reds-root>/val_sharp/000..029,
    <reds-root>/train_sharp_bicubic/X4/000..239, <reds-root>/val_sharp_bicubic/X4/000..029
Output:
    <dst>/HR/000..269/00000000..00000099.png, <dst>/LR_bicubic_X4/000..269/...
"""

import argparse
import os
import shutil
from pathlib import Path

REDS4 = ("000", "011", "015", "020")


def find_dir(root, name, sub=None):
    for cand in [root / name, root / name / name, root / "train" / name, root / "val" / name]:
        d = cand / sub if sub else cand
        if d.is_dir() and any(d.iterdir()):
            return d
    hits = [p for p in root.rglob(name) if p.is_dir()]
    for h in hits:
        d = h / sub if sub else h
        if d.is_dir():
            return d
    raise FileNotFoundError(f"{name}{'/' + sub if sub else ''} not found under {root}")


def place(src, dst, copy):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copytree(src, dst)
    else:
        try:
            os.symlink(src.resolve(), dst, target_is_directory=True)
        except OSError:  # e.g. Windows without symlink privilege
            shutil.copytree(src, dst)


def verify(root):
    root = Path(root)
    problems = 0
    for kind in ("HR", "LR_bicubic_X4"):
        clips = sorted(p.name for p in (root / kind).iterdir() if p.is_dir())
        counts = {c: len(list((root / kind / c).glob("*.png"))) for c in clips}
        bad = {c: n for c, n in counts.items() if n != 100}
        print(f"{kind}: {len(clips)} clips; REDS4 present: {all(c in counts for c in REDS4)}; clips without 100 frames: {bad or 'none'}")
        problems += bool(bad) + (len(clips) not in (240, 270))
    print("OK" if not problems else f"{problems} problem(s) found")
    return problems == 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reds-root")
    p.add_argument("--dst")
    p.add_argument("--copy", action="store_true")
    p.add_argument("--no-val", action="store_true", help="use only the 240 train_sharp clips (as the original code did)")
    p.add_argument("--verify")
    args = p.parse_args()
    if args.verify:
        raise SystemExit(0 if verify(args.verify) else 1)
    if not (args.reds_root and args.dst):
        p.error("--reds-root and --dst are required unless --verify is given")
    root, dst = Path(args.reds_root), Path(args.dst)
    parts = [("train_sharp", "train_sharp_bicubic", 0)] + ([] if args.no_val else [("val_sharp", "val_sharp_bicubic", 240)])
    for hr_name, lr_name, offset in parts:
        hr_dir, lr_dir = find_dir(root, hr_name), find_dir(root, lr_name, "X4")
        for clip in sorted(p.name for p in hr_dir.iterdir() if p.is_dir()):
            new = f"{int(clip) + offset:03d}"
            place(hr_dir / clip, dst / "HR" / new, args.copy)
            place(lr_dir / clip, dst / "LR_bicubic_X4" / new, args.copy)
    verify(dst)


if __name__ == "__main__":
    main()
