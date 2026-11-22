"""Plate boxes on HR frames (YOLOv9 plate detector, ``forknet.lp.ocr.PlateLocator``) for the plate-text loss
(``data.train.plate_boxes``): ``{clip: {frame_stem: [x1, y1, x2, y2]}}``; frames without a detection are left out.

    python scripts/lp/plate_boxes.py --hr-root data/lp/lpsr/train/sharp --out data/lp/train_boxes.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forknet.data import list_clips, list_frames, read_image  # noqa: E402
from forknet.lp.ocr import PlateLocator  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hr-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()
    det = PlateLocator(conf_thresh=args.conf)
    boxes, n, found = {}, 0, 0
    for clip in list_clips(args.hr_root):
        for f in list_frames(Path(args.hr_root) / clip):
            n += 1
            b = det(read_image(f))
            if b is not None:
                boxes.setdefault(clip, {})[f.stem] = list(b)
                found += 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(boxes))
    print(f"plates found in {found}/{n} frames of {len(boxes)} clips -> {args.out}")


if __name__ == "__main__":
    main()
