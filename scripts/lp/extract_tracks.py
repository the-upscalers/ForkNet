"""Build LP video clips from raw traffic / CCTV video: detect plates, track them, crop a stable
square window around each track, and save each long-enough track as a clip of HR frames.

Requires ``pip install ultralytics`` and a YOLO plate-detector checkpoint (``--detector``), e.g. a
YOLOv8/YOLO11 model fine-tuned on a public plate dataset. Tracking uses Ultralytics' ByteTrack.

    python scripts/lp/extract_tracks.py --videos raw_videos/ --detector plate_yolov8n.pt --out data/lp_mined/HR \\
        --crop 256 --min-frames 10

The output is an HR clip folder tree; run scripts/prepare_lp_dataset.py on it to synthesise LR frames,
and scripts/lp/auto_label.py to recover plate strings.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from forknet.data import write_image  # noqa: E402

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def smooth(values, k=5):
    v = np.asarray(values, dtype=np.float64)
    if len(v) < k:
        return v
    pad = np.pad(v, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", required=True, help="video file or folder of videos")
    p.add_argument("--detector", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--crop", type=int, default=256, help="output crop size (HR)")
    p.add_argument("--context", type=float, default=2.0, help="crop side = context x plate width (before resizing)")
    p.add_argument("--min-frames", type=int, default=10)
    p.add_argument("--min-plate-width", type=int, default=24, help="skip tracks whose plate is never this wide (px)")
    p.add_argument("--conf", type=float, default=0.35)
    args = p.parse_args()

    import cv2
    from ultralytics import YOLO

    model = YOLO(args.detector)
    src = Path(args.videos)
    videos = [src] if src.is_file() else sorted(v for v in src.rglob("*") if v.suffix.lower() in VIDEO_EXT)
    meta = []
    for vid in videos:
        # Pass 1: detection + tracking (boxes only, frames are not kept in memory).
        tracks = defaultdict(list)  # id -> [(frame_idx, cx, cy, w, h)]
        for i, r in enumerate(model.track(str(vid), stream=True, persist=True, conf=args.conf, verbose=False, tracker="bytetrack.yaml")):
            if r.boxes.id is None:
                continue
            for tid, (cx, cy, w, h) in zip(r.boxes.id.int().tolist(), r.boxes.xywh.tolist()):
                tracks[tid].append((i, cx, cy, w, h))
        # Smoothed square crop window per kept track, indexed by frame.
        jobs = defaultdict(list)  # frame_idx -> [(clip name, k, x, y, side)]
        for tid, obs in tracks.items():
            if len(obs) < args.min_frames or max(o[3] for o in obs) < args.min_plate_width:
                continue
            idx = [o[0] for o in obs]
            cx, cy = smooth([o[1] for o in obs]), smooth([o[2] for o in obs])
            side = int(round(np.median([o[3] for o in obs]) * args.context))
            name = f"{vid.stem}_t{tid:04d}"
            for k, (fi, x, y) in enumerate(zip(idx, cx, cy)):
                jobs[fi].append((name, k, x, y, side))
            meta.append({"clip": name, "video": str(vid), "track": tid, "frames": len(idx), "first_frame": idx[0],
                         "median_plate_width_px": float(np.median([o[3] for o in obs]))})
            print(f"{name}: {len(idx)} frames, plate ~{meta[-1]['median_plate_width_px']:.0f}px", flush=True)
        # Pass 2: decode again and write the crops.
        cap, fi = cv2.VideoCapture(str(vid)), 0
        while jobs:
            ok, img = cap.read()
            if not ok:
                break
            for name, k, x, y, side in jobs.pop(fi, []):
                x0, y0 = int(round(x - side / 2)), int(round(y - side / 2))
                padded = cv2.copyMakeBorder(img, side, side, side, side, cv2.BORDER_REPLICATE)
                crop = padded[y0 + side : y0 + 2 * side, x0 + side : x0 + 2 * side]
                crop = cv2.resize(crop, (args.crop, args.crop), interpolation=cv2.INTER_AREA)
                write_image(Path(args.out) / name / f"{k:08d}.png", cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            fi += 1
        cap.release()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "tracks.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(meta)} clips written to {args.out}")


if __name__ == "__main__":
    main()
