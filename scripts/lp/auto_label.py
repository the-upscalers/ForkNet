"""Recover plate strings for LP video clips from their HR frames.

Every frame is read by each OCR engine; per-engine readings are merged by temporal consensus
(forknet.lp.text.consensus). A clip label is accepted automatically when two independent engines
produce the same consensus string (status ``auto_agree``), or when a single engine is highly
self-consistent (``auto_single``); otherwise it is marked ``review``. Contact sheets (8 frames +
proposed label) are written for manual verification. Edit the CSV and set status to ``verified``.

    python scripts/lp/auto_label.py --hr-root data/lp/lpsr/test/sharp --engines easyocr fast_plate_ocr \\
        --out results/lp_labels/test_labels.csv --sheets results/lp_labels/sheets
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from forknet.data import list_frames, read_image, resolve_clips  # noqa: E402
from forknet.lp.ocr import PlateLocator, build_engine, crop_plate  # noqa: E402
from forknet.lp.text import consensus  # noqa: E402


def contact_sheet(frames, text, path, n=8):
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    tiles = [Image.fromarray(frames[i]).resize((192, 192)) for i in idx]
    sheet = Image.new("RGB", (192 * len(tiles), 222), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, (192 * k, 30))
    ImageDraw.Draw(sheet).text((6, 6), f"{path.stem}: {text or '?'}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hr-root", required=True)
    p.add_argument("--clips", nargs="*")
    p.add_argument("--split-file")
    p.add_argument("--split")
    p.add_argument("--engines", nargs="+", default=["easyocr", "fast_plate_ocr"])
    p.add_argument("--every", type=int, default=1, help="read every k-th frame")
    p.add_argument("--min-agreement", type=float, default=0.6)
    p.add_argument("--min-support", type=int, default=5)
    p.add_argument("--detector", default="yolo-v9-t-384-license-plate-end2end",
                   help="plate detector (open-image-models) used to crop plates before OCR; 'none' reads whole frames")
    p.add_argument("--boxes", help="write the per-frame plate boxes (JSON) for scripts/lp/ocr_eval.py --boxes")
    p.add_argument("--out", required=True)
    p.add_argument("--sheets")
    args = p.parse_args()

    locate = None
    if args.detector != "none":
        try:
            locate = PlateLocator(args.detector)
        except Exception as e:
            print(f"[warn] plate detector unavailable ({e}); reading whole frames")
    engines = []
    for name in args.engines:
        try:
            engines.append(build_engine(name))
        except Exception as e:  # missing optional dependency: continue with the others
            print(f"[warn] engine {name} unavailable: {e}")
    if not engines:
        raise SystemExit("No OCR engine available")
    clips = resolve_clips(args.hr_root, args.clips, None, args.split_file, args.split)
    rows, all_boxes = [], {}
    for clip in clips:
        paths = list_frames(Path(args.hr_root) / clip)[:: args.every]
        frames = [read_image(f) for f in paths]
        boxes = [locate(f) for f in frames] if locate else [None] * len(frames)
        all_boxes[clip] = {Path(fp).stem: b for fp, b in zip(paths, boxes)}
        crops = [crop_plate(f, b) for f, b in zip(frames, boxes)]
        row = {"clip": clip, "num_frames": len(frames), "plate_found": sum(b is not None for b in boxes)}
        per_engine = {}
        for eng in engines:
            text, support, agree = consensus(eng(c) for c in crops)
            per_engine[eng.name] = (text, support, agree)
            row.update({f"{eng.name}_text": text, f"{eng.name}_support": support, f"{eng.name}_agreement": round(agree, 3)})
        texts = {t for t, _, _ in per_engine.values() if t}
        best = max(per_engine.values(), key=lambda v: (v[2], v[1]))
        if len(per_engine) > 1 and len(texts) == 1 and all(t for t, _, _ in per_engine.values()):
            row.update(label=texts.pop(), status="auto_agree")
        elif best[0] and best[2] >= args.min_agreement and best[1] >= args.min_support:
            row.update(label=best[0], status="auto_single")
        else:
            row.update(label=best[0], status="review")
        rows.append(row)
        print(f"{clip}: {row['label'] or '-':12s} {row['status']}  " +
              "  ".join(f"{k}={v[0]}({v[2]:.2f})" for k, v in per_engine.items()), flush=True)
        if args.sheets:
            contact_sheet(frames, row["label"], Path(args.sheets) / f"{clip}.jpg")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["clip", "label", "status", "num_frames", "plate_found"]
    fields += sorted({k for r in rows for k in r} - set(fields))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if args.boxes:
        Path(args.boxes).parent.mkdir(parents=True, exist_ok=True)
        Path(args.boxes).write_text(json.dumps(all_boxes))
    counts = {s: sum(r["status"] == s for r in rows) for s in ("auto_agree", "auto_single", "review")}
    print(f"\n{len(rows)} clips: {counts}  ->  {args.out}")


if __name__ == "__main__":
    main()
