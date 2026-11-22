"""Plate-recognition accuracy of SR outputs (the downstream task).

For each method directory (``<dir>/<clip>/<frame>.png``) every frame is read by the OCR engine.
Reported per method:
    clip_exact  share of clips whose temporal-consensus string equals the label
    clip_char   mean character accuracy (1 - CER) of the consensus string
    frame_exact share of individual frames read exactly
    frame_char  mean per-frame character accuracy

    python scripts/lp/ocr_eval.py --labels results/lp_labels/test_labels.csv --status auto_agree verified \\
        --method HR=data/lp/lpsr/test/sharp Bicubic=results/lp/bicubic/images ForkNet=results/lp/forknet/images \\
        --engine easyocr --out results/lp_ocr.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from forknet.data import list_frames, read_image  # noqa: E402
from forknet.lp.ocr import build_engine, crop_plate  # noqa: E402
from forknet.lp.text import char_accuracy, consensus, normalize  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True)
    p.add_argument("--status", nargs="*", default=["auto_agree", "verified"], help="label statuses to use")
    p.add_argument("--method", nargs="+", required=True, metavar="NAME=DIR")
    p.add_argument("--engine", default="easyocr")
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--reference", help="method name (e.g. Bicubic): report clips each method fixes / breaks w.r.t. it")
    p.add_argument("--boxes", help="plate boxes from scripts/lp/auto_label.py --boxes (found on the HR frames); "
                   "every method is read inside the same box, so only reconstruction quality differs")
    p.add_argument("--digits-only", action="store_true",
                   help="compare digits only (labels without letters, e.g. RLPR, whose Hangul characters are removed)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    boxes = json.loads(Path(args.boxes).read_text()) if args.boxes else {}
    norm = (lambda t: "".join(c for c in normalize(t) if c.isdigit())) if args.digits_only else normalize

    with open(args.labels) as f:
        labels = {r["clip"]: norm(r["label"]) for r in csv.DictReader(f) if r["status"] in args.status and r["label"]}
    if not labels:
        raise SystemExit("No usable labels")
    engine = build_engine(args.engine)
    report = {"engine": args.engine, "num_clips": len(labels), "methods": {}}
    for spec in args.method:
        name, _, root = spec.partition("=")
        clip_exact, clip_char, frame_exact, frame_char, per_clip = [], [], [], [], {}
        for clip, gt in labels.items():
            frames = list_frames(Path(root) / clip)[:: args.every]
            reads = [engine(crop_plate(read_image(fp), boxes.get(clip, {}).get(Path(fp).stem))) for fp in frames]
            reads = [(norm(t), c) for t, c in reads]
            text, _, _ = consensus(reads)
            clip_exact.append(text == gt)
            clip_char.append(char_accuracy(text, gt))
            frame_exact += [t == gt for t, _ in reads]
            frame_char += [char_accuracy(t, gt) for t, _ in reads]
            per_clip[clip] = {"label": gt, "prediction": text, "correct": text == gt}
        m = {"clip_exact": float(np.mean(clip_exact)), "clip_char": float(np.mean(clip_char)),
             "frame_exact": float(np.mean(frame_exact)), "frame_char": float(np.mean(frame_char)), "clips": per_clip}
        report["methods"][name] = m
        print(f"{name:16s} clip exact {m['clip_exact']:.3f}  clip char {m['clip_char']:.3f}  "
              f"frame exact {m['frame_exact']:.3f}  frame char {m['frame_char']:.3f}", flush=True)
    if args.reference:
        # SR-induced misreads: clips the reference reads correctly but the method gets wrong ("breaks"),
        # and the converse ("fixes"); McNemar's exact test on the discordant pairs.
        from math import comb

        ref = report["methods"][args.reference]["clips"]
        print(f"\nRelative to {args.reference}:")
        for name, m in report["methods"].items():
            if name == args.reference:
                continue
            fixes = sum(not ref[c]["correct"] and m["clips"][c]["correct"] for c in ref)
            breaks = sum(ref[c]["correct"] and not m["clips"][c]["correct"] for c in ref)
            n = fixes + breaks
            p_val = min(1.0, 2 * sum(comb(n, k) for k in range(min(fixes, breaks) + 1)) / 2**n) if n else 1.0
            m["vs_reference"] = {"reference": args.reference, "fixes": fixes, "breaks": breaks, "mcnemar_p": p_val}
            print(f"  {name:16s} fixes {fixes:3d}  breaks {breaks:3d}  McNemar p = {p_val:.3g}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
