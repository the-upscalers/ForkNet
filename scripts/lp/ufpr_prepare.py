"""Convert UFPR-SR-Plates (Nascimento et al., JBCS 2025; academic use only, licence from the authors) to this repo's layout.

UFPR-SR-Plates: 10,000 tracks, each with 5 consecutive real low-resolution and 5 high-resolution crops of the same
plate (Brazilian / Mercosur, ~19x35 to 21x49 px LR) and JSON annotations with the plate text (plus layout and corners),
split 40/20/40 into train/val/test by plate. The release's exact file layout is not documented publicly, so tracks are
discovered: a track is a folder holding images whose names contain ``lr`` and ``hr``; the plate text is the first
string field whose key contains ``text`` (or ``plate``/``label``) in a JSON file of the track (majority over files).
Run with ``--inspect`` first: it prints what was found without writing anything.

The LR frames of a track are resized (bicubic) to their median size, rounded to even numbers, so a video SR model can
process them; the HR crops are not pixel-aligned with the LR ones (different moment of the track), so the metric is
plate recognition, not PSNR. Writes, under ``--out``:

    lr/<track>/<NN>.png       LR sequence (input of every SR method)
    bicubic/<track>/<NN>.png  x4 bicubic of lr (baseline)
    hr/<track>/<NN>.png       HR crops (recognition upper bound)
    labels.csv                clip,label,status,layout,split  (status = verified)

    python scripts/lp/ufpr_prepare.py --root data/UFPR-SR-Plates --split test --out data/ufpr_test
"""

import argparse
import collections
import csv
import json
import re
import statistics
from pathlib import Path

from PIL import Image

IMG = {".png", ".jpg", ".jpeg", ".bmp"}


def even(x):
    return max(2, int(round(x / 2)) * 2)


def find_text(obj, keys=("text", "plate", "label")):
    """First string value whose key contains one of ``keys`` (depth-first), and a layout if present."""
    text, layout = None, None
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            for k, v in o.items():
                kl = k.lower()
                if isinstance(v, str) and text is None and any(s in kl for s in keys) and re.fullmatch(r"[A-Za-z0-9 -]{4,10}", v):
                    text = v
                if isinstance(v, str) and layout is None and "layout" in kl:
                    layout = v
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(o, list):
            stack.extend(o)
    return text, layout


def discover(root):
    tracks = collections.defaultdict(lambda: {"lr": [], "hr": [], "json": []})
    for p in Path(root).rglob("*"):
        if not p.is_file():
            continue
        stem = p.stem.lower()
        if p.suffix.lower() in IMG:
            if re.search(r"(^|[^a-z])lr([^a-z]|$)", stem):
                tracks[p.parent]["lr"].append(p)
            elif re.search(r"(^|[^a-z])hr([^a-z]|$)", stem):
                tracks[p.parent]["hr"].append(p)
        elif p.suffix.lower() == ".json":
            tracks[p.parent]["json"].append(p)
    return {d: v for d, v in tracks.items() if v["lr"]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--split", default="test", help="keep tracks whose path contains this word ('' = all)")
    p.add_argument("--out")
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--limit", type=int, help="at most N tracks (smoke tests)")
    args = p.parse_args()
    tracks = discover(args.root)
    splits = collections.Counter(next((s for s in ("train", "val", "test") if s in str(d).lower()), "?") for d in tracks)
    print(f"{len(tracks)} track folders with LR images; split words in paths: {dict(splits)}")
    sel = sorted(d for d in tracks if args.split.lower() in str(d).lower()) if args.split else sorted(tracks)
    if args.limit:
        sel = sel[: args.limit]
    if args.inspect or not args.out:
        for d in sel[:3]:
            t = tracks[d]
            print(f"\n{d}\n  lr: {[x.name for x in sorted(t['lr'])]}\n  hr: {[x.name for x in sorted(t['hr'])]}\n  json: {[x.name for x in t['json']]}")
            for j in t["json"][:2]:
                print("  ", j.read_text()[:400].replace("\n", " "))
            print("  -> text/layout:", find_text(json.loads(t["json"][0].read_text())) if t["json"] else None,
                  " LR sizes:", [Image.open(x).size for x in sorted(t["lr"])])
        print(f"\nselected {len(sel)} tracks for split '{args.split}'")
        return
    out = Path(args.out)
    rows, missing = [], 0
    for d in sel:
        t = tracks[d]
        reads = [find_text(json.loads(j.read_text())) for j in t["json"]]
        texts = [re.sub(r"[^A-Z0-9]", "", x.upper()) for x, _ in reads if x]
        if not texts:
            missing += 1
            continue
        label = collections.Counter(texts).most_common(1)[0][0]
        layout = next((lay for _, lay in reads if lay), "")
        name = "_".join(d.relative_to(args.root).parts)
        lrs = [Image.open(x).convert("RGB") for x in sorted(t["lr"])]
        w = even(statistics.median(im.width for im in lrs))
        h = even(statistics.median(im.height for im in lrs))
        for i, im in enumerate(lrs):
            lr = im.resize((w, h), Image.BICUBIC) if im.size != (w, h) else im
            for sub, img in (("lr", lr), ("bicubic", lr.resize((4 * w, 4 * h), Image.BICUBIC))):
                (out / sub / name).mkdir(parents=True, exist_ok=True)
                img.save(out / sub / name / f"{i:02d}.png")
        for i, x in enumerate(sorted(t["hr"])):
            (out / "hr" / name).mkdir(parents=True, exist_ok=True)
            Image.open(x).convert("RGB").save(out / "hr" / name / f"{i:02d}.png")
        rows.append({"clip": name, "label": label, "status": "verified", "layout": layout, "split": args.split})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "labels.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["clip", "label", "status", "layout", "split"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} tracks -> {out} ({missing} without a readable plate text)")


if __name__ == "__main__":
    main()
