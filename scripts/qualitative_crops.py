"""Select and export crops for qualitative comparison figures, with the per-crop PSNR of every method.

Candidate patches are the high-detail regions of the ground-truth frame (top quartile of Sobel gradient energy, which
favours text, signs and edges); within them a patch is picked by its PSNR difference between ``--ours`` and ``--ref``:
``--select best`` takes the largest difference (stated in the caption), ``--select median`` a typical one. One crop per clip,
from the middle frame. Writes ``<out>/<clip>_<frame>/{<method>.png, frame.png, meta.json}``.

    python scripts/qualitative_crops.py --gt data/REDS4/GT --method Bicubic=out/bic BasicVSR++=out/bvpp Ours=out/ours \\
        --ref BasicVSR++ --ours Ours --patch 128 --out figs/reds4_cv2
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True, help="root with <clip>/<frame>.png")
    p.add_argument("--method", nargs="+", required=True, metavar="NAME=DIR")
    p.add_argument("--ref", required=True)
    p.add_argument("--ours", required=True)
    p.add_argument("--clips", nargs="*")
    p.add_argument("--patch", type=int, default=128, help="HR patch size")
    p.add_argument("--select", choices=("best", "median"), default="best")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    methods = dict(m.split("=", 1) for m in args.method)
    gt_root, out_root, ps = Path(args.gt), Path(args.out), args.patch
    clips = args.clips or sorted(d.name for d in gt_root.iterdir() if d.is_dir())
    for clip in clips:
        names = sorted(f.stem for f in (Path(methods[args.ours]) / clip).glob("*.png"))
        if not names:
            continue
        name = names[len(names) // 2]
        gt = cv2.imread(str(gt_root / clip / f"{name}.png"))
        sr = {m: cv2.imread(str(Path(d) / clip / f"{name}.png")) for m, d in methods.items()}
        if gt is None or any(v is None for v in sr.values()):
            print(f"{clip}: missing frame {name}, skipped")
            continue
        h, w = gt.shape[:2]
        gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY).astype(np.float32)
        energy = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        size = min(ps, h, w)
        step = max(size // 4, 1)
        cands = []
        for y in range(0, h - size + 1, step):
            for x in range(0, w - size + 1, step):
                cands.append((float(energy[y:y + size, x:x + size].mean()), y, x))
        thr = np.percentile([c[0] for c in cands], 75)
        scored = []
        for e, y, x in cands:
            if e < thr:
                continue
            g = gt[y:y + size, x:x + size]
            d = psnr(sr[args.ours][y:y + size, x:x + size], g) - psnr(sr[args.ref][y:y + size, x:x + size], g)
            scored.append((d, y, x))
        scored.sort()
        d, y, x = scored[-1] if args.select == "best" else scored[len(scored) // 2]
        dest = out_root / f"{clip}_{name}"
        dest.mkdir(parents=True, exist_ok=True)
        meta = {"clip": clip, "frame": name, "box": [int(x), int(y), size, size], "select": args.select,
                "ours_minus_ref_db": d, "psnr": {}, "frame_psnr": {}}
        for m, img in {**sr, "GT": gt}.items():
            crop = img[y:y + size, x:x + size]
            cv2.imwrite(str(dest / f"{m}.png"), crop)
            if m != "GT":
                meta["psnr"][m] = psnr(crop, gt[y:y + size, x:x + size])
                meta["frame_psnr"][m] = psnr(img, gt)
        frame = gt.copy()
        cv2.rectangle(frame, (x, y), (x + size - 1, y + size - 1), (0, 0, 255), max(2, w // 300))
        cv2.imwrite(str(dest / "frame.png"), frame)
        (dest / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"{clip}/{name}: box {x},{y} {size}px  {args.ours} - {args.ref} = {d:+.2f} dB", flush=True)


if __name__ == "__main__":
    main()
