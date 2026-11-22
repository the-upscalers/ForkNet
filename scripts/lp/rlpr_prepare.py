"""Convert RLPR (Na et al., CVIU 2025; Mendeley DOI 10.17632/4rs5wpvckz.2, CC BY-NC-ND 4.0) to this repo's layout.

RLPR: 200 real dash-cam plate sequences of 31 low-quality crops each, a pseudo-ground-truth plate (a high-quality frame of
the same plate, not pixel-aligned) and the digits of the plate (Hangul removed for privacy). The crops of one sequence
can differ slightly in size; every frame is resized (bicubic) to the sequence's median size, rounded to even numbers,
so a video SR model can process the sequence. Writes, under ``--out``:

    lr/<sample>/<NN>.png        LR sequence (input of every SR method)
    bicubic/<sample>/<NN>.png   x4 bicubic of lr (baseline)
    pseudo_gt/<sample>/00.png   Pseudo_GT_ROI (recognition upper bound; no PSNR, it is not aligned)
    mflpr2/<sample>/00.png      SR_ROI released with the dataset (MF-LPR2 output, for reference)
    labels.csv                  clip,label,status  (label = digits only, status = verified)

    python scripts/lp/rlpr_prepare.py --zip rlpr.zip --out data/rlpr
"""

import argparse
import csv
import io
import re
import statistics
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def even(x):
    return max(2, int(round(x / 2)) * 2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    z = zipfile.ZipFile(args.zip)
    names = z.namelist()
    label_file = next(n for n in names if n.endswith("Label/Labels.txt"))
    labels = [ln.strip() for ln in z.read(label_file).decode().splitlines() if ln.strip()]
    samples = sorted({m.group(1) for n in names if (m := re.search(r"/Dataset/(sample_\d+)/", n))})
    if len(samples) != len(labels):
        raise SystemExit(f"{len(samples)} samples but {len(labels)} labels")
    rows = []
    for sample, label in zip(samples, labels):  # Labels.txt line i <-> sample_{i:03d}
        frames = sorted(n for n in names if f"/{sample}/Plate_crop/" in n and n.endswith(".png"))
        imgs = [Image.open(io.BytesIO(z.read(n))).convert("RGB") for n in frames]
        w = even(statistics.median(im.width for im in imgs))
        h = even(statistics.median(im.height for im in imgs))
        for i, im in enumerate(imgs):
            lr = im.resize((w, h), Image.BICUBIC) if im.size != (w, h) else im
            for sub, img in (("lr", lr), ("bicubic", lr.resize((4 * w, 4 * h), Image.BICUBIC))):
                path = out / sub / sample / f"{i:02d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                img.save(path)
        for sub, fname in (("pseudo_gt", "Pseudo_GT_ROI.png"), ("mflpr2", "SR_ROI.png")):
            src = next((n for n in names if n.endswith(f"/{sample}/{fname}")), None)
            if src:
                path = out / sub / sample / "00.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.open(io.BytesIO(z.read(src))).convert("RGB").save(path)
        rows.append({"clip": sample, "label": re.sub(r"\D", "", label), "status": "verified"})
    with open(out / "labels.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["clip", "label", "status"])
        wr.writeheader()
        wr.writerows(rows)
    sizes = np.array([Image.open(next((out / "lr" / s).iterdir())).size for s in samples])
    print(f"{len(samples)} sequences -> {out}; LR width {sizes[:, 0].min()}-{sizes[:, 0].max()}, "
          f"height {sizes[:, 1].min()}-{sizes[:, 1].max()} px")


if __name__ == "__main__":
    main()
