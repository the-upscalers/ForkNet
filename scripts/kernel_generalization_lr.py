"""Build x4 LR test sets with resampling kernels that are NOT in the kernel-mix training pool, to test how far the
recipe generalizes. The pool (forknet/data/degradations.py) holds MATLAB bicubic, aliased OpenCV cubic and the
antialiased Pillow bilinear / box / Lanczos. Held-out kernels, as used by common pipelines:

  aliased (no low-pass before decimation): cv2_linear, cv2_nearest, cv2_lanczos4, torch_bicubic (a = -0.75)
  antialiased / other:                      pil_bicubic, cv2_area, ffmpeg_bicubic (swscale default scaler)

Writes <out>/<kernel>/<clip>/<frame>.png and <out>/GT/<clip>/<frame>.png (first --max-frames frames per clip).

    python scripts/kernel_generalization_lr.py --hr-root data/lp/test/sharp --split-file splits/lp_lpsr.json \\
        --split test_clean --max-frames 40 --out /tmp/kgen
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

KERNELS = ["cv2_linear", "cv2_nearest", "cv2_lanczos4", "torch_bicubic", "pil_bicubic", "cv2_area", "ffmpeg_bicubic"]


def ffmpeg_exe():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def down(img, kernel, size):
    """img: (H, W, 3) uint8 BGR; size: (w, h)."""
    if kernel.startswith("cv2_"):
        flag = {"cv2_linear": cv2.INTER_LINEAR, "cv2_nearest": cv2.INTER_NEAREST, "cv2_lanczos4": cv2.INTER_LANCZOS4,
                "cv2_area": cv2.INTER_AREA}[kernel]
        return cv2.resize(img, size, interpolation=flag)
    if kernel == "torch_bicubic":
        t = torch.from_numpy(img).permute(2, 0, 1)[None].float()
        o = F.interpolate(t, size=size[::-1], mode="bicubic", align_corners=False, antialias=False)
        return o[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()
    if kernel == "pil_bicubic":
        rgb = Image.fromarray(img[..., ::-1]).resize(size, Image.BICUBIC)
        return np.asarray(rgb)[..., ::-1].copy()
    if kernel == "ffmpeg_bicubic":
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "in.png", Path(d) / "out.png"
            cv2.imwrite(str(src), img)
            subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", "-i", str(src), "-vf",
                            f"scale={size[0]}:{size[1]}:flags=bicubic", str(dst)], check=True)
            return cv2.imread(str(dst))
    raise ValueError(kernel)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hr-root", required=True)
    p.add_argument("--split-file")
    p.add_argument("--split")
    p.add_argument("--clips", nargs="*")
    p.add_argument("--max-frames", type=int, default=40)
    p.add_argument("--kernels", nargs="*", default=KERNELS)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    hr_root, out = Path(args.hr_root), Path(args.out)
    clips = args.clips or (json.loads(Path(args.split_file).read_text())[args.split] if args.split_file
                           else sorted(d.name for d in hr_root.iterdir() if d.is_dir()))
    for clip in clips:
        for f in sorted((hr_root / clip).glob("*.png"))[: args.max_frames]:
            hr = cv2.imread(str(f))
            (out / "GT" / clip).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "GT" / clip / f.name), hr)
            size = (hr.shape[1] // 4, hr.shape[0] // 4)
            for k in args.kernels:
                (out / k / clip).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / k / clip / f.name), down(hr, k, size))
    print(f"{len(clips)} clips x {len(args.kernels)} kernels -> {out}")


if __name__ == "__main__":
    main()
