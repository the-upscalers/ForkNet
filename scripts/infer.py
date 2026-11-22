"""Super-resolve LR videos (frame folders or video files) with a trained checkpoint.

    python scripts/infer.py --checkpoint ckpt.pth --input path/to/lr_frames_or_video.mp4 --out outputs/demo
    python scripts/infer.py --checkpoint ckpt.pth --input path/to/root_of_clip_folders --out outputs/demo
"""

import argparse
import os

# Full-frame REDS inference fragments the CUDA cache (OOM with GBs reserved but unused); must precede `import torch`.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from forknet.data import list_frames, read_image, tensor_to_uint8, write_image  # noqa: E402
from forknet.inference import upscale_sequence  # noqa: E402
from forknet.utils import build_model_from_checkpoint, get_logger  # noqa: E402

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


def read_video(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames, fps = [], cap.get(cv2.CAP_PROP_FPS) or 25
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise IOError(f"No frames decoded from {path}")
    return frames, fps


def write_video(path, frames, fps):
    import cv2

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def sequences(inp):
    inp = Path(inp)
    if inp.is_file() and inp.suffix.lower() in VIDEO_EXT:
        frames, fps = read_video(inp)
        yield inp.stem, frames, [f"{i:08d}" for i in range(len(frames))], fps
    elif inp.is_dir() and list_frames(inp):
        paths = list_frames(inp)
        yield inp.name, [read_image(p) for p in paths], [p.stem for p in paths], None
    elif inp.is_dir():
        for sub in sorted(p for p in inp.iterdir() if p.is_dir() or p.suffix.lower() in VIDEO_EXT):
            yield from sequences(sub)
    else:
        raise FileNotFoundError(inp)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chunk-size", type=int, default=15)
    p.add_argument("--overlap", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--tile", type=int, help="process frames in overlapping TILE x TILE LR tiles (large inputs)")
    p.add_argument("--video", action="store_true", help="also write an .mp4 per sequence")
    p.add_argument("--max-frames", type=int, help="only the first N frames of each sequence")
    args = p.parse_args()

    log = get_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = build_model_from_checkpoint(args.checkpoint)
    model.to(device).eval()
    for name, frames, names, fps in sequences(args.input):
        frames, names = frames[: args.max_frames], names[: args.max_frames]
        lrs =torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(255)[None].to(device)
        srs = upscale_sequence(model, lrs, args.chunk_size, args.overlap, args.amp, tile=args.tile)
        out = [tensor_to_uint8(srs[0, t]) for t in range(srs.size(1))]
        for img, n in zip(out, names):
            write_image(Path(args.out) / name / f"{n}.png", img)
        if args.video:
            write_video(Path(args.out) / f"{name}.mp4", out, fps or 25)
        log.info(f"{name}: {len(out)} frames {frames[0].shape[:2]} -> {out[0].shape[:2]}")


if __name__ == "__main__":
    main()
