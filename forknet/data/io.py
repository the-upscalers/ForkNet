from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_frames(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTENSIONS)


def list_clips(root):
    return sorted(p.name for p in Path(root).iterdir() if p.is_dir())


def read_image(path):
    """Reads an image as an RGB uint8 array (H, W, 3)."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def write_image(path, img):
    """Writes an RGB uint8 array (H, W, 3)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def tensor_to_uint8(x):
    """(3, H, W) float tensor in [0, 1] -> (H, W, 3) uint8 array, rounding like BasicSR's tensor2img."""
    x = x.detach().float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return (x * 255.0).round().astype(np.uint8)
