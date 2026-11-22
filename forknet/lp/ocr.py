"""Pluggable plate-text recognisers. Each engine maps an RGB uint8 crop to ``(text, confidence)``.

* ``easyocr``          generic scene-text OCR (``pip install easyocr``), robust default.
* ``fast_plate_ocr``   plate-specific CCT/ViT recogniser trained on plates from many countries
                       (``pip install fast-plate-ocr[onnx]``).

Using two unrelated engines lets labels be cross-checked (see scripts/lp/auto_label.py).
"""

import numpy as np

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class EasyOCREngine:
    name = "easyocr"

    def __init__(self, gpu=None):
        import easyocr
        import torch

        self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available() if gpu is None else gpu, verbose=False)

    def __call__(self, img):
        res = self.reader.readtext(img, allowlist=ALLOWLIST, detail=1, paragraph=False)
        if not res:
            return "", 0.0
        res.sort(key=lambda r: r[0][0][0])  # left to right
        text = "".join(r[1] for r in res)
        conf = float(np.mean([r[2] for r in res]))
        return text, conf


class FastPlateOCREngine:
    name = "fast_plate_ocr"

    def __init__(self, model="cct-xs-v1-global-model"):
        try:  # fast-plate-ocr >= 1.0
            from fast_plate_ocr import LicensePlateRecognizer

            self.rec = LicensePlateRecognizer(model)
            self.legacy = False
        except ImportError:  # 0.x API
            from fast_plate_ocr import ONNXPlateRecognizer

            self.rec = ONNXPlateRecognizer("global-plates-mobile-vit-v2-model")
            self.legacy = True

    def __call__(self, img):
        import cv2

        if not self.legacy:  # >= 1.0: list of images in the model's colour mode -> list of PlatePrediction
            mode = getattr(getattr(self.rec, "config", None), "image_color_mode", "grayscale")
            x = img if mode == "rgb" else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            pred = self.rec.run([x], return_confidence=True)[0]
            probs = getattr(pred, "char_probs", None)
            return pred.plate.replace("_", ""), float(np.mean(probs)) if probs is not None and len(probs) else 1.0
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        out = self.rec.run(gray, return_confidence=True)
        texts, probs = out if isinstance(out, tuple) else (out, None)
        text = texts[0] if isinstance(texts, (list, tuple)) else str(texts)
        text = text.replace("_", "")
        conf = float(np.mean(probs[0])) if probs is not None else 1.0
        return text, conf


ENGINES = {"easyocr": EasyOCREngine, "fast_plate_ocr": FastPlateOCREngine}


class PlateLocator:
    """Plate detector (YOLOv9 from ``pip install open-image-models``): RGB uint8 frame -> most confident
    ``(x1, y1, x2, y2)`` box, or None. Recognisers such as fast-plate-ocr expect a tight plate crop;
    on whole 256x256 vehicle crops they hallucinate."""

    def __init__(self, model="yolo-v9-t-384-license-plate-end2end", conf_thresh=0.25):
        try:  # current API
            from open_image_models import create_detector

            self.det = create_detector(model, conf_thresh=conf_thresh)
        except ImportError:  # older releases (pip may resolve one next to other pinned packages)
            from open_image_models import LicensePlateDetector

            self.det = LicensePlateDetector(detection_model=model, conf_thresh=conf_thresh)

    def __call__(self, img):
        res = self.det.predict(np.ascontiguousarray(img[..., ::-1]))  # expects BGR
        if not res:
            return None
        b = max(res, key=lambda r: r.confidence).bounding_box
        return int(b.x1), int(b.y1), int(b.x2), int(b.y2)


def crop_plate(img, box, margin=0.15, min_height=64):
    """Crops ``box`` (plus ``margin`` of its size on each side) and enlarges it to at least ``min_height``
    rows (bicubic) so small plates are not below the recognisers' working resolution. ``box=None``
    returns the frame unchanged."""
    if box is None:
        return img
    import cv2

    x1, y1, x2, y2 = box
    mx, my = int(round((x2 - x1) * margin)), int(round((y2 - y1) * margin))
    h, w = img.shape[:2]
    crop = img[max(0, y1 - my) : min(h, y2 + my), max(0, x1 - mx) : min(w, x2 + mx)]
    if crop.size == 0:
        return img
    if crop.shape[0] < min_height:
        f = min_height / crop.shape[0]
        crop = cv2.resize(crop, (round(crop.shape[1] * f), min_height), interpolation=cv2.INTER_CUBIC)
    return crop


def build_engine(name, **kwargs):
    if name not in ENGINES:
        raise ValueError(f"Unknown OCR engine {name!r}; choose from {list(ENGINES)}")
    return ENGINES[name](**kwargs)
