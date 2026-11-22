"""Optional training losses on top of the Charbonnier pixel loss (all off unless their weight is > 0).

* ``perceptual_weight``: LPIPS (VGG) between SR and HR frames (a subsample of each clip, for speed).
* ``gan_weight``: adversarial loss with the U-Net discriminator with spectral normalisation of Real-ESRGAN /
  RealBasicVSR (vanilla GAN, logits); the discriminator is trained alternately with its own optimiser.
* ``text_weight``: plate text-feature loss - L1 between the features of a frozen text recogniser (EasyOCR's
  VGG feature extractor) on the plate region of SR and HR frames; needs plate masks from the dataset
  (``data.train.plate_boxes``). A recognition-aware loss without a differentiable OCR decoder.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class UNetDiscriminatorSN(nn.Module):
    """Real-ESRGAN's U-Net discriminator with spectral normalisation (per-pixel real/fake logits)."""

    def __init__(self, c_in=3, c=64):
        super().__init__()
        sn = spectral_norm
        self.conv0 = nn.Conv2d(c_in, c, 3, 1, 1)
        self.conv1 = sn(nn.Conv2d(c, c * 2, 4, 2, 1, bias=False))
        self.conv2 = sn(nn.Conv2d(c * 2, c * 4, 4, 2, 1, bias=False))
        self.conv3 = sn(nn.Conv2d(c * 4, c * 8, 4, 2, 1, bias=False))
        self.conv4 = sn(nn.Conv2d(c * 8, c * 4, 3, 1, 1, bias=False))
        self.conv5 = sn(nn.Conv2d(c * 4, c * 2, 3, 1, 1, bias=False))
        self.conv6 = sn(nn.Conv2d(c * 2, c, 3, 1, 1, bias=False))
        self.conv7 = sn(nn.Conv2d(c, c, 3, 1, 1, bias=False))
        self.conv8 = sn(nn.Conv2d(c, c, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(c, 1, 3, 1, 1)

    def forward(self, x):
        act = lambda t: F.leaky_relu(t, 0.2, inplace=True)  # noqa: E731
        x0 = act(self.conv0(x))
        x1 = act(self.conv1(x0))
        x2 = act(self.conv2(x1))
        x3 = act(self.conv3(x2))
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
        x4 = act(self.conv4(up(x3, x2))) + x2
        x5 = act(self.conv5(up(x4, x1))) + x1
        x6 = act(self.conv6(up(x5, x0))) + x0
        return self.conv9(act(self.conv8(act(self.conv7(x6)))))


class TextFeatureLoss(nn.Module):
    """L1 between frozen EasyOCR recogniser features of the plate region (bounding box of the mask) in SR and HR."""

    def __init__(self, device, height=64, width=192):
        super().__init__()
        import easyocr

        reader = easyocr.Reader(["en"], gpu=device.type == "cuda", verbose=False)
        rec = reader.recognizer
        self.extractor = (rec.module if hasattr(rec, "module") else rec).FeatureExtraction.to(device).eval()
        for p in self.extractor.parameters():
            p.requires_grad_(False)
        self.size = (height, width)

    def _crops(self, x, boxes):
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).clamp(0, 1)
        crops = [F.interpolate(gray[i : i + 1, :, y0:y1, x0:x1], size=self.size, mode="bilinear", align_corners=False)
                 for i, (y0, y1, x0, x1) in boxes]
        return (torch.cat(crops) - 0.5) / 0.5  # EasyOCR input normalisation

    def forward(self, sr, hr, mask):
        """sr, hr: (N, 3, H, W) in [0, 1]; mask: (N, 1, H, W). Frames without a plate are skipped."""
        boxes = []
        for i in range(mask.shape[0]):
            ys, xs = torch.nonzero(mask[i, 0] > 0.5, as_tuple=True)
            if len(ys) >= 16:
                boxes.append((i, (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)))
        if not boxes:
            return sr.sum() * 0.0
        with torch.autocast(device_type=sr.device.type, enabled=False):
            f_sr = self.extractor(self._crops(sr.float(), boxes))
            with torch.no_grad():
                f_hr = self.extractor(self._crops(hr.float(), boxes))
            return F.l1_loss(f_sr, f_hr)


class AuxLosses:
    """Builds the enabled auxiliary losses from the ``train`` config; ``generator_loss`` adds them to the pixel
    loss, ``discriminator_step`` trains the discriminator (GAN only)."""

    def __init__(self, tcfg, device, world_size=1, local_rank=0):
        self.w_p = float(tcfg.get("perceptual_weight", 0.0))
        self.w_g = float(tcfg.get("gan_weight", 0.0))
        self.w_t = float(tcfg.get("text_weight", 0.0))
        self.frames = int(tcfg.get("aux_frames", 3))  # frames per clip used by the perceptual / GAN losses
        self.enabled = self.w_p > 0 or self.w_g > 0 or self.w_t > 0
        self.lpips = self.disc = self.text = None
        if self.w_p > 0:
            import lpips

            self.lpips = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
            for p in self.lpips.parameters():
                p.requires_grad_(False)
        if self.w_g > 0:
            disc = UNetDiscriminatorSN().to(device)
            self.disc_raw = disc
            if world_size > 1:
                disc = nn.parallel.DistributedDataParallel(disc, device_ids=[local_rank] if device.type == "cuda" else None)
            self.disc = disc
            self.d_opt = torch.optim.Adam(disc.parameters(), lr=float(tcfg.get("disc_lr", 1e-4)), betas=(0.9, 0.99))
        if self.w_t > 0:
            self.text = TextFeatureLoss(device)

    def _sample(self, x):
        n, t = x.shape[:2]
        idx = torch.linspace(0, t - 1, min(self.frames, t)).round().long()
        return x[:, idx].flatten(0, 1)

    def generator_loss(self, sr, hr, mask=None):
        """Weighted sum of the enabled losses and a dict of their values (for logging)."""
        total, logs = sr.new_zeros(()), {}
        if self.lpips is not None or self.disc is not None:
            s, h = self._sample(sr), self._sample(hr)
            if self.lpips is not None:
                lp = self.lpips(s * 2 - 1, h * 2 - 1).mean()
                total, logs["lpips"] = total + self.w_p * lp, lp.item()
            if self.disc is not None:
                for p in self.disc.parameters():
                    p.requires_grad_(False)
                g = F.binary_cross_entropy_with_logits(self.disc(s), torch.ones_like(s[:, :1]))
                total, logs["gan_g"] = total + self.w_g * g, g.item()
        if self.text is not None and mask is not None:
            tl = self.text(sr.flatten(0, 1), hr.flatten(0, 1), mask.flatten(0, 1))
            total, logs["text"] = total + self.w_t * tl, tl.item()
        return total, logs

    def discriminator_step(self, sr, hr):
        if self.disc is None:
            return {}
        for p in self.disc.parameters():
            p.requires_grad_(True)
        s, h = self._sample(sr.detach()).float(), self._sample(hr).float()
        self.d_opt.zero_grad(set_to_none=True)
        real = self.disc(h)
        fake = self.disc(s)
        loss = F.binary_cross_entropy_with_logits(real, torch.ones_like(real)) + \
            F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
        loss.backward()
        self.d_opt.step()
        return {"gan_d": loss.item()}

    def state_dict(self):
        return {"disc": self.disc_raw.state_dict(), "d_opt": self.d_opt.state_dict()} if self.disc is not None else {}

    def load_state_dict(self, state):
        if self.disc is not None and state.get("disc"):
            self.disc_raw.load_state_dict(state["disc"])
            self.d_opt.load_state_dict(state["d_opt"])
