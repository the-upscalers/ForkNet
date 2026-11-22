"""Parameter, FLOP and runtime measurement."""

import time

import torch
from torch.utils.flop_counter import FlopCounterMode


def param_counts(model):
    total = sum(p.numel() for p in model.parameters())
    spynet = sum(p.numel() for n, p in model.named_parameters() if n.startswith("spynet."))
    return {"params_total": total, "params_spynet": spynet, "params_without_spynet": total - spynet}


@torch.no_grad()
def flops_per_frame(model, lr_size=(180, 320), num_frames=7, device="cpu"):
    """FLOPs per output frame for an LR input of ``lr_size`` (REDS: 180x320).

    Counted by ``torch.utils.flop_counter`` (convolutions and matrix products, 1 MAC = 2 FLOPs;
    interpolation, warping and element-wise ops are not counted). BasicVSR / BasicVSR++ /
    IconVSR tables report mmcv ``get_flops`` values, which count MACs; compare with ``macs``.
    """
    x = torch.rand(1, num_frames, 3, *lr_size, device=device)
    counter = FlopCounterMode(display=False)
    with counter:
        model(x)
    flops = counter.get_total_flops() / num_frames
    return {"flops_per_frame": flops, "macs_per_frame": flops / 2, "lr_size": list(lr_size), "num_frames": num_frames}


@torch.no_grad()
def runtime_per_frame(model, lr_size=(180, 320), num_frames=5, device="cuda", warmup=2, repeats=5, amp=True):
    x = torch.rand(1, num_frames, 3, *lr_size, device=device)
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=amp):
        for _ in range(warmup):
            model(x)
        sync()
        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
        sync()
    seconds = (time.perf_counter() - start) / (repeats * num_frames)
    out = {"seconds_per_frame": seconds, "fps": 1.0 / seconds, "num_frames": num_frames, "amp": amp}
    if device.startswith("cuda"):
        out["gpu"] = torch.cuda.get_device_name(torch.device(device))
        out["max_memory_gb"] = torch.cuda.max_memory_allocated(torch.device(device)) / 1e9
    return out
