"""Measure training throughput (seconds per iteration) and peak memory for training configs.

Uses random tensors of the training shape, so the result is the GPU-bound cost per iteration
(data loading must keep up separately: use >= 8 dataloader workers). Use it to size a cloud budget:

    python scripts/benchmark_train.py --configs configs/train_reds.yaml configs/ablation/*.yaml --out results/bench.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from forknet import ForkNet, ForkNetConfig  # noqa: E402
from forknet.losses import CharbonnierLoss  # noqa: E402
from forknet.utils import load_config  # noqa: E402


def bench(cfg, batch, frames, patch, ckpt, iters, warmup):
    model = ForkNet(ForkNetConfig.from_dict(cfg["model"])).cuda().train()
    model.set_grad_checkpointing(ckpt)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = CharbonnierLoss()
    lr = torch.rand(batch, frames, 3, patch, patch, device="cuda")
    hr = torch.rand(batch, frames, 3, patch * 4, patch * 4, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_fn(model(lr), hr)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters, torch.cuda.max_memory_allocated() / 1e9


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out")
    args = p.parse_args()
    gpu = torch.cuda.get_device_name(0)
    rows = {}
    for path in args.configs:
        cfg = load_config(path)
        batch, frames = cfg["train"]["batch_size"], cfg["data"]["train"]["num_frames"]
        patch = cfg["data"]["train"]["patch_size"]
        for ckpt in (False, True):
            key = f"{Path(path).stem}{' +ckpt' if ckpt else ''}"
            try:
                s, mem = bench(cfg, batch, frames, patch, ckpt, args.iters, args.warmup)
                rows[key] = {"sec_per_iter": s, "peak_mem_gb": mem, "batch": batch, "frames": frames,
                             "hours_per_100k": s * 1e5 / 3600}
                print(f"{key:40s} b{batch}x{frames}  {s:.3f} s/it  {mem:5.1f} GB  {s * 1e5 / 3600:6.1f} h/100k", flush=True)
            except torch.OutOfMemoryError:
                rows[key] = {"oom": True, "batch": batch, "frames": frames}
                print(f"{key:40s} b{batch}x{frames}  out of memory", flush=True)
            torch.cuda.empty_cache()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"gpu": gpu, "torch": torch.__version__, "runs": rows}, indent=2))


if __name__ == "__main__":
    main()
