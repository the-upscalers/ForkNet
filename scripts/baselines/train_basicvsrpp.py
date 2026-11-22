"""Fine-tune the official BasicVSR++ (REDS weights) with the kernel-mix degradation, using exactly the data pipeline,
loss and schedule of the Fork-Net kernel-mix fine-tune (configs/finetune_kernelmix_from_paper.yaml): 15-frame clips of
64x64 LR patches, batch 4, Charbonnier loss, Adam (0.9, 0.99), lr 1e-4 (SpyNet x0.125), cosine decay to 1e-7, AMP.
Checkpoints are written as {"params": ..., "iter": ...} and are read by scripts/baselines/basicvsrpp.py --weights.

    python scripts/baselines/train_basicvsrpp.py --hr-root data/REDS/train/train_sharp --out experiments/bvpp_km \\
        --iters 30000 --time-budget-h 6
"""

import argparse
import math
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from basicvsrpp import WEIGHTS_URL, BasicVSRPlusPlus, load_official_pp  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from forknet.data import VideoTrainDataset  # noqa: E402
from forknet.losses import CharbonnierLoss  # noqa: E402
from forknet.utils import load_config  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "configs" / "finetune_kernelmix_from_paper.yaml"),
                   help="Fork-Net config whose data.train section (degradation, clip length, patch size) is reused")
    p.add_argument("--hr-root", required=True)
    p.add_argument("--weights", default=str(ROOT / "weights" / Path(WEIGHTS_URL).name))
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=30000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--flow-lr-mult", type=float, default=0.125)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--save-every", type=int, default=10000)
    p.add_argument("--time-budget-h", type=float, default=None, help="stop (and save) when this many hours have passed")
    args = p.parse_args()

    t0 = time.time()
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    device = torch.device("cuda")
    if not Path(args.weights).exists():
        Path(args.weights).parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WEIGHTS_URL, args.weights)
    model = BasicVSRPlusPlus()
    model.load_state_dict(load_official_pp(args.weights), strict=True)
    model.to(device).train()

    dcfg = dict(load_config(args.config)["data"]["train"])
    dcfg.update(hr_root=args.hr_root, lr_root=None)
    ds = VideoTrainDataset(**dcfg)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=True)
    flow = [q for n, q in model.named_parameters() if n.startswith("spynet.")]
    rest = [q for n, q in model.named_parameters() if not n.startswith("spynet.")]
    opt = torch.optim.Adam([{"params": rest, "lr": args.lr}, {"params": flow, "lr": args.lr * args.flow_lr_mult}],
                           betas=(0.9, 0.99))
    eta_min = 1e-7

    def lam(base):
        return lambda it: (eta_min + 0.5 * (base - eta_min) * (1 + math.cos(math.pi * min(it, args.iters) / args.iters))) / base

    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lam(args.lr), lam(args.lr * args.flow_lr_mult)])
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = CharbonnierLoss(1e-8)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"BasicVSR++ {n_params / 1e6:.2f}M params; {len(ds)} samples; degradation {dcfg.get('degradation')}", flush=True)

    def save(name, it):
        torch.save({"params": model.state_dict(), "iter": it}, out / "checkpoints" / name)

    it, last = 0, time.time()
    while it < args.iters:
        for batch in dl:
            lrs, hrs = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                sr = model(lrs)
            loss = loss_fn(sr.float(), hrs)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            it += 1
            if it % 100 == 0:
                now = time.time()
                eta = (now - last) / 100 * (args.iters - it) / 3600
                print(f"iter {it}/{args.iters} loss {loss.item():.5f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(now - last) / 100:.2f}s/it eta {eta:.1f}h", flush=True)
                last = now
            if it % args.save_every == 0 and it < args.iters:
                save(f"iter{it}.pth", it)
            if args.time_budget_h and time.time() - t0 > args.time_budget_h * 3600:
                print(f"time budget reached at iteration {it}", flush=True)
                save(f"final_iter{it}.pth", it)
                return
            if it >= args.iters:
                break
    save(f"final_iter{it}.pth", it)
    print(f"done at iteration {it}", flush=True)


if __name__ == "__main__":
    main()
