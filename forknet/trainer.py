"""Iteration-based trainer (single GPU, or multi-GPU via ``torchrun``)."""

import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .archs import ForkNet, ForkNetConfig
from .data import VideoTestDataset, VideoTrainDataset
from .evaluation import evaluate
from .aux_losses import AuxLosses
from .losses import CharbonnierLoss
from .utils.telemetry import Telemetry
from .utils import (
    append_jsonl,
    environment_info,
    get_logger,
    load_checkpoint,
    save_checkpoint,
    save_config,
    seed_worker,
    set_seed,
)


PAPER_CHECKPOINTS = (
    "weights/forknet_reds_paper.pth",
    "/kaggle/input/**/forknet_reds_paper.pth",
    "/kaggle/input/**/checkpoints_efficient_net_final_cont_cont/best_checkpoint.pth",
)


def resolve_pretrained(spec):
    """``auto:paper`` -> the paper's REDS checkpoint (release or original format) wherever it is available;
    ``runpod:<path>`` -> a checkpoint uploaded by a RunPod run, looked up by file name among attached Kaggle inputs."""
    import glob

    root = Path(__file__).resolve().parents[1]
    if str(spec).startswith("runpod:"):
        name = Path(spec.split(":", 1)[1]).name
        hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
        if not hits:
            raise FileNotFoundError(f"{spec}: {name} not found under /kaggle/input (pass pretrained=<path> instead)")
        return hits[0]
    if spec != "auto:paper":
        return spec
    for pattern in PAPER_CHECKPOINTS:
        hits = sorted(glob.glob(pattern if pattern.startswith("/") else str(root / pattern), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Paper checkpoint not found; looked for {PAPER_CHECKPOINTS}")


def cosine_lambda(base_lr, eta_min, total_iters):
    """LR multiplier equal to BasicSR's CosineAnnealingRestartLR with a single period."""
    return lambda it: (eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * min(it, total_iters) / total_iters))) / base_lr


class Trainer:
    def __init__(self, cfg, resume=None):
        self.cfg = cfg
        self.tcfg = cfg["train"]
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if self.world_size > 1:
            # Long timeout: the other ranks wait in the next all-reduce while rank 0 validates or saves.
            from datetime import timedelta

            dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(hours=2))
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        self.is_main = self.rank == 0

        self.exp_dir = Path(cfg.get("output_dir", "experiments")) / cfg["name"]
        self.log = get_logger(log_file=self.exp_dir / "train.log", rank=self.rank)
        # Live dashboard (Weights & Biases) when WANDB_API_KEY is set; otherwise a no-op.
        self.telemetry = Telemetry(cfg, self.exp_dir, enabled=self.is_main)
        set_seed(cfg.get("seed", 0) + self.rank, cfg.get("deterministic", False))
        if self.is_main:
            save_config(cfg, self.exp_dir / "config.yaml")
            self.log.info(f"Environment: {environment_info()}")

        self.model_cfg = ForkNetConfig.from_dict(cfg.get("model", {}))
        model = ForkNet(self.model_cfg, spynet_path=cfg.get("spynet_path"))
        if cfg.get("pretrained"):
            cfg["pretrained"] = resolve_pretrained(cfg["pretrained"])
            sd, pre_cfg, _ = load_checkpoint(cfg["pretrained"])
            model.load_state_dict(sd, strict=True)
            self.log.info(f"Initialised from {cfg['pretrained']} (architecture {pre_cfg})")
        model.to(self.device)
        if self.tcfg.get("grad_checkpoint"):
            model.set_grad_checkpointing(True)
            self.log.info("Activation checkpointing enabled")
        self.log.info(f"ForkNet {self.model_cfg}: {model.num_params() / 1e6:.3f}M params "
                      f"({model.num_params(False) / 1e6:.3f}M without SpyNet)")

        flow_params = [p for n, p in model.named_parameters() if n.startswith("spynet.")]
        other_params = [p for n, p in model.named_parameters() if not n.startswith("spynet.")]
        lr, flow_lr = self.tcfg["lr"], self.tcfg["lr"] * self.tcfg.get("flow_lr_mult", 0.125)
        self.optimizer = torch.optim.Adam(
            [{"params": other_params, "lr": lr}, {"params": flow_params, "lr": flow_lr}],
            betas=tuple(self.tcfg.get("betas", (0.9, 0.99))),
            weight_decay=self.tcfg.get("weight_decay", 0.0),
        )
        total, eta_min = self.tcfg["total_iters"], self.tcfg.get("eta_min", 1e-7)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, [cosine_lambda(lr, eta_min, total), cosine_lambda(flow_lr, eta_min, total)]
        )
        self.amp = self.tcfg.get("amp", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.loss_fn = CharbonnierLoss(self.tcfg.get("loss_eps", 1e-8))
        # Optional perceptual / adversarial / plate-text losses (forknet/aux_losses.py); off by default.
        self.aux = AuxLosses(self.tcfg, self.device, self.world_size, self.local_rank)

        self.raw_model = model
        self.model = model
        if self.world_size > 1:
            self.model = DistributedDataParallel(
                model, device_ids=[self.local_rank] if self.device.type == "cuda" else None, find_unused_parameters=self.tcfg.get("fix_flow_iters", 0) > 0
            )

        self.start_iter = 0
        self.best_psnr = -float("inf")
        if resume:
            self._resume(resume)
        self._build_data()

    def _build_data(self):
        dcfg = self.cfg["data"]
        self.train_set = VideoTrainDataset(**dcfg["train"])
        per_gpu = self.tcfg["batch_size"] // self.world_size
        if per_gpu * self.world_size != self.tcfg["batch_size"]:
            raise ValueError("train.batch_size must be divisible by the number of GPUs")
        self.sampler = DistributedSampler(self.train_set, shuffle=True, seed=self.cfg.get("seed", 0)) if self.world_size > 1 else None
        generator = torch.Generator()
        generator.manual_seed(self.cfg.get("seed", 0) + self.rank)
        # train.num_workers is the total over all GPUs (the machine's CPUs are shared).
        workers = self.tcfg.get("num_workers", 4)
        workers = max(1, workers // self.world_size) if workers > 0 else 0
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=per_gpu,
            shuffle=self.sampler is None,
            sampler=self.sampler,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=generator,
            persistent_workers=workers > 0,
        )
        self.val_set = VideoTestDataset(**dcfg["val"]) if dcfg.get("val") else None
        self.log.info(f"Train samples: {len(self.train_set)} from {len(self.train_set.clips)} clips; "
                      f"val clips: {len(self.val_set) if self.val_set else 0}")

    def _resume(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.raw_model.load_state_dict(ckpt["params"], strict=True)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.start_iter = ckpt["iter"]
        self.best_psnr = ckpt.get("best_psnr", self.best_psnr)
        self.aux.load_state_dict(ckpt.get("aux", {}))
        self.log.info(f"Resumed from {path} at iteration {self.start_iter}")

    def _save(self, name, it):
        if not self.is_main:
            return
        save_checkpoint(
            self.exp_dir / "checkpoints" / name,
            self.raw_model,
            self.model_cfg,
            iter=it,
            best_psnr=self.best_psnr,
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            scaler=self.scaler.state_dict(),
            aux=self.aux.state_dict(),
        )

    def _set_flow_trainable(self, trainable):
        for p in self.raw_model.spynet.parameters():
            p.requires_grad_(trainable)

    def validate(self, it):
        if not (self.is_main and self.val_set):
            return None
        vcfg = self.cfg.get("val", {})
        self.raw_model.eval()
        res = evaluate(self.raw_model, self.val_set, self.device, vcfg.get("chunk_size", 15), vcfg.get("overlap", 0), self.amp)
        self.raw_model.train()
        avg = res["average"]
        self.log.info(f"[val] iter {it}: PSNR {avg['psnr_rgb']:.4f} dB  SSIM {avg['ssim_rgb']:.4f}")
        append_jsonl(self.exp_dir / "val_log.jsonl", {"iter": it, **avg, "clips": res["clips"]})
        self.telemetry.log({f"val/{k}": v for k, v in avg.items()}, step=it)
        if avg["psnr_rgb"] > self.best_psnr:
            self.best_psnr = avg["psnr_rgb"]
            self._save("best.pth", it)
        return avg

    def _out_of_time(self, deadline):
        """Whether the wall-clock budget is spent; all ranks agree, so they stop at the same iteration."""
        if deadline is None:
            return False
        done = time.time() > deadline
        if self.world_size > 1:
            flag = torch.tensor(float(done), device=self.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            done = bool(flag.item())
        return done

    def fit(self):
        total = self.tcfg["total_iters"]
        fix_flow = self.tcfg.get("fix_flow_iters", 0)
        log_every, val_every, save_every = (self.tcfg.get(k, d) for k, d in (("log_every", 100), ("val_every", 5000), ("save_every", 5000)))
        grad_clip = self.tcfg.get("grad_clip")
        # Wall-clock budget: stop cleanly (and save) before e.g. Kaggle's 12-hour session limit.
        deadline = time.time() + self.tcfg["max_hours"] * 3600 if self.tcfg.get("max_hours") else None
        it, epoch = self.start_iter, 0
        self.model.train()
        tic, running = time.time(), 0.0
        while it < total:
            if self._out_of_time(deadline):
                break
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            for batch in self.train_loader:
                lrs, hrs = batch[0], batch[1]
                mask = batch[2].to(self.device, non_blocking=True) if len(batch) > 2 else None
                it += 1
                if it > total:
                    break
                self._set_flow_trainable(it > fix_flow)
                lrs, hrs = lrs.to(self.device, non_blocking=True), hrs.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                    sr = self.model(lrs)
                    loss = self.loss_fn(sr, hrs)
                    aux_logs = {}
                    if self.aux.enabled:
                        extra, aux_logs = self.aux.generator_loss(sr, hrs, mask)
                        loss = loss + extra
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at iteration {it}")
                self.scaler.scale(loss).backward()
                if grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                aux_logs.update(self.aux.discriminator_step(sr, hrs))
                running += loss.item()

                if it % log_every == 0 and self.is_main:
                    elapsed = time.time() - tic
                    rec = {"iter": it, "loss": running / log_every, "lr": self.scheduler.get_last_lr()[0],
                           "sec_per_iter": elapsed / log_every, "flow_trainable": it > fix_flow, **aux_logs}
                    eta_h = (total - it) * rec["sec_per_iter"] / 3600
                    self.log.info(f"iter {it}/{total} loss {rec['loss']:.5f} lr {rec['lr']:.2e} "
                                  f"{rec['sec_per_iter']:.2f}s/it eta {eta_h:.1f}h")
                    append_jsonl(self.exp_dir / "train_log.jsonl", rec)
                    self.telemetry.log({f"train/{k}": float(v) for k, v in rec.items() if k != "iter"}, step=it)
                    tic, running = time.time(), 0.0
                if it % save_every == 0:
                    self._save("latest.pth", it)
                if val_every and it % val_every == 0:
                    self.validate(it)
                if it >= total:
                    break
                if self._out_of_time(deadline):
                    self.log.info(f"Time budget of {self.tcfg['max_hours']} h reached at iteration {it}; stopping")
                    break
            epoch += 1
        self._save("latest.pth", it)
        if it >= total:
            self._save("final.pth", it)
        self.telemetry.finish()
        if self.world_size > 1:
            dist.destroy_process_group()
        return it
