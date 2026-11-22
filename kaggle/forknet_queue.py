"""Kaggle queue kernel: train a list of configs one after another within one session, evaluate each finished model.

Set QUEUE (semicolon-separated config paths) and MAX_HOURS at the top when pushing (env vars are not available on Kaggle).
For every config:
  * resume from a previous session's output if attached (experiments/<name>/checkpoints/latest.pth);
  * probe the batch size (4 plain -> 4 + activation checkpointing -> 2 -> 1);
  * train within its share of the remaining time;
  * evaluate final.pth (or latest.pth, tagged _partial) on LP-test-clean (standard + legacy protocol) and REDS4.
LP configs (data paths under data/lp/lpsr) are pointed at the LP dataset, all others at REDS.
Results: /kaggle/working/experiments/<name>/ and /kaggle/working/results/<name>_{reds4,lp}/metrics.json.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

QUEUE = os.environ.get("QUEUE", "configs/finetune_corrected_from_paper.yaml").split(";")
MAX_HOURS = float(os.environ.get("MAX_HOURS", 11.3))
EVAL_RESERVE_H = 0.4  # per model
T0 = time.time()
W = Path("/kaggle/working")
PY = sys.executable


def ds(slug):
    for depth in range(4):
        hits = sorted(glob.glob("/kaggle/input/" + "*/" * depth + slug))
        if hits:
            return hits[0]
    return None


def first(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def hours_left():
    return MAX_HOURS - (time.time() - T0) / 3600


CODE = W / "ForkNet"
shutil.copytree(ds("forknet-code"), CODE, dirs_exist_ok=True)
os.chdir(CODE)
subprocess.run(f"{PY} -m pip install -q bsconv", shell=True)
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
sys.path.insert(0, str(CODE))
from forknet.utils import load_config  # noqa: E402

val, reds = ds("validation"), ds("reds-vsr")
lpsr = (ds("forknet-lp-data") and ds("forknet-lp-data") + "/lpsr") or (ds("lpsr-dataset") and ds("lpsr-dataset") + "/ALPR_Dataset")
common = [f"output_dir={W}/experiments", "train.num_workers=4", "train.save_every=2000"]
reds_paths = [
    f"data.train.lr_root={first([reds + '/REDS_LR/train/train_sharp_bicubic/X4', reds + '/**/train_sharp_bicubic/X4'])}",
    f"data.train.hr_root={first([reds + '/REDS/train/train_sharp', reds + '/**/train_sharp'])}",
    f"data.val.lr_root={val}/REDS_LR", f"data.val.hr_root={val}/REDS", "data.val.max_frames=30",
] if reds and val else []


def data_overrides(cfg):
    """Kaggle paths for the config's training data: LP configs (data/lp/lpsr/...) -> the LP dataset, otherwise REDS."""
    if str(cfg["data"]["train"].get("hr_root", "")).startswith("data/lp/lpsr"):
        return [f"data.{part}.{kind}_root={cfg['data'][part][kind + '_root'].replace('data/lp/lpsr', lpsr)}"
                for part in ("train", "val") if cfg["data"].get(part) for kind in ("lr", "hr")]
    return list(reds_paths)
status = {}


def log_status():
    (W / "results").mkdir(parents=True, exist_ok=True)
    (W / "results" / "queue_status.json").write_text(json.dumps(status, indent=2))


import torch  # noqa: E402

NGPU = torch.cuda.device_count()
# Kaggle's "T4 x2" machines: data-parallel over both GPUs (same quota cost as one); batch is the total.
DDP = [PY, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={NGPU}"]
LAUNCHERS = [DDP, [PY]] if NGPU > 1 else [[PY]]
SIZES = [(4, False), (4, True), (2, True), (1, True)]
if os.environ.get("GRAD_CKPT") == "1":  # the 3-iteration probe can pass and a later batch still run out of memory
    SIZES = [s for s in SIZES if s[1]]


def train_cmd(cfg_path, overrides, batch, ckpt, extra=(), launch=(PY,)):
    return [*launch, "scripts/train.py", "--config", cfg_path, "--resume", "auto", "--set", *overrides,
            f"train.batch_size={batch}", f"train.grad_checkpoint={str(ckpt).lower()}", *extra]


for i, cfg_path in enumerate(QUEUE):
    cfg = load_config(cfg_path)
    name = cfg["name"]
    overrides = data_overrides(cfg) + common
    if cfg["data"]["train"].get("degradation") is not None:
        overrides.append("data.train.lr_root=null")
    exp = W / "experiments" / name
    prev = first([f"/kaggle/input/**/experiments/{name}/checkpoints/latest.pth"])
    if prev and not exp.exists():
        shutil.copytree(Path(prev).parents[1], exp, dirs_exist_ok=True)
        print(f"[{name}] resuming from {prev}", flush=True)
    final = exp / "checkpoints" / "final.pth"
    if not final.exists():
        share = (hours_left() - EVAL_RESERVE_H * (len(QUEUE) - i)) / (len(QUEUE) - i)
        if share < 0.3:
            status[name] = "skipped: no time left"
            log_status()
            continue
        # Both GPUs first (falls back to one GPU if distributed training fails); batch must split evenly over the GPUs.
        for launch, batch, ckpt in [(l, b, c) for l in LAUNCHERS for b, c in SIZES if l is not DDP or b % NGPU == 0]:
            probe = subprocess.run(train_cmd(cfg_path, overrides, batch, ckpt,
                                             ["train.total_iters=3", "train.max_hours=null", "train.val_every=0",
                                              "name=_probe", "pretrained=null"], launch), capture_output=True, text=True)
            shutil.rmtree(W / "experiments" / "_probe", ignore_errors=True)
            if probe.returncode == 0:
                break
            print(f"[{name}] batch {batch} ckpt={ckpt} {'DDP' if launch is DDP else '1 GPU'} failed: {probe.stderr[-600:]}", flush=True)
        gpus = NGPU if launch is DDP else 1
        print(f"[{name}] training with batch {batch}, checkpointing {ckpt}, {gpus} GPU(s), budget {share:.2f} h", flush=True)
        t = time.time()
        rc = subprocess.run(train_cmd(cfg_path, overrides, batch, ckpt, [f"train.max_hours={share:.3f}"], launch)).returncode
        status[name] = {"train_rc": rc, "train_hours": round((time.time() - t) / 3600, 2), "batch": batch, "grad_checkpoint": ckpt,
                        "gpus": gpus, "finished": final.exists()}
        log_status()
    # Evaluate the final model, or the latest checkpoint of an unfinished run (results tagged "_partial").
    latest = exp / "checkpoints" / "latest.pth"
    ckpt_eval, tag = (final, "") if final.exists() else (latest, "_partial")
    if ckpt_eval.exists():
        res = W / "results"
        if lpsr:
            subprocess.run(f"{PY} scripts/test.py --config configs/test_lp.yaml --checkpoint {ckpt_eval} --protocol both --save-images "
                           f"--out {res}/{name}_lp{tag} --set data.test.lr_root={lpsr}/test/X4 data.test.hr_root={lpsr}/test/sharp "
                           f"data.test.split_file=splits/lp_lpsr.json data.test.split=test_clean", shell=True)
            # 15-frame chunks too: recurrent models trained on short clips can diverge on long LP clips
            subprocess.run(f"{PY} scripts/test.py --config configs/test_lp.yaml --checkpoint {ckpt_eval} --out {res}/{name}_lp_chunk15{tag} "
                           f"--set data.test.lr_root={lpsr}/test/X4 data.test.hr_root={lpsr}/test/sharp "
                           f"data.test.split_file=splits/lp_lpsr.json data.test.split=test_clean test.chunk_size=15", shell=True)
        if val:
            subprocess.run(f"{PY} scripts/test.py --config configs/test_reds4.yaml --checkpoint {ckpt_eval} --out {res}/{name}_reds4{tag} "
                           f"--set data.test.lr_root={val}/REDS_LR data.test.hr_root={val}/REDS", shell=True)
        log_status()

shutil.rmtree(CODE, ignore_errors=True)
print(json.dumps(status, indent=2))
