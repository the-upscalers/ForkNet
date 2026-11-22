"""Kaggle training kernel: one ~11 h session of Fork-Net training on REDS, resumable across sessions.

Attach: <owner>/forknet-code, uom200647r/reds-vsr, uom200647r/validation, and (to continue a previous
session) the previous version of this kernel's output, e.g. kernel source <owner>/forknet-train.
Environment variables (set at the top of the file when pushing):
    CONFIG      config file (default configs/train_reds.yaml)
    RUN_NAME    experiment name (default: config name)
    MAX_HOURS   training budget for this session (default 11)
    BATCH       batch size; "auto" probes 4 and falls back to 2 / 1 on out-of-memory
"""

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

CONFIG = os.environ.get("CONFIG", "configs/train_reds.yaml")
RUN_NAME = os.environ.get("RUN_NAME", "")
MAX_HOURS = float(os.environ.get("MAX_HOURS", 11))
BATCH = os.environ.get("BATCH", "auto")
W = Path("/kaggle/working")


def first(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def ds(slug):
    """Mounted path of an attached dataset; Kaggle mounts at /kaggle/input/<slug> or deeper (e.g. datasets/<owner>/<slug>)."""
    for depth in range(4):
        hits = sorted(glob.glob("/kaggle/input/" + "*/" * depth + slug))
        if hits:
            return hits[0]
    return None


code = Path(ds("forknet-code"))
CODE = W / "ForkNet"
if CODE.exists():
    shutil.rmtree(CODE)
shutil.copytree(code, CODE)
os.chdir(CODE)
subprocess.run(f"{sys.executable} -m pip install -q bsconv", shell=True)
subprocess.run("nvidia-smi", shell=True)

val = ds("validation")
reds = ds("reds-vsr")
paths = [
    f"data.train.lr_root={first([reds + '/REDS_LR/train/train_sharp_bicubic/X4', reds + '/**/train_sharp_bicubic/X4'])}",
    f"data.train.hr_root={first([reds + '/REDS/train/train_sharp', reds + '/**/train_sharp'])}",
    f"data.val.lr_root={val}/REDS_LR", f"data.val.hr_root={val}/REDS", "data.val.max_frames=50",
    f"output_dir={W}/experiments", "train.num_workers=4", "train.save_every=2000", "train.val_every=20000",
]
sys.path.insert(0, str(CODE))
from forknet.utils import load_config  # noqa: E402

cfg = load_config(CONFIG)
if cfg["data"]["train"].get("degradation") is not None:  # real-world model: LR is synthesised from HR
    paths.append("data.train.lr_root=null")
if RUN_NAME:
    paths.append(f"name={RUN_NAME}")
name = RUN_NAME or cfg["name"]

# Continue from a previous session's output if one is attached.
prev = first([f"/kaggle/input/**/experiments/{name}/checkpoints/latest.pth"])
if prev:
    prev_dir = Path(prev).parents[1]
    shutil.copytree(prev_dir, W / "experiments" / name, dirs_exist_ok=True)
    print(f"Resuming from {prev}", flush=True)


import torch  # noqa: E402

NGPU = torch.cuda.device_count()
# Kaggle's "T4 x2" machines: data-parallel over both GPUs (same quota cost as one); batch is the total.
DDP = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={NGPU}"]
LAUNCHERS = [DDP, [sys.executable]] if NGPU > 1 else [[sys.executable]]


def run(batch, ckpt, extra=(), launch=(sys.executable,)):
    cmd = [*launch, "scripts/train.py", "--config", CONFIG, "--resume", "auto", "--set", *paths,
           f"train.batch_size={batch}", f"train.grad_checkpoint={str(ckpt).lower()}", *extra]
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, capture_output=bool(extra), text=True)


# Keep the paper's batch size if possible: batch 4 plain, then with activation checkpointing, then smaller.
sizes = [(4, False), (4, True), (2, True), (1, True)] if BATCH == "auto" else [(int(BATCH), False), (int(BATCH), True)]
# Both GPUs first (falls back to one GPU if distributed training fails); batch must split evenly over the GPUs.
candidates = [(launch, b, c) for launch in LAUNCHERS for b, c in sizes if launch is not DDP or b % NGPU == 0]
for launch, batch, ckpt in candidates:
    probe = run(batch, ckpt, ["train.total_iters=3", "train.log_every=1", "train.max_hours=null", "name=_probe", "train.val_every=0"], launch)
    shutil.rmtree(W / "experiments" / "_probe", ignore_errors=True)
    if probe.returncode == 0:
        break
    print(f"batch {batch} (checkpointing={ckpt}, {'DDP' if launch is DDP else '1 GPU'}) failed:\n{probe.stderr[-2000:]}", flush=True)
else:
    raise SystemExit("Training does not fit even with batch 1 and checkpointing")
print(f"Using batch size {batch}, activation checkpointing {ckpt}, {'DDP over ' + str(NGPU) + ' GPUs' if launch is DDP else '1 GPU'}", flush=True)
paths.append(f"train.max_hours={MAX_HOURS}")
rc = run(batch, ckpt, launch=launch).returncode
# Keep the output small: drop the data copy of the code, keep experiments/.
shutil.rmtree(CODE, ignore_errors=True)
sys.exit(rc)
