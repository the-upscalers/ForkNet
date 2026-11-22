"""Kaggle kernel: GPU training-throughput benchmark of every training config (synthetic data, ~20 min)."""
import glob, os, shutil, subprocess, sys
from pathlib import Path


def ds(slug):
    for depth in range(4):
        hits = sorted(glob.glob("/kaggle/input/" + "*/" * depth + slug))
        if hits:
            return hits[0]


CODE = Path("/kaggle/working/ForkNet")
shutil.copytree(ds("forknet-code"), CODE, dirs_exist_ok=True)
os.chdir(CODE)
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
cfgs = " ".join(["configs/train_reds.yaml", "configs/train_reds_legacy.yaml"] + sorted(glob.glob("configs/ablation/*.yaml"))
                + sorted(glob.glob("configs/tradeoff/*.yaml")))
rc = subprocess.run(f"{sys.executable} scripts/benchmark_train.py --configs {cfgs} --out /kaggle/working/bench.json", shell=True).returncode
shutil.rmtree(CODE, ignore_errors=True)
sys.exit(rc)
