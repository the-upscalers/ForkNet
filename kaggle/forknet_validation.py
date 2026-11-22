"""Kaggle validation run for the Fork-Net repository.

Datasets attached to the notebook:
    <owner>/forknet-code   this repository (uploaded with kaggle/push_code.sh)
    uom200647r/validation  REDS4 HR (REDS/000..020) and LR (REDS_LR/000..020)
    uom200647r/reds-vsr    REDS train_sharp (HR) and train_sharp_bicubic/X4 (LR)

Results are written to /kaggle/working/results (summary.md + JSON files).
Environment variables: TRAIN_ITERS (default 4000), SKIP_TRAIN=1 to skip the training comparison.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

W = Path("/kaggle/working")
RES = W / "results"
RES.mkdir(parents=True, exist_ok=True)
TRAIN_ITERS = int(os.environ.get("TRAIN_ITERS", 4000))
status = {}


def sh(cmd, name=None, check=False):
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, shell=True).returncode
    if name:
        status[name] = {"returncode": rc, "minutes": round((time.time() - t0) / 60, 2)}
        (RES / "status.json").write_text(json.dumps(status, indent=2))
    if check and rc:
        raise SystemExit(f"{cmd} failed with {rc}")
    return rc


def find_dir(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(patterns)


# --------------------------------------------------------------------------- setup
code_src = Path(find_dir(["/kaggle/input/forknet-code", "/kaggle/input/*/forknet-code", "/kaggle/input/*/*/forknet-code"]))
if not (code_src / "forknet").exists():
    code_src = Path(find_dir([f"{code_src}/*/forknet/.."]))
CODE = W / "ForkNet"
if CODE.exists():
    shutil.rmtree(CODE)
shutil.copytree(code_src, CODE)
os.chdir(CODE)
sh("nvidia-smi", "nvidia-smi")
sh(f"{sys.executable} -m pip install -q bsconv torchmetrics scikit-image", "pip")
sh(f"{sys.executable} -c \"import torch; print(torch.__version__, torch.cuda.get_device_name(0))\"")

val_root = find_dir(["/kaggle/input/validation", "/kaggle/input/*/validation", "/kaggle/input/*/*/validation"])
REDS4_HR, REDS4_LR = f"{val_root}/REDS", f"{val_root}/REDS_LR"
reds_root = find_dir(["/kaggle/input/reds-vsr", "/kaggle/input/*/reds-vsr", "/kaggle/input/*/*/reds-vsr"])
TRAIN_HR = find_dir([f"{reds_root}/REDS/train/train_sharp", f"{reds_root}/*/train/train_sharp", f"{reds_root}/**/train_sharp"])
TRAIN_LR = find_dir([f"{reds_root}/REDS_LR/train/train_sharp_bicubic/X4", f"{reds_root}/**/train_sharp_bicubic/X4"])
print({"REDS4_HR": REDS4_HR, "REDS4_LR": REDS4_LR, "TRAIN_HR": TRAIN_HR, "TRAIN_LR": TRAIN_LR}, flush=True)
REDS4 = "['000','011','015','020']"
test_set = [f"data.test.lr_root={REDS4_LR}", f"data.test.hr_root={REDS4_HR}"]

# --------------------------------------------------------------------------- 1. tests
# pixel-level-vsr is private: the clone only works with credentials; otherwise the equivalence tests are skipped.
sh("GIT_TERMINAL_PROMPT=0 git clone -q -b fork-net https://github.com/the-upscalers/pixel-level-vsr /kaggle/tmp/pixel-level-vsr")
sh(f"PIXEL_LEVEL_VSR_ROOT=/kaggle/tmp/pixel-level-vsr {sys.executable} -m pytest -q -rs --junitxml={RES}/pytest.xml 2>&1 | tee {RES}/pytest.log",
   "pytest")
sh(f"tail -3 {RES}/pytest.log")

# --------------------------------------------------------------------------- 2. complexity
sh(f"{sys.executable} scripts/complexity.py --runtime --device cuda --out {RES}/complexity.json 2>&1 | tee {RES}/complexity.log",
   "complexity")

# --------------------------------------------------------------------------- 3. bicubic baselines on REDS4
for kernel in ("matlab", "torch"):
    sh(f"{sys.executable} scripts/test.py --config configs/test_reds4.yaml --bicubic {kernel} --protocol both "
       f"--out {RES}/reds4_bicubic_{kernel} --set {' '.join(test_set)}", f"bicubic_{kernel}")

# --------------------------------------------------------------------------- 4. BasicVSR baseline (validates metric code)
sh(f"{sys.executable} scripts/baselines/basicvsr.py --lr-root {REDS4_LR} --clips 000 011 015 020 --out /kaggle/tmp/basicvsr_reds4",
   "basicvsr_infer")
sh(f"{sys.executable} scripts/evaluate_outputs.py --sr /kaggle/tmp/basicvsr_reds4 --gt {REDS4_HR} --clips 000 011 015 020 "
   f"--name BasicVSR --out {RES}/reds4_basicvsr.json", "basicvsr_eval")

# --------------------------------------------------------------------------- 5. short training: corrected vs legacy propagation
if not os.environ.get("SKIP_TRAIN"):
    common = [
        f"data.train.lr_root={TRAIN_LR}", f"data.train.hr_root={TRAIN_HR}",
        f"data.val.lr_root={REDS4_LR}", f"data.val.hr_root={REDS4_HR}", "data.val.max_frames=30",
        f"train.total_iters={TRAIN_ITERS}", "train.num_workers=4", "train.val_every=0",
        f"train.save_every={TRAIN_ITERS}", "train.log_every=100", f"output_dir={W}/experiments",
    ]
    for cfg_name, cfg_file in (("corrected", "configs/train_reds.yaml"), ("legacy_prop", "configs/ablation/legacy_propagation.yaml")):
        sh(f"{sys.executable} scripts/train.py --config {cfg_file} --set name=short_{cfg_name} {' '.join(common)} "
           f"2>&1 | tee {RES}/train_{cfg_name}.log", f"train_{cfg_name}")
        ckpt = W / "experiments" / f"short_{cfg_name}" / "checkpoints" / "final.pth"
        sh(f"{sys.executable} scripts/test.py --config configs/test_reds4.yaml --checkpoint {ckpt} --protocol both "
           f"--out {RES}/reds4_short_{cfg_name} --set {' '.join(test_set)}", f"test_{cfg_name}")
        shutil.copy(W / "experiments" / f"short_{cfg_name}" / "train_log.jsonl", RES / f"train_{cfg_name}.jsonl")

# --------------------------------------------------------------------------- summary
files = sorted(str(p) for p in RES.glob("reds4_*/metrics.json")) + [str(RES / "reds4_basicvsr.json")]
files = [f for f in files if Path(f).exists()]
if files:
    sh(f"{sys.executable} scripts/make_tables.py {' '.join(files)} --out {RES}/tables.md")
lines = ["# Fork-Net Kaggle validation", "", "## Step status", "", "```json", json.dumps(status, indent=2), "```", ""]
if (RES / "tables.md").exists():
    lines += ["## REDS4 results", "", (RES / "tables.md").read_text()]
if (RES / "complexity.log").exists():
    lines += ["## Complexity", "", "```", (RES / "complexity.log").read_text(), "```"]
if (RES / "pytest.log").exists():
    lines += ["## Tests", "", "```", "\n".join((RES / "pytest.log").read_text().splitlines()[-15:]), "```"]
(RES / "summary.md").write_text("\n".join(lines))
print("\n".join(lines))
