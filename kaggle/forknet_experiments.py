"""Kaggle experiments kernel: Tier-0 re-evaluation of the paper and a GPU test of the research framework.

Works with either account's datasets (found by searching /kaggle/input):
  code      forknet-code
  weights   forknet-weights (release format)  or  checkpoints-efficient-net-final + fine-tuned-forknet (original)
  LP data   forknet-lp-data/{lpsr,alpr_subset,real_lr_videos}  or  lpsr-dataset/ALPR_Dataset + alpr-vsr + lpr-val-videos
  VideoLQ   videolq-lr/VideoLQ  or  video-lq-dataset-new
  REDS4     validation (REDS / REDS_LR);  REDS train: reds-vsr (only for the training smoke tests)

Outputs: /kaggle/working/results/{summary.md, *.json, *.csv, logs/}. STAGES env var selects stages (default all):
  tests, reds4 (Fork-Net + bicubic) or reds4_forknet, lp (all LP methods), lp_basicvsr (BasicVSR only), alpr,
  robustness or robustness_reds4, labels, videolq, realvideos, smoke.
If a previous run of this kernel is attached (kernel source), its LP output images are reused for
lp_basicvsr / labels when the lp stage is not re-run.
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
TMP = Path("/kaggle/tmp")
STAGES = os.environ.get("STAGES", "tests,reds4,lp,robustness,labels,videolq,realvideos,smoke").split(",")
status = {}
PY = sys.executable


def sh(cmd, name=None):
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    if name:  # keep each step's output (the kernel log is not always downloadable)
        (RES / "logs").mkdir(exist_ok=True)
        cmd = f"( {cmd} ) 2>&1 | tee {RES}/logs/{name}.log"
    # pipefail: a failing command is not hidden by a trailing `| tail` / `| tee`
    rc = subprocess.run(["bash", "-o", "pipefail", "-c", cmd]).returncode
    if name:
        status[name] = {"rc": rc, "min": round((time.time() - t0) / 60, 1)}
        (RES / "status.json").write_text(json.dumps(status, indent=2))
    return rc


def find(*patterns):
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


# ------------------------------------------------------------------ setup
code = ds("forknet-code")
CODE = W / "ForkNet"
shutil.rmtree(CODE, ignore_errors=True)
shutil.copytree(code, CODE)
os.chdir(CODE)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv", "gpu")
sh(f"{PY} -m pip install -q bsconv torchmetrics scikit-image lpips pyiqa easyocr 'fast-plate-ocr[onnx]' open-image-models imageio-ffmpeg", "pip")

paper_ckpt = find(f"{ds('forknet-weights')}/forknet_reds_paper.pth", f"{ds('checkpoints-efficient-net-final')}/**/best_checkpoint.pth")
lp_ft_ckpt = find(f"{ds('forknet-weights')}/forknet_lp_finetuned_paper.pth", f"{ds('fine-tuned-forknet')}/**/best_finetuned_checkpoint.pth")
lpsr = (ds("forknet-lp-data") and ds("forknet-lp-data") + "/lpsr") or (ds("lpsr-dataset") and ds("lpsr-dataset") + "/ALPR_Dataset")
alpr = (ds("forknet-lp-data") and ds("forknet-lp-data") + "/alpr_subset") or (ds("alpr-vsr") and ds("alpr-vsr") + "/ALPR_Dataset")
real_videos = (ds("forknet-lp-data") and ds("forknet-lp-data") + "/real_lr_videos") or ds("lpr-val-videos")
videolq = (ds("videolq-lr") and ds("videolq-lr") + "/VideoLQ") or ds("video-lq-dataset-new")
val = ds("validation")
reds = ds("reds-vsr")
found = dict(paper_ckpt=paper_ckpt, lp_ft_ckpt=lp_ft_ckpt, lpsr=lpsr, alpr=alpr, real_videos=real_videos, videolq=videolq, val=val, reds=reds)
print(json.dumps(found, indent=2), flush=True)
(RES / "inputs.json").write_text(json.dumps(found, indent=2))
SPLITS = "splits/lp_lpsr.json"
methods = {"ForkNet-paper": paper_ckpt, "ForkNet-LPft-paper": lp_ft_ckpt}
# EXTRA_CKPTS="Name=file.pth,...": more checkpoints (found anywhere under /kaggle/input) scored like the paper ones.
for spec in filter(None, os.environ.get("EXTRA_CKPTS", "").split(",")):
    name, fname = spec.split("=", 1)
    methods[name] = find(f"/kaggle/input/**/{fname}")
    print(f"extra checkpoint {name}: {methods[name]}", flush=True)
LP_SPLITS = os.environ.get("LP_SPLITS", "test,test_clean").split(",")
BASICVSR = os.environ.get("SKIP_BASICVSR", "0") != "1"

# ------------------------------------------------------------------ 1. tests
if "tests" in STAGES:
    sh(f"{PY} -m pytest -q -x 2>&1 | tail -5 | tee {RES}/pytest.log", "tests")

# ------------------------------------------------------------------ 2. REDS4: re-derive the paper number, then the standard protocol
if ("reds4" in STAGES or "reds4_forknet" in STAGES) and paper_ckpt and val:
    reds4 = f"--set data.test.lr_root={val}/REDS_LR data.test.hr_root={val}/REDS"
    sh(f"{PY} scripts/test.py --config configs/test_reds4.yaml --checkpoint {paper_ckpt} --protocol both --out {RES}/reds4_forknet_paper {reds4}", "reds4_forknet_paper")
if "reds4" in STAGES and val:
    sh(f"{PY} scripts/test.py --config configs/test_reds4.yaml --bicubic matlab --protocol both --out {RES}/reds4_bicubic {reds4}", "reds4_bicubic")

# ------------------------------------------------------------------ 3. LP test split(s): every method, same protocol
lp_outputs = {}
if "lp" in STAGES and lpsr:
    for split in LP_SPLITS:
        lp = f"--set data.test.lr_root={lpsr}/test/X4 data.test.hr_root={lpsr}/test/sharp data.test.split_file={SPLITS} data.test.split={split}"
        for kernel in ("matlab", "torch"):
            sh(f"{PY} scripts/test.py --config configs/test_lp.yaml --bicubic {kernel} --protocol both --save-images "
               f"--out {RES}/lp_{split}_bicubic_{kernel} {lp}", f"lp_{split}_bicubic_{kernel}")
        for name, ckpt in methods.items():
            if ckpt:
                sh(f"{PY} scripts/test.py --config configs/test_lp.yaml --checkpoint {ckpt} --protocol both --save-images "
                   f"--out {RES}/lp_{split}_{name} {lp}", f"lp_{split}_{name}")


def lp_images(run):
    """Output images of an LP test run: this session's, else an attached previous run's."""
    here = RES / run / "images"
    return str(here) if here.exists() else find(f"/kaggle/input/**/results/{run}/images")


# BasicVSR (official REDS weights) on whole clips (as published) and on 15-frame chunks (as Fork-Net is evaluated):
# on aliased LR its forward recurrence diverges over long clips. Then LPIPS / tOF for every method on the clean split.
if ("lp" in STAGES or "lp_basicvsr" in STAGES) and lpsr:
    if BASICVSR:
        sh(f"{PY} scripts/baselines/basicvsr.py --lr-root {lpsr}/test/X4 --out {TMP}/lp_basicvsr", "lp_basicvsr_infer")
        sh(f"{PY} scripts/baselines/basicvsr.py --lr-root {lpsr}/test/X4 --chunk-size 15 --out {TMP}/lp_basicvsr_c15", "lp_basicvsr_c15_infer")
    lp_outputs = {"Bicubic": lp_images("lp_test_bicubic_matlab"), "BasicVSR": f"{TMP}/lp_basicvsr",
                  "BasicVSR-chunk15": f"{TMP}/lp_basicvsr_c15", **{n: lp_images(f"lp_test_{n}") for n, c in methods.items() if c}}
    lp_outputs = {n: d for n, d in lp_outputs.items() if d and Path(d).exists()}
    for name, d in lp_outputs.items():
        if "lp" in STAGES or name.startswith("BasicVSR"):
            sh(f"{PY} scripts/evaluate_outputs.py --sr {d} --gt {lpsr}/test/sharp --split-file {SPLITS} --split test_clean "
               f"--extra lpips tof --name {name} --out {RES}/lp_test_clean_full_{name}.json", f"lp_extra_{name}")

# The paper's LP table may have used the ALPR subset's test split: re-derive with the legacy protocol too.
if ("lp" in STAGES or "alpr" in STAGES) and lpsr:
    if alpr:
        a = f"--set data.test.lr_root={alpr}/test/X4 data.test.hr_root={alpr}/test/sharp data.test.split_file=null data.test.split=null"
        for name, ckpt in methods.items():
            if ckpt:
                sh(f"{PY} scripts/test.py --config configs/test_lp.yaml --checkpoint {ckpt} --protocol both --out {RES}/alpr_test_{name} {a}", f"alpr_{name}")
        sh(f"{PY} scripts/test.py --config configs/test_lp.yaml --bicubic matlab --protocol both --out {RES}/alpr_test_bicubic {a}", "alpr_bicubic")

# ------------------------------------------------------------------ 4. robustness sweep (LP clean test; REDS4 subset)
if "robustness" in STAGES and lpsr and paper_ckpt:
    sh(f"{PY} scripts/degradation_sweep.py --hr-root {lpsr}/test/sharp --split-file {SPLITS} --split test_clean "
       f"--cache {TMP}/sweep_lp --checkpoint {paper_ckpt} --name ForkNet-paper --bicubic --basicvsr --max-frames 40 "
       f"--out {RES}/robustness_lp.json 2>&1 | tail -25", "robustness_lp")
if ("robustness" in STAGES or "robustness_reds4" in STAGES) and paper_ckpt and val:
    sh(f"{PY} scripts/degradation_sweep.py --hr-root {val}/REDS --clips 000 011 015 020 --only clean down_cv2_cubic h264_crf28 combo_mild "
       f"--cache {TMP}/sweep_reds --checkpoint {paper_ckpt} --name ForkNet-paper --bicubic --basicvsr --max-frames 15 --amp "
       f"--out {RES}/robustness_reds4.json 2>&1 | tail -12", "robustness_reds4")

# ------------------------------------------------------------------ 5. plate labels and recognition accuracy
if "labels" in STAGES and lpsr:
    # Plates are detected on the HR frames (YOLOv9, open-image-models) and every method is read inside the same box.
    sh(f"{PY} scripts/lp/auto_label.py --hr-root {lpsr}/test/sharp --engines easyocr fast_plate_ocr --boxes {RES}/lp_test_boxes.json "
       f"--out {RES}/lp_test_labels.csv --sheets {RES}/label_sheets 2>&1 | tail -40", "lp_auto_label")
    if (RES / "lp_test_labels.csv").exists() and lp_outputs:
        specs = " ".join([f"HR={lpsr}/test/sharp"] + [f"{n}={d}" for n, d in lp_outputs.items() if Path(d).exists()])
        for engine in ("easyocr", "fast_plate_ocr"):
            sh(f"{PY} scripts/lp/ocr_eval.py --labels {RES}/lp_test_labels.csv --status auto_agree auto_single verified "
               f"--boxes {RES}/lp_test_boxes.json --reference Bicubic --method {specs} --engine {engine} "
               f"--out {RES}/lp_ocr_{engine}.json", f"lp_ocr_{engine}")

# ------------------------------------------------------------------ 6. VideoLQ (real-world, no GT): no-reference quality
if "videolq" in STAGES and videolq and paper_ckpt:
    vlq = {"forknet": paper_ckpt, **{n: c for n, c in methods.items() if c and n not in ("ForkNet-paper", "ForkNet-LPft-paper")}}
    for name, ckpt in vlq.items():
        sh(f"{PY} scripts/infer.py --checkpoint {ckpt} --input {videolq} --out {TMP}/videolq_{name} --chunk-size 5 --amp --tile 128 --max-frames 20", f"videolq_{name}")
    if BASICVSR:
        sh(f"{PY} scripts/baselines/basicvsr.py --lr-root {videolq} --out {TMP}/videolq_basicvsr --chunk-size 10 --max-frames 20", "videolq_basicvsr")
    for name in [*vlq, "basicvsr"] if BASICVSR else list(vlq):
        sh(f"{PY} scripts/eval_noref.py --sr {TMP}/videolq_{name} --every 5 --name {name} --out {RES}/videolq_{name}.json 2>&1 | tail -2", f"videolq_noref_{name}")
        shutil.rmtree(TMP / f"videolq_{name}", ignore_errors=True)

# ------------------------------------------------------------------ 7. real LR plate videos: qualitative + OCR readings
if "realvideos" in STAGES and real_videos and paper_ckpt:
    for name, ckpt in methods.items():
        if ckpt:
            sh(f"{PY} scripts/infer.py --checkpoint {ckpt} --input {real_videos} --out {RES}/real_videos_{name} --chunk-size 15 --video", f"real_{name}")

# ------------------------------------------------------------------ 8. GPU smoke tests: every training config, speed and memory
if "smoke" in STAGES and reds and val:
    tr = (f"data.train.lr_root={find(reds + '/REDS_LR/train/train_sharp_bicubic/X4')} data.train.hr_root={find(reds + '/REDS/train/train_sharp')} "
          f"data.val=null train.total_iters=40 train.log_every=20 train.val_every=0 train.num_workers=4 output_dir={TMP}/smoke")
    for cfg in sorted(glob.glob("configs/train_reds*.yaml") + glob.glob("configs/ablation/*.yaml") + glob.glob("configs/tradeoff/*.yaml")):
        name = Path(cfg).stem
        extra = "data.train.lr_root=null" if "real" in name else ""
        sh(f"{PY} scripts/train.py --config {cfg} --set {tr} {extra} name={name} 2>&1 | grep -E 'iter 40|Error|error' | tee -a {RES}/smoke.log", f"smoke_{name}")
    sh(f"{PY} scripts/complexity.py --runtime --device cuda --out {RES}/complexity.json --config configs/train_reds.yaml "
       f"configs/ablation/*.yaml 2>&1 | tee {RES}/complexity.log", "complexity")

# ------------------------------------------------------------------ summary
tables = sorted(str(p) for p in RES.glob("*/metrics.json")) + sorted(str(p) for p in RES.glob("lp_test_clean_full_*.json"))
if tables:
    sh(f"{PY} scripts/make_tables.py {' '.join(tables)} --out {RES}/tables.md")
(RES / "summary.md").write_text("# Experiments\n\n```json\n" + json.dumps(status, indent=2) + "\n```\n\n" +
                                ((RES / "tables.md").read_text() if (RES / "tables.md").exists() else ""))
shutil.rmtree(CODE, ignore_errors=True)
print((RES / "summary.md").read_text())
