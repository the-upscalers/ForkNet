"""RunPod: run a list of pilot configs on one pod, evaluate them, upload the results to Kaggle, then terminate.

Expects the code in the current directory (fetched by the pod's start command) and KAGGLE_API_TOKEN in the
environment (a RunPod secret). Data comes straight from Kaggle onto the container disk (~100 GB needed:
the 39 GB REDS zip is deleted after unpacking):

    python runpod/run_pilots.py --configs configs/pilot/main.yaml configs/pilot/kernel_mix.yaml \\
        --upload joelsathiyendra/forknet-runpod-pilots

Progress lines start with "[pilots]" so they can be read from the pod log. On success the pod removes itself
(runpodctl, RUNPOD_POD_ID); if the upload fails it keeps running for --hold-hours so results can be rescued.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
DATA = Path("/data")
OUT = Path("/workspace/out")
T0 = time.time()


def log(msg):
    print(f"[pilots] {(time.time() - T0) / 60:6.1f} min  {msg}", flush=True)


def sh(cmd, name):
    log(f"start {name}")
    t = time.time()
    rc = subprocess.run(["bash", "-o", "pipefail", "-c", cmd]).returncode
    log(f"done {name}: rc={rc} in {(time.time() - t) / 60:.1f} min")
    return rc


def first(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None


def fetch(slug, dest):
    """Bulk-download and unzip a Kaggle dataset, retrying on transient errors."""
    dest.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        if sh(f"kaggle datasets download {slug} -p {dest} --unzip 2>&1 | tr '\\r' '\\n' | grep -v '%|' | tail -3",
              f"download {slug} (try {attempt + 1})") == 0 and any(dest.iterdir()):
            return
        time.sleep(60 * (attempt + 1))
    raise SystemExit(f"could not download {slug}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--upload", required=True, help="private Kaggle dataset to create, owner/slug")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--hold-hours", type=float, default=2.0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    status = {}

    fetch("uom200647r/validation", DATA / "val")
    fetch("danniles/lpsr-dataset", DATA / "lpsr")
    fetch("uom200647r/reds-vsr", DATA / "reds")
    lr = first(f"{DATA}/reds/**/train_sharp_bicubic/X4")
    hr = first(f"{DATA}/reds/**/train/train_sharp")
    val = first(f"{DATA}/val/**/REDS_LR")
    val = str(Path(val).parent) if val else None
    lp = first(f"{DATA}/lpsr/**/ALPR_Dataset")
    log(f"data: lr={lr} hr={hr} val={val} lp={lp}")
    sh("df -h /data | tail -1; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader", "disk/gpu")

    for cfg_path in args.configs:
        name = Path(cfg_path).stem
        sys.path.insert(0, ".")
        from forknet.utils import load_config

        cfg = load_config(cfg_path)
        exp_name = cfg["name"]
        lr_root = "null" if cfg["data"]["train"].get("degradation") is not None else lr
        rc = sh(f"{PY} scripts/train.py --config {cfg_path} --set data.train.lr_root={lr_root} data.train.hr_root={hr} "
                f"data.val=null output_dir={OUT}/experiments train.num_workers={args.workers} train.batch_size=4 "
                f"train.grad_checkpoint=false 2>&1 | grep -E 'iter [0-9]+000/|Error|error|Traceback|Resumed|params'",
                f"train {name}")
        final = OUT / "experiments" / exp_name / "checkpoints" / "final.pth"
        status[name] = {"train_rc": rc, "finished": final.exists()}
        if final.exists():
            status[name]["reds4_rc"] = sh(
                f"{PY} scripts/test.py --config configs/test_reds4.yaml --checkpoint {final} --out {OUT}/results/{exp_name}_reds4 "
                f"--set data.test.lr_root={val}/REDS_LR data.test.hr_root={val}/REDS 2>&1 | grep -E '\\[standard\\]|Error'",
                f"eval REDS4 {name}")
            status[name]["lp_rc"] = sh(
                f"{PY} scripts/test.py --config configs/test_lp.yaml --checkpoint {final} --protocol both "
                f"--out {OUT}/results/{exp_name}_lp --set data.test.lr_root={lp}/test/X4 data.test.hr_root={lp}/test/sharp "
                f"data.test.split_file=splits/lp_lpsr.json data.test.split=test_clean 2>&1 | grep -E '\\[standard\\]|\\[legacy\\]|Error'",
                f"eval LP {name}")
        (OUT / "status.json").write_text(json.dumps(status, indent=2))
        print("[pilots] STATUS " + json.dumps(status), flush=True)

    # Keep the upload small: final checkpoints, logs, metrics (no images).
    up = Path("/workspace/upload")
    shutil.rmtree(up, ignore_errors=True)
    shutil.copytree(OUT, up, ignore=shutil.ignore_patterns("images", "latest.pth"))
    owner_slug = args.upload
    (up / "dataset-metadata.json").write_text(json.dumps(
        {"title": owner_slug.split("/")[1], "id": owner_slug, "licenses": [{"name": "other"}]}))
    rc = sh(f"kaggle datasets create -p {up} --dir-mode zip 2>&1 | tail -2 || "
            f"kaggle datasets version -p {up} -m 'runpod pilots' --dir-mode zip 2>&1 | tail -2", "upload to Kaggle")
    print("[pilots] FINAL " + json.dumps(status), flush=True)
    pod = os.environ.get("RUNPOD_POD_ID")
    if rc == 0 and pod:
        log("RUN_DONE, removing pod")
        subprocess.run(["runpodctl", "remove", "pod", pod])
    log(f"upload rc={rc}; holding {args.hold_hours} h for manual rescue")
    time.sleep(args.hold_hours * 3600)
    if pod:
        subprocess.run(["runpodctl", "stop", "pod", pod])


if __name__ == "__main__":
    main()
