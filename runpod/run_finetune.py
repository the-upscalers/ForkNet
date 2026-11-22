"""RunPod: budget-capped fine-tuning with a checkpoint, evaluation and Kaggle upload every N iterations.

For each config (run in order): train in the background; every ``--eval-every`` iterations snapshot the
checkpoint (weights only), evaluate it (REDS4 first 30 frames per clip + LP test_clean), and push the
snapshots and metrics to a private Kaggle dataset. Training of the last run is stopped (cleanly, by the
trainer's time budget) so that the whole pod stays under ``--budget-usd``; the final checkpoint gets the
full REDS4 evaluation. The pod removes itself at the end, so a stop at any point loses at most N iterations.

    python runpod/run_finetune.py --configs configs/finetune_corrected_from_paper.yaml \\
        --upload joelsathiyendra/forknet-runpod-finetune --budget-usd 10 --price-per-hour 0.59
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402
from run_pilots import DATA, PY, fetch, first, log, sh  # noqa: E402

from forknet.utils.telemetry import Telemetry  # noqa: E402

OUT = Path("/workspace/out")
UP = Path("/workspace/upload")
EVAL_RESERVE_H = 0.5  # final full evaluation + upload + slack


def spent(args, t_start):
    return args.spent_usd + (time.time() - t_start) / 3600 * args.price_per_hour


def upload(slug, first_time):
    """Create the dataset on the first call, new versions afterwards. The CLI can exit 1 after a successful
    create, and a repeated create only prints "already in use" (exit 0), so fall back to a version then."""
    (UP / "dataset-metadata.json").write_text(json.dumps({"title": slug.split("/")[1], "id": slug, "licenses": [{"name": "other"}]}))
    if first_time:
        out = UP.parent / "upload_create.txt"
        rc = sh(f"kaggle datasets create -p {UP} --dir-mode zip 2>&1 | tail -2 | tee {out}", "upload (create)")
        if "already in use" not in out.read_text():
            return rc
        # A new version replaces all files: merge in what earlier pods uploaded (ours win on name clashes).
        prev = UP.parent / "upload_prev"
        sh(f"kaggle datasets download {slug} -p {prev} --unzip 2>&1 | tail -1", "download earlier uploads")
        if prev.exists():
            shutil.copytree(prev, UP, dirs_exist_ok=True, ignore=lambda d, names: [n for n in names if (UP / Path(d).relative_to(prev) / n).is_file()])
    return sh(f"kaggle datasets version -p {UP} -m 'snapshot {time.strftime('%H:%M')}' --dir-mode zip 2>&1 | tail -2", "upload")


def evaluate(ckpt, tag, paths, full_reds4=False):
    res = OUT / "results"
    frames = "" if full_reds4 else "data.test.max_frames=30"
    sh(f"{PY} scripts/test.py --config configs/test_reds4.yaml --checkpoint {ckpt} --out {res}/{tag}_reds4{'_full' if full_reds4 else ''} "
       f"--set data.test.lr_root={paths['val']}/REDS_LR data.test.hr_root={paths['val']}/REDS {frames} 2>&1 | grep -E '\\[standard\\]|Error'",
       f"eval REDS4 {tag}")
    sh(f"{PY} scripts/test.py --config configs/test_lp.yaml --checkpoint {ckpt} --out {res}/{tag}_lp --set "
       f"data.test.lr_root={paths['lp']}/test/X4 data.test.hr_root={paths['lp']}/test/sharp data.test.split_file=splits/lp_lpsr.json "
       f"data.test.split=test_clean 2>&1 | grep -E '\\[standard\\]|Error'", f"eval LP {tag}")
    # Whole-clip LP inference can diverge for recurrent models trained on short clips (pilot kernel_mix):
    # also score 15-frame chunks, as REDS4 is scored.
    sh(f"{PY} scripts/test.py --config configs/test_lp.yaml --checkpoint {ckpt} --out {res}/{tag}_lp_chunk15 --set "
       f"data.test.lr_root={paths['lp']}/test/X4 data.test.hr_root={paths['lp']}/test/sharp data.test.split_file=splits/lp_lpsr.json "
       f"data.test.split=test_clean test.chunk_size=15 2>&1 | grep -E '\\[standard\\]|Error'", f"eval LP chunk15 {tag}")
    UP.mkdir(parents=True, exist_ok=True)
    shutil.copytree(res, UP / "results", dirs_exist_ok=True, ignore=shutil.ignore_patterns("images"))


def eval_metrics(tag):
    """Averages of the snapshot evaluations written by evaluate(), flattened for telemetry."""
    out = {}
    for suffix, key in (("reds4", "reds4"), ("reds4_full", "reds4_full"), ("lp", "lp_whole"), ("lp_chunk15", "lp_chunk15")):
        f = OUT / "results" / f"{tag}_{suffix}" / "metrics.json"
        if f.exists():
            avg = json.loads(f.read_text())["standard"]["average"]
            out.update({f"eval/{key}_psnr": avg["psnr_rgb"], f"eval/{key}_ssim": avg["ssim_rgb"]})
    return out


def forward_progress(proc, log_path):
    """Copy the trainer's output to its log file; echo every 1000th-iteration line and errors to the pod log."""
    keep = re.compile(r"iter \d+000/|Error|error|Traceback|Resumed|Initialised")
    with open(log_path, "w") as f:
        for line in proc.stdout:
            f.write(line)
            f.flush()
            if keep.search(line):
                print("[train] " + line.rstrip(), flush=True)


def snapshot(latest, name):
    """Weights-only copy of a (stable) latest.pth; returns (path, iteration)."""
    ck = torch.load(latest, map_location="cpu", weights_only=False)
    it = ck["iter"]
    dst = UP / "checkpoints" / f"{name}_iter{it}.pth"
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"params": ck["params"], "config": ck["config"], "iter": it}, dst)
    return dst, it


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--upload", required=True)
    p.add_argument("--budget-usd", type=float, required=True, help="hard cap for this pod, including setup")
    p.add_argument("--price-per-hour", type=float, required=True)
    p.add_argument("--spent-usd", type=float, default=0.0, help="already spent on this pod before the script started")
    p.add_argument("--eval-every", type=int, default=10000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ckpt-dataset", default="joelsathiyendra/forknet-ft-ckpts",
                   help="fallback for runpod: checkpoints (looked up by file name)")
    args = p.parse_args()
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    fetch("uom200647r/validation", DATA / "val")
    fetch("danniles/lpsr-dataset", DATA / "lpsr")
    from forknet.utils import load_config

    if any("/lp/" not in load_config(c)["data"]["train"]["hr_root"] for c in args.configs):  # LP-only pods skip 39 GB
        fetch("uom200647r/reds-vsr", DATA / "reds")
    sh(f"kaggle datasets download joelsathiyendra/forknet-weights -f forknet_reds_paper.pth -p {DATA}/weights 2>&1 | tail -1",
       "download paper checkpoint")
    paper = first(f"{DATA}/weights/forknet_reds_paper.pth*")
    if paper and paper.endswith(".zip"):
        sh(f"cd {DATA}/weights && unzip -o {paper}", "unzip paper checkpoint")
        paper = f"{DATA}/weights/forknet_reds_paper.pth"
    paths = {"lr": first(f"{DATA}/reds/**/train_sharp_bicubic/X4"), "hr": first(f"{DATA}/reds/**/train/train_sharp"),  # None on LP-only pods
             "val": str(Path(first(f"{DATA}/val/**/REDS_LR")).parent), "lp": first(f"{DATA}/lpsr/**/ALPR_Dataset")}
    log(f"paths {paths} paper={paper}")
    uploaded = False

    from forknet.utils import load_config

    for cfg_path in args.configs:
        cfg = load_config(cfg_path)
        name = cfg["name"]
        hours_left = (args.budget_usd - spent(args, t_start)) / args.price_per_hour - EVAL_RESERVE_H
        if hours_left < 0.5:
            log(f"skip {name}: budget exhausted (spent ${spent(args, t_start):.2f})")
            continue
        if "/lp/" in cfg["data"]["train"]["hr_root"]:  # LP fine-tune
            lr_root, hr_root = f"{paths['lp']}/train/X4", f"{paths['lp']}/train/sharp"
        else:
            lr_root = "null" if cfg["data"]["train"].get("degradation") is not None else paths["lr"]
            hr_root = paths["hr"]
        pretrained = paper if cfg.get("pretrained") else "null"  # from scratch when the config has no pretrained
        if str(cfg.get("pretrained", "")).startswith("runpod:"):  # a checkpoint uploaded by an earlier pod
            rel = cfg["pretrained"].split(":", 1)[1]
            sh(f"kaggle datasets download {args.upload} -f {rel} -p {DATA}/weights 2>&1 | tail -1", f"download {rel}")
            if not first(f"{DATA}/weights/{Path(rel).name}*"):  # later pods replaced it in the upload dataset
                sh(f"kaggle datasets download {args.ckpt_dataset} -f {Path(rel).name} -p {DATA}/weights 2>&1 | tail -1",
                   f"download {Path(rel).name} from {args.ckpt_dataset}")
            got = first(f"{DATA}/weights/{Path(rel).name}*")
            if got and got.endswith(".zip"):
                sh(f"cd {DATA}/weights && unzip -o {got}", f"unzip {rel}")
                got = got[:-4]
            if not got or not Path(got).exists():
                log(f"skip {name}: pretrained {rel} not found")
                continue
            pretrained = got
        extra = []
        if cfg["data"]["train"].get("plate_boxes") == "auto":  # plate-text loss: boxes on the training HR frames
            boxes = DATA / f"{name}_train_boxes.json"
            if sh(f"{PY} scripts/lp/plate_boxes.py --hr-root {hr_root} --out {boxes} 2>&1 | tail -2", "plate boxes") != 0 or not boxes.exists():
                raise RuntimeError("plate box detection failed; not training without boxes")
            extra.append(f"data.train.plate_boxes={boxes}")
        exp = OUT / "experiments" / name
        cmd = [PY, "scripts/train.py", "--config", cfg_path, "--resume", "auto", "--set", f"pretrained={pretrained}",
               f"data.train.lr_root={lr_root}", f"data.train.hr_root={hr_root}", "data.val=null",
               f"output_dir={OUT}/experiments", f"train.num_workers={args.workers}", "train.batch_size=4",
               "train.grad_checkpoint=false", f"train.save_every={args.eval_every}", "train.val_every=0",
               f"train.max_hours={hours_left:.3f}", *extra]
        log(f"train {name} with a {hours_left:.2f} h budget (spent so far ${spent(args, t_start):.2f})")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=forward_progress, args=(proc, OUT / f"{name}_train.log"), daemon=True).start()
        tele = Telemetry(cfg, exp, job_type="eval")
        latest, seen = exp / "checkpoints" / "latest.pth", 0
        while proc.poll() is None:
            if latest.exists() and time.time() - latest.stat().st_mtime > 30 and latest.stat().st_mtime > seen:
                seen = latest.stat().st_mtime
                snap, it = snapshot(latest, name)
                log(f"{name}: snapshot at iteration {it}")
                evaluate(snap, f"{name}_iter{it}", paths)
                tele.log(eval_metrics(f"{name}_iter{it}"), step=it)
                uploaded = upload(args.upload, not uploaded) == 0 or uploaded
                print(f"[pilots] SNAPSHOT {name} {it} spent ${spent(args, t_start):.2f}", flush=True)
            time.sleep(60)
        log(f"train {name} exited rc={proc.returncode}")
        final = exp / "checkpoints" / "final.pth"
        if final.exists() or latest.exists():
            snap, it = snapshot(final if final.exists() else latest, f"{name}_final")
            evaluate(snap, f"{name}_final_iter{it}", paths, full_reds4=True)
            tele.log({k.replace("eval/", "final/"): v for k, v in eval_metrics(f"{name}_final_iter{it}").items()}, step=it)
        tele.finish()
        if (exp / "train_log.jsonl").exists():
            shutil.copy(exp / "train_log.jsonl", UP / f"{name}_train_log.jsonl")
        uploaded = upload(args.upload, not uploaded) == 0 or uploaded

    rc = upload(args.upload, not uploaded)
    log(f"ALL_DONE spent ${spent(args, t_start):.2f}, final upload rc={rc}")


def remove_pod():
    """Self-removal, also after a crash: an idle pod bills by the hour."""
    pod = os.environ.get("RUNPOD_POD_ID")
    if not pod:
        return
    subprocess.run(["runpodctl", "remove", "pod", pod])
    time.sleep(120)  # only reached if removal failed: stop instead (GPU billing ends)
    subprocess.run(["runpodctl", "stop", "pod", pod])


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        log("CRASHED; uploading what exists before removing the pod")
        if UP.exists() and any(UP.iterdir()):
            upload(sys.argv[sys.argv.index("--upload") + 1], first_time=True)
        time.sleep(int(os.environ.get("CRASH_HOLD_S", "0")))  # keep the pod log readable for a while
        raise
    finally:
        remove_pod()
