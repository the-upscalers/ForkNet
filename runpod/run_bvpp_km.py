"""RunPod: BasicVSR++ fine-tuned with the kernel-mix recipe (scripts/baselines/train_basicvsrpp.py), then scored like
every other method: REDS4 (whole clips, benchmark and OpenCV-cubic LR, the official BasicVSR++ alongside on the latter),
LP test (PSNR/SSIM/LPIPS/tOF on test_clean; plate recognition with the verified labels). Checkpoints and results go to a
private Kaggle dataset after every stage; the pod removes itself at the end (after a short hold if it crashed).

    python runpod/run_bvpp_km.py --upload joelsathiyendra/forknet-bvpp-km --budget-usd 5.5 --price-per-hour 0.59
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
from run_finetune import UP, remove_pod, upload  # noqa: E402
from run_pilots import DATA, PY, fetch, first, log, sh  # noqa: E402

TMP = Path("/workspace/tmp")
EVAL_RESERVE_H = 0.8


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--upload", required=True)
    p.add_argument("--budget-usd", type=float, required=True)
    p.add_argument("--price-per-hour", type=float, required=True)
    p.add_argument("--iters", type=int, default=30000)
    args = p.parse_args()
    t0 = time.time()
    res = UP / "results"
    res.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    created = False

    def push():
        nonlocal created
        created = upload(args.upload, not created) == 0 or created

    fetch("uom200647r/reds-vsr", DATA / "reds")
    fetch("uom200647r/validation", DATA / "val")
    fetch("danniles/lpsr-dataset", DATA / "lpsr")
    hr_train = first(f"{DATA}/reds/**/train/train_sharp")
    val = str(Path(first(f"{DATA}/val/**/REDS_LR")).parent)
    lpsr = first(f"{DATA}/lpsr/**/ALPR_Dataset")
    log(f"paths train={hr_train} val={val} lp={lpsr}")

    hours = args.budget_usd / args.price_per_hour - (time.time() - t0) / 3600 - EVAL_RESERVE_H
    exp = Path("/workspace/out/bvpp_km")
    sh(f"{PY} -u scripts/baselines/train_basicvsrpp.py --hr-root {hr_train} --out {exp} --iters {args.iters} "
       f"--time-budget-h {hours:.2f} 2>&1 | grep --line-buffered -v Warning", "train BasicVSR++ kernel mix")
    ckpt = first(f"{exp}/checkpoints/final_iter*.pth")
    if not ckpt:
        raise RuntimeError("no final checkpoint")
    (UP / "checkpoints").mkdir(exist_ok=True)
    shutil.copy(ckpt, UP / "checkpoints" / f"basicvsrpp_kernelmix_{Path(ckpt).name}")
    push()

    km = f"{PY} scripts/baselines/basicvsrpp.py --weights {ckpt}"
    official = f"{PY} scripts/baselines/basicvsrpp.py"
    # REDS4 with OpenCV-cubic LR (whole 100-frame clips), made from the HR frames
    cv2_lr = TMP / "reds4_cv2"
    for clip in ("000", "011", "015", "020"):
        (cv2_lr / clip).mkdir(parents=True, exist_ok=True)
        for f in sorted((Path(val) / "REDS" / clip).glob("*.png")):
            gt = cv2.imread(str(f))
            cv2.imwrite(str(cv2_lr / clip / f.name), cv2.resize(gt, (gt.shape[1] // 4, gt.shape[0] // 4), interpolation=cv2.INTER_CUBIC))
    for name, cmd, lr in (("BasicVSR++-kernelmix", km, f"{val}/REDS_LR"), ("BasicVSR++-kernelmix_cv2", km, cv2_lr),
                          ("BasicVSR++_cv2", official, cv2_lr)):
        out = TMP / name
        if sh(f"{cmd} --lr-root {lr} --out {out}", f"REDS4 {name}") == 0:
            sh(f"{PY} scripts/evaluate_outputs.py --sr {out} --gt {val}/REDS --name {name} --out {res}/reds4_{name}.json", f"eval {name}")
        shutil.rmtree(out, ignore_errors=True)
    push()

    # LP test: images, quality on test_clean, recognition with the verified labels
    lp_out = TMP / "lp_bvpp_km"
    if sh(f"{km} --lr-root {lpsr}/test/X4 --out {lp_out}", "LP BasicVSR++-kernelmix") == 0:
        sh(f"{PY} scripts/evaluate_outputs.py --sr {lp_out} --gt {lpsr}/test/sharp --split-file splits/lp_lpsr.json "
           f"--split test_clean --extra lpips tof --name BasicVSR++-kernelmix --out {res}/lp_test_clean_full_BasicVSR++-kernelmix.json", "LP eval")
        sh(f"{PY} scripts/test.py --config configs/test_lp.yaml --bicubic matlab --save-images --out {TMP}/lp_Bicubic --set "
           f"data.test.lr_root={lpsr}/test/X4 data.test.hr_root={lpsr}/test/sharp data.test.split_file=null data.test.split=null", "LP bicubic")
        bic = first(f"{TMP}/lp_Bicubic/**/images")
        for engine in ("fast_plate_ocr", "easyocr"):
            sh(f"{PY} scripts/lp/ocr_eval.py --labels data_lp/lp_test_labels.csv --status verified --boxes data_lp/lp_test_boxes.json "
               f"--reference Bicubic --engine {engine} --method 'HR={lpsr}/test/sharp' 'Bicubic={bic}' 'BasicVSR++-kernelmix={lp_out}' "
               f"--out {res}/lp_ocr_bvpp_km_{engine}.json", f"OCR {engine}")
    push()
    for f in sorted(res.glob("*.json")):
        avg = json.loads(f.read_text()).get("standard", {}).get("average", {})
        log(f"{f.name}: {json.dumps({k: round(v, 4) for k, v in avg.items() if k in ('psnr_rgb', 'ssim_rgb', 'lpips', 'tof')})}")
    log("ALL_DONE")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        log("CRASHED; uploading what exists, holding the pod briefly so the log can be read")
        if UP.exists() and any(UP.iterdir()):
            upload(sys.argv[sys.argv.index("--upload") + 1], first_time=True)
        time.sleep(int(os.environ.get("CRASH_HOLD_S", "300")))
        raise
    finally:
        remove_pod()
