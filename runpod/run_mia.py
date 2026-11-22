"""RunPod: MIA-VSR (Zhou et al., CVPR 2024; official code and REDS weights) zero-shot, scored like every other method:
REDS4 port check, LP test (PSNR/SSIM/LPIPS/tOF on test_clean, plate recognition with the verified labels) and Vid4 BI.
Results go to a private Kaggle dataset after every stage; the pod removes itself at the end (also after a crash).

    python runpod/run_mia.py --upload joelsathiyendra/forknet-opt-mia
"""

import argparse
import glob
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_finetune import UP, remove_pod, upload  # noqa: E402
from run_pilots import DATA, PY, fetch, first, log, sh  # noqa: E402

TMP = Path("/workspace/tmp")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--upload", required=True)
    args = p.parse_args()
    res = UP / "results"
    res.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    uploaded = False

    def push():
        nonlocal uploaded
        uploaded = upload(args.upload, not uploaded) == 0 or uploaded

    sh(f"{PY} -m pip install -q timm einops gdown 2>&1 | tail -1", "pip")
    sh("git clone -q --depth 1 https://github.com/CVL-UESTC/MIA-VSR /workspace/MIA-VSR && "
       "gdown -q --folder https://drive.google.com/drive/folders/1DvsUP-FVwENIpLyeQPQMHc7QnxX1F5Pd -O /workspace/mia_w; "
       "gdown -q --folder https://drive.google.com/drive/folders/1SyCIPlCZp4SkORMaUVKzT0M07nDAhdrl -O /workspace/mia_w2; "
       "find /workspace/mia_w* -name '*.pth'", "get MIA-VSR")
    weights = first("/workspace/mia_w*/**/*REDS*.pth") or first("/workspace/mia_w*/**/*.pth")
    log(f"weights {weights}")
    if not weights:
        raise RuntimeError("MIA-VSR weights not downloaded")
    fetch("uom200647r/validation", DATA / "val")
    fetch("danniles/lpsr-dataset", DATA / "lpsr")
    fetch("uom200647r/vid4-dataset", DATA / "vid4")
    val = str(Path(first(f"{DATA}/val/**/REDS_LR")).parent)
    lpsr = first(f"{DATA}/lpsr/**/ALPR_Dataset")
    vid4 = str(Path(first(f"{DATA}/vid4/**/BIx4/calendar")).parent.parent)
    mia = f"{PY} scripts/baselines/sota_repo.py --model miavsr --repo /workspace/MIA-VSR --weights {weights} --amp --chunk-size 16"
    split = "--split-file splits/lp_lpsr.json --split test_clean"

    # REDS4 port check (published 32.78 dB RGB with longer windows)
    if sh(f"{mia} --lr-root {val}/REDS_LR --out {TMP}/reds4", "REDS4 MIA-VSR") == 0:
        sh(f"{PY} scripts/evaluate_outputs.py --sr {TMP}/reds4 --gt {val}/REDS --name MIA-VSR --out {res}/reds4_MIA-VSR.json", "REDS4 eval")
    shutil.rmtree(TMP / "reds4", ignore_errors=True)
    push()

    # LP test: images, quality on test_clean, recognition (verified labels, plate boxes found on HR) vs bicubic
    if sh(f"{mia} --lr-root {lpsr}/test/X4 --out {TMP}/lp_MIA-VSR", "LP MIA-VSR") == 0:
        sh(f"{PY} scripts/evaluate_outputs.py --sr {TMP}/lp_MIA-VSR --gt {lpsr}/test/sharp {split} --extra lpips tof --name MIA-VSR "
           f"--out {res}/lp_test_clean_full_MIA-VSR.json", "LP eval")
        sh(f"{PY} scripts/test.py --config configs/test_lp.yaml --bicubic matlab --save-images --out {TMP}/lp_Bicubic --set "
           f"data.test.lr_root={lpsr}/test/X4 data.test.hr_root={lpsr}/test/sharp data.test.split_file=null data.test.split=null", "LP bicubic")
        bic = first(f"{TMP}/lp_Bicubic/**/images")
        for engine in ("fast_plate_ocr", "easyocr"):
            sh(f"{PY} scripts/lp/ocr_eval.py --labels data_lp/lp_test_labels.csv --status verified --boxes data_lp/lp_test_boxes.json "
               f"--reference Bicubic --engine {engine} --method 'HR={lpsr}/test/sharp' 'Bicubic={bic}' 'MIA-VSR={TMP}/lp_MIA-VSR' "
               f"--out {res}/lp_ocr_MIA-VSR_{engine}.json", f"OCR {engine}")
    push()

    # Vid4 BI (cross-dataset for REDS-trained models)
    if sh(f"{mia} --lr-root {vid4}/BIx4 --out {TMP}/vid4", "Vid4 MIA-VSR") == 0:
        sh(f"{PY} scripts/evaluate_outputs.py --sr {TMP}/vid4 --gt {vid4}/GT --name MIA-VSR --out {res}/vid4_MIA-VSR.json", "Vid4 eval")
    push()
    for f in sorted(glob.glob(f"{res}/*.json")):
        log(f"{Path(f).name}: {json.dumps(json.loads(Path(f).read_text()).get('standard', {}).get('average', ''))[:200]}")
    log("ALL_DONE")


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except BaseException:
        log("CRASHED; uploading what exists before removing the pod")
        if UP.exists() and any(UP.iterdir()):
            upload(sys.argv[sys.argv.index("--upload") + 1], first_time=True)
        raise
    finally:
        remove_pod()
