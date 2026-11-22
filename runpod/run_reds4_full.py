"""RunPod (48 GB GPU): REDS4 port check of BasicVSR++, RVRT and MIA-VSR with whole 100-frame clips, as published, instead of
the short windows and tiles a 16 GB GPU needs. Falls back to 128-pixel spatial tiles only if a model runs out of memory.
Results go to a private Kaggle dataset; the pod removes itself at the end (after a short hold if it crashed).

    python runpod/run_reds4_full.py --upload joelsathiyendra/forknet-opt-reds4full
"""

import argparse
import json
import os
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

    sh(f"{PY} -m pip install -q timm einops gdown 2>&1 | tail -1", "pip")
    sh("git clone -q --depth 1 https://github.com/JingyunLiang/RVRT /workspace/RVRT && wget -q -O /workspace/rvrt_reds.pth "
       "https://github.com/JingyunLiang/RVRT/releases/download/v0.0/001_RVRT_videosr_bi_REDS_30frames.pth", "get RVRT")
    sh("git clone -q --depth 1 https://github.com/CVL-UESTC/MIA-VSR /workspace/MIA-VSR && "
       "gdown -q --folder https://drive.google.com/drive/folders/1DvsUP-FVwENIpLyeQPQMHc7QnxX1F5Pd -O /workspace/mia_w", "get MIA-VSR")
    mia_w = first("/workspace/mia_w/**/*REDS*.pth")
    fetch("uom200647r/validation", DATA / "val")
    val = str(Path(first(f"{DATA}/val/**/REDS_LR")).parent)

    run = {"BasicVSR++": f"{PY} scripts/baselines/basicvsrpp.py",
           "RVRT": f"{PY} scripts/baselines/sota_repo.py --model rvrt --repo /workspace/RVRT --weights /workspace/rvrt_reds.pth"}
    if mia_w:
        run["MIA-VSR"] = f"{PY} scripts/baselines/sota_repo.py --model miavsr --repo /workspace/MIA-VSR --weights {mia_w}"
    summary, created = {}, False
    for name, cmd in run.items():
        out = TMP / name
        for setting in ("--chunk-size 100", "--chunk-size 100 --tile 128"):
            shutil.rmtree(out, ignore_errors=True)
            if sh(f"{cmd} --lr-root {val}/REDS_LR --out {out} {setting}", f"{name} {setting}") == 0:
                j = res / f"reds4_full_{name}.json"
                if sh(f"{PY} scripts/evaluate_outputs.py --sr {out} --gt {val}/REDS --name {name} --out {j}", f"eval {name}") == 0:
                    avg = json.loads(j.read_text())["standard"]["average"]
                    summary[name] = {"setting": setting, "psnr_rgb": avg["psnr_rgb"], "ssim_rgb": avg["ssim_rgb"]}
                    log(f"{name} ({setting}): {avg['psnr_rgb']:.2f} dB / {avg['ssim_rgb']:.4f}")
                break
        shutil.rmtree(out, ignore_errors=True)
        (res / "summary.json").write_text(json.dumps(summary, indent=1))
        created = upload(args.upload, not created) == 0 or created
    log(f"ALL_DONE {json.dumps(summary)}")


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
