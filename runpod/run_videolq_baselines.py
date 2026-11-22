"""RunPod (A40): VideoLQ no-reference quality (NIQE, BRISQUE) for BasicVSR, BasicVSR++, RVRT and MIA-VSR with the protocol
used for Fork-Net and RealBasicVSR (first 20 frames per video, every 5th frame scored). Each model's result is uploaded to
the Kaggle dataset as soon as it is done, so a time limit or crash never loses finished models; the pod removes itself.

    python runpod/run_videolq_baselines.py --upload joelsathiyendra/forknet-videolq-baselines
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
    created = False

    sh(f"{PY} -m pip install -q timm einops gdown pyiqa 2>&1 | tail -1", "pip")
    sh("git clone -q --depth 1 https://github.com/JingyunLiang/RVRT /workspace/RVRT && wget -q -O /workspace/rvrt_reds.pth "
       "https://github.com/JingyunLiang/RVRT/releases/download/v0.0/001_RVRT_videosr_bi_REDS_30frames.pth", "get RVRT")
    sh("git clone -q --depth 1 https://github.com/CVL-UESTC/MIA-VSR /workspace/MIA-VSR && "
       "gdown -q --folder https://drive.google.com/drive/folders/1DvsUP-FVwENIpLyeQPQMHc7QnxX1F5Pd -O /workspace/mia_w", "get MIA-VSR")
    mia_w = first("/workspace/mia_w/**/*REDS*.pth")
    fetch("joelsathiyendra/video-lq-dataset-new", DATA / "videolq")
    videolq = str(Path(first(f"{DATA}/videolq/**/000/*.png")).parent.parent)
    log(f"videolq {videolq} ({len(list(Path(videolq).iterdir()))} videos)")

    run = {"BasicVSR": f"{PY} scripts/baselines/basicvsr.py --chunk-size 10",
           "BasicVSR++": f"{PY} scripts/baselines/basicvsrpp.py --chunk-size 10",
           "RVRT": f"{PY} scripts/baselines/sota_repo.py --model rvrt --repo /workspace/RVRT --weights /workspace/rvrt_reds.pth "
                   "--amp --chunk-size 10 --tile 256"}
    if mia_w:
        run["MIA-VSR"] = (f"{PY} scripts/baselines/sota_repo.py --model miavsr --repo /workspace/MIA-VSR --weights {mia_w} "
                          "--amp --chunk-size 10 --tile 256")
    summary = {}
    for name, cmd in run.items():
        out = TMP / name
        if sh(f"{cmd} --lr-root {videolq} --max-frames 20 --out {out}", f"VideoLQ {name}") == 0:
            j = res / f"videolq_{name}.json"
            if sh(f"{PY} scripts/eval_noref.py --sr {out} --every 5 --name {name} --out {j}", f"no-ref {name}") == 0:
                summary[name] = json.loads(j.read_text()).get("average")
                log(f"{name}: {summary[name]}")
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
