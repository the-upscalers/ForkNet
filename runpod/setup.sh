#!/usr/bin/env bash
# One-time setup of a RunPod (or any Linux GPU) machine.
#   export KAGGLE_API_TOKEN=...            # account with access to the datasets below
#   bash runpod/setup.sh /workspace/data   # run from the repository root
# Afterwards the configs' default paths (data/REDS, data/lp, data/Vid4, data/VideoLQ) resolve.
set -euo pipefail
DATA=${1:-/workspace/data}
OWNER_LP=${OWNER_LP:-uom200590j}   # owner of the private copies (forknet-lp-data, videolq-lr, forknet-weights)
mkdir -p "$DATA" data weights

python -m pip install -q -r requirements.txt kaggle imageio-ffmpeg lpips pyiqa easyocr "fast-plate-ocr[onnx]"
python -m pip install -q -e .

dl() {  # dl <kaggle-ref> <subdir>
  if [ ! -d "$DATA/$2" ]; then
    echo "== downloading $1"; kaggle datasets download "$1" -p "$DATA/$2" --unzip -q
  fi
}
dl uom200647r/reds-vsr reds                  # ~42 GB: REDS train_sharp + X4
dl uom200647r/validation validation          # REDS4 + REDS val_sharp (+ X4)
dl uom200647r/vid4-dataset vid4
dl "$OWNER_LP/forknet-lp-data" lp            # lpsr/, alpr_subset/, real_lr_videos/
dl "$OWNER_LP/videolq-lr" videolq
dl "$OWNER_LP/forknet-weights" weights

# REDS: 270 clips (train_sharp 000-239 + val_sharp renamed 240-269), BasicVSR protocol.
STAGE="$DATA/reds_stage"; mkdir -p "$STAGE/train_sharp_bicubic" "$STAGE/val_sharp_bicubic"
f() { find "$1" -type d -path "*$2" | head -1; }
ln -sfn "$(f "$DATA/reds" /train/train_sharp)" "$STAGE/train_sharp"
ln -sfn "$(f "$DATA/reds" /train_sharp_bicubic/X4)" "$STAGE/train_sharp_bicubic/X4"
ln -sfn "$(f "$DATA/validation" /val/val_sharp)" "$STAGE/val_sharp"
ln -sfn "$(f "$DATA/validation" /val_sharp_bicubic/X4)" "$STAGE/val_sharp_bicubic/X4"
python scripts/prepare_reds.py --reds-root "$STAGE" --dst data/REDS

ln -sfn "$DATA/lp" data/lp
ln -sfn "$(find "$DATA/vid4" -maxdepth 3 -type d -name GT -printf '%h\n' | head -1)" data/Vid4
ln -sfn "$(find "$DATA/videolq" -maxdepth 2 -type d -name VideoLQ | head -1)" data/VideoLQ
cp -n "$DATA"/weights/*.pth weights/ 2>/dev/null || true

python -m pytest -q -x
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "Setup complete."
