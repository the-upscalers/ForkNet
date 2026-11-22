#!/usr/bin/env bash
# Evaluate every finished experiment on all axes (REDS4, Vid4, LP test_clean, robustness sweep, VideoLQ).
#   bash runpod/eval_all.sh experiments results
set -uo pipefail
EXP=${1:-experiments}; OUT=${2:-results}
SPLIT="data.test.split_file=splits/lp_lpsr.json data.test.split=test_clean"
for ck in "$EXP"/*/checkpoints/final.pth weights/forknet_*paper.pth; do
  [ -f "$ck" ] || continue
  name=$(basename "$(dirname "$(dirname "$ck")")"); [[ "$ck" == weights/* ]] && name=$(basename "$ck" .pth)
  echo "== $name"
  python scripts/test.py --config configs/test_reds4.yaml --checkpoint "$ck" --out "$OUT/reds4/$name"
  python scripts/test.py --config configs/test_vid4.yaml --checkpoint "$ck" --out "$OUT/vid4/$name"
  python scripts/test.py --config configs/test_lp.yaml --checkpoint "$ck" --save-images --out "$OUT/lp/$name" --set $SPLIT
  python scripts/degradation_sweep.py --hr-root data/lp/lpsr/test/sharp --split-file splits/lp_lpsr.json --split test_clean \
      --cache cache/sweep_lp --checkpoint "$ck" --name "$name" --out "$OUT/robustness_lp.json"
  python scripts/infer.py --checkpoint "$ck" --input data/VideoLQ --out "cache/videolq/$name" --chunk-size 5 --amp --max-frames 20 \
      && python scripts/eval_noref.py --sr "cache/videolq/$name" --every 5 --name "$name" --out "$OUT/videolq/$name.json"
done
python scripts/make_tables.py "$OUT"/reds4/*/metrics.json --out "$OUT/table_reds4.md"
python scripts/make_tables.py "$OUT"/lp/*/metrics.json --out "$OUT/table_lp.md"
