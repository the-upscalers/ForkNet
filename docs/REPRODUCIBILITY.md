# Reproducibility

## 1. Evaluation protocol

All numbers produced by this repository use the protocol of BasicVSR / IconVSR / BasicVSR++ /
MIA-VSR (BasicSR):

* **REDS4** = clips 000, 011, 015, 020 of REDS (all 100 frames), LR = official bicubic ×4.
  These four clips are excluded from training.
* SR frames are clamped to [0, 1] and rounded to uint8 before scoring.
* PSNR and SSIM per frame on RGB, `crop_border = 0` (Y-channel values are also reported);
  mean over the frames of a clip, then mean over clips.
* SSIM: 11×11 Gaussian window, σ = 1.5, K₁ = 0.01, K₂ = 0.03, valid filtering (BasicSR).
* Fork-Net runs on disjoint 15-frame chunks (`test.chunk_size: 15`) to bound GPU memory.
  `test.overlap` enables overlapping windows.
* Baselines are scored on their saved frames with `scripts/evaluate_outputs.py`, i.e. the same code.

`scripts/test.py --protocol legacy` re-creates the original validation loop (see
[FINDINGS.md](FINDINGS.md), F6). Do not mix the two protocols in one table.

### Validation of the evaluation code

| Check | Expected (published) | This repository |
|---|---|---|
| Bicubic (MATLAB kernel), REDS4 PSNR / SSIM | 26.14 / 0.7292 | **26.1355 / 0.7292** (Kaggle, 2026-10-20) |
| BasicVSR official weights, REDS4 PSNR / SSIM | 31.42 / 0.8909 | **31.4170 / 0.8909** (Kaggle, 2026-10-20) |
| PSNR vs `skimage.metrics.peak_signal_noise_ratio` | identical | `tests/test_metrics.py` (abs. diff < 1e-10) |
| SSIM vs `skimage.metrics.structural_similarity` (Gaussian) | identical | `tests/test_metrics.py` (abs. diff < 1e-6) |
| Fork-Net outputs vs original implementation (same weights) | identical | `tests/test_legacy_equivalence.py` (max. diff ≤ 1.2e-7) |

## 2. Evaluating original-format checkpoints

```bash
python scripts/inspect_checkpoint.py best_checkpoint.pth --export weights/forknet_reds_legacy.pth
python scripts/test.py --config configs/test_reds4.yaml --checkpoint weights/forknet_reds_legacy.pth --protocol both
python scripts/test.py --config configs/test_lp.yaml    --checkpoint weights/forknet_lp_legacy.pth   --protocol both
```

Original-format checkpoints are loaded with the three `legacy_*` switches on, so the network computes
exactly what it computed during training. `--protocol standard` gives numbers that are comparable
with other methods.

## 3. Training

| Setting | `configs/train_reds.yaml` |
|---|---|
| Data | REDS 266 clips (270 minus REDS4), 15-frame clips, 64×64 LR patches, random temporal window, h/v flips |
| Optimiser | Adam, β = (0.9, 0.99), lr 2×10⁻⁴ (SpyNet 2.5×10⁻⁵), no weight decay |
| Schedule | cosine annealing to 1×10⁻⁷ over 300k iterations |
| SpyNet | pretrained (BasicSR), frozen for the first 5k iterations |
| Batch | 4 (total over GPUs), mixed precision (fp16) |
| Loss | Charbonnier, ε = 1×10⁻⁸ |
| Seed | 0 (`seed`; `deterministic: true` selects deterministic kernels where PyTorch has them. `grid_sample`'s backward pass has none, so GPU training is not bitwise reproducible. Report several seeds instead) |

`configs/train_reds_legacy.yaml` reproduces the original code instead. Every run stores its
resolved config, environment (Python / PyTorch / CUDA / GPU), JSONL training and validation logs,
and checkpoints with optimiser, scheduler and AMP-scaler state for exact resumption.

## 4. Kaggle validation

`kaggle/forknet_validation.py` runs the test suite (the equivalence tests against the original code need
access to the private `pixel-level-vsr` repository and are skipped on Kaggle; they were run locally),
complexity and runtime, the bicubic and BasicVSR REDS4 checks, and two
short REDS training runs (corrected vs. original propagation) followed by REDS4 evaluation.
`bash kaggle/push_code.sh <kaggle-user>` uploads the code as a private dataset and starts it.

Results of the first run (2026-10-20, Kaggle Tesla T4 15 GB, `uom200590j/forknet-validation`):

* tests: 72 passed (legacy-equivalence tests error because the original repository is private; fixed to skip);
* REDS4 checks above reproduce the published bicubic and BasicVSR numbers to the reported precision, which validates
  data loading, quantisation and the PSNR / SSIM implementation;
* PyTorch-kernel bicubic gives 26.24 / 0.7319, i.e. the MATLAB kernel must be used for the "Bicubic" row;
* training with batch 4 and 15-frame 64x64 clips **does not fit in 15 GB** (out of memory). Activation checkpointing
  (`train.grad_checkpoint: true`, numerically identical gradients, `tests/test_new_features.py`) or batch 2 is
  required on 16 GB GPUs; the Kaggle training kernel probes this automatically.
