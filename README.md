# Fork-Net

Code, configurations and results for

> *Investigating the Resampling-Kernel Gap in Video Super-Resolution for License Plate Recognition.*
> T. J. Sathiyendra, D. A. Niles, T. Nagaratnam, T. D. Ambegoda. Submitted to IEEE T-ITS.

including **Fork-Net**, a lightweight U-Net for ×4 video super-resolution with second-order
bidirectional propagation, and the kernel-mix fine-tuning recipe applied to Fork-Net and BasicVSR++.

**Reviewers:** [docs/PAPER.md](docs/PAPER.md) maps every table and figure to its configuration, script and
result file. `python scripts/verify_paper_results.py` checks the printed numbers of Tables II–IV and the
significance tests against `results/paper/` in a few seconds on CPU, with no data or GPU.

Fork-Net super-resolves the bicubic-upsampled frame with a U-Net, injects LR features into the
encoder, and propagates features forward and backward in time (from t±1 and t±2, SpyNet-aligned)
inside the decoder with ConvNeXt-style aggregation. Blueprint-separable convolutions keep it at
**4.37M parameters** (including SpyNet). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

> [!IMPORTANT]
> Consolidating the research code uncovered bugs in the propagation of the original implementation
> (flows applied in the wrong direction, wrong time steps in the backward branch) and a non-standard
> evaluation protocol. The default configuration fixes them; `legacy_*` switches reproduce the original
> behaviour bit-for-bit so existing checkpoints still work. See [docs/FINDINGS.md](docs/FINDINGS.md).

## Installation

```bash
git clone https://github.com/the-upscalers/ForkNet && cd ForkNet
python -m pip install -e ".[dev,video]"      # or: pip install -r requirements.txt
make test                                    # ~3 min on CPU
```

Tested with Python 3.13 + PyTorch 2.14 (CPU, Windows) and on Kaggle GPUs (versions in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)). SpyNet weights ship in `weights/`.

## Data

**REDS** (training and REDS4 testing). Download `train_sharp`, `val_sharp`, `train_sharp_bicubic`
and `val_sharp_bicubic` from the [REDS page](https://seungjunnah.github.io/Datasets/reds.html), then:

```bash
python scripts/prepare_reds.py --reds-root /path/to/REDS --dst data/REDS   # 270 clips, BasicVSR protocol
python scripts/prepare_reds.py --verify data/REDS
```

**License-plate video dataset** (256×256 plate crops; 221 train / 30 val / 31 test clips, leak-free subsets
in `splits/lp_lpsr.json`). The frames and plate strings are available for non-commercial research under the
[data-use agreement](docs/DATA_USE_AGREEMENT.md); datasheet in [docs/DATASET.md](docs/DATASET.md). The archive
includes the LR frames used in the paper (OpenCV cubic ×4); to regenerate them from the HR frames, pass
`--kernel cv2_bicubic` to `scripts/prepare_lp_dataset.py`.

Data paths are config entries and can be overridden from the command line, e.g.
`--set data.train.lr_root=/mnt/REDS/LR`.

## Training

```bash
python scripts/train.py --config configs/train_reds.yaml                  # 1 GPU
torchrun --nproc_per_node 2 scripts/train.py --config configs/train_reds.yaml  # 2 GPUs, total batch 4
python scripts/train.py --config configs/train_reds.yaml --resume experiments/forknet_reds/checkpoints/latest.pth
python scripts/train.py --config configs/finetune_lp.yaml --set pretrained=experiments/forknet_reds/checkpoints/final.pth
```

| Config | Purpose |
|---|---|
| `configs/train_reds.yaml` | main model, paper hyper-parameters, corrected propagation |
| `configs/train_reds_legacy.yaml` | exact behaviour of the original training code |
| `configs/ablation/*.yaml` | Table 1: first order, residual blocks instead of ConvNeXt, no BSConv, original propagation |
| `configs/finetune_lp.yaml` | fine-tuning on the LP dataset |

Each run writes `experiments/<name>/{config.yaml, train.log, train_log.jsonl, val_log.jsonl, checkpoints/}`.

## Evaluation

```bash
# Fork-Net on REDS4: standard protocol (all frames, uint8, per-frame PSNR/SSIM) + original protocol
python scripts/test.py --config configs/test_reds4.yaml --checkpoint weights/forknet_reds.pth --protocol both --save-images
python scripts/test.py --config configs/test_reds4.yaml --bicubic          # MATLAB-bicubic baseline
python scripts/test.py --config configs/test_lp.yaml --checkpoint weights/forknet_lp.pth

# Any other method: score its saved frames with the same code
python scripts/baselines/basicvsr.py --lr-root data/REDS/LR_bicubic_X4 --clips 000 011 015 020 --out outputs/basicvsr
python scripts/evaluate_outputs.py --sr outputs/basicvsr --gt data/REDS/HR --clips 000 011 015 020 --name BasicVSR --out results/reds4/basicvsr.json

python scripts/make_tables.py results/*/*/metrics.json results/reds4/*.json   # Markdown + LaTeX
python scripts/complexity.py --runtime                                        # params, MACs, FLOPs, ms/frame
python scripts/infer.py --checkpoint weights/forknet_reds.pth --input video.mp4 --out outputs/demo --video
```

`scripts/inspect_checkpoint.py <ckpt>` identifies the architecture of any checkpoint written by the
original research code and can export it in the release format (`--export`).

## Research toolkit

| Tool | Purpose |
|---|---|
| `configs/ablation/*.yaml` | no propagation, forward-only, first order, no flow, original propagation, residual blocks, no BSConv, no LR injection, widths 32/48 |
| `configs/tradeoff/*.yaml` | batch-length trade-off at a fixed frame budget |
| `configs/train_reds_real.yaml` | real-world training with RealBasicVSR-style random degradations (`forknet/data/degradations.py`) |
| `scripts/degradation_sweep.py` + `configs/robustness.yaml` | robustness to kernel shift, blur, noise, JPEG, H.264 |
| `scripts/evaluate_outputs.py --extra lpips tof` | perceptual and temporal-consistency metrics |
| `scripts/eval_noref.py` | NIQE / BRISQUE for real videos without ground truth (VideoLQ) |
| `scripts/lp/{auto_label,ocr_eval,extract_tracks}.py` | plate labels by two-engine OCR consensus, recognition accuracy, mining plate clips from raw video |
| `splits/lp_lpsr.json` | LP splits incl. the leakage-free `test_clean` |
| `kaggle/`, `runpod/` | GPU jobs on Kaggle (resumable 11 h sessions) and the full training matrix on RunPod |

## Reproducibility

[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) describes the evaluation protocol, validation
of the evaluation code against published numbers, and Kaggle runs. `kaggle/` contains the validation kernel used to test this repository on a GPU.

## Repository layout

```
forknet/archs/      Fork-Net, SpyNet, building blocks, flow utilities
forknet/data/       paired video datasets, MATLAB-compatible imresize
forknet/            metrics, losses, chunked inference, evaluation, trainer
forknet/utils/      configs, checkpoints (incl. original-format loading), complexity
configs/            training / ablation / fine-tuning / test configs
scripts/            train, test, evaluate_outputs, infer, complexity, inspect_checkpoint, data preparation, baselines
tests/              unit, integration, end-to-end and legacy-equivalence tests
docs/               architecture, findings, reproducibility, dataset datasheet
kaggle/             GPU validation kernel
```

## Citation

See [CITATION.cff](CITATION.cff).

## Acknowledgements

SpyNet, flow warping and the BasicVSR baseline follow [BasicSR](https://github.com/XPixelGroup/BasicSR)
and [MMEditing](https://github.com/open-mmlab/mmagic) (Apache-2.0). The U-Net block is adapted from
[SR-UNet](https://github.com/fede-vaccaro/fast-sr-unet); BSConv from [Haase & Amthor (2020)](https://github.com/zeiss-microscopy/BSConv).
