# Reproducing the paper

*Investigating the Resampling-Kernel Gap in Video Super-Resolution for License Plate Recognition*
(T. J. Sathiyendra, D. A. Niles, T. Nagaratnam, T. D. Ambegoda; submitted to IEEE T-ITS).

There are three levels of reproduction, from cheapest to most expensive:

| Level | What it checks | Needs | Time |
|---|---|---|---|
| 1. Check the printed numbers | Tables II–IV and every McNemar test quoted in the text, against the released per-clip results | Python 3, nothing else | seconds, CPU |
| 2. Re-evaluate the released models | REDS4, Vid4, the degradation sweep, RLPR and the LP tables from the released weights | the datasets below, one 16 GB GPU | hours |
| 3. Re-train | every fine-tune in the paper | REDS, one 48 GB GPU (or 16 GB with `train.grad_checkpoint: true`) | 4.5 GPU-h per Fork-Net fine-tune |

## 1. Check the printed numbers

```bash
python scripts/verify_paper_results.py      # prints one line per number; exits 1 on any mismatch
```

`results/paper/` holds the result files the tables were built from: per-clip PSNR/SSIM/LPIPS/tOF (`lp_test/`),
per-clip recognition correctness for both recognizers on the 51 LP-rec clips (`lp_rec/`) and the 200 RLPR
sequences (`rlpr/`), the degradation sweep and kernel-generalization runs (`sweep/`), and REDS4 (`reds4/`). Plate
strings and file paths were removed; recognition is stored as correct / incorrect per clip, which is all the
McNemar tests need.

## 2. Tables and figures

All fine-tunes start from a released checkpoint. Configs written for our cloud runs point at `runpod:` paths;
override them with `--set pretrained=weights/<file>`.

| Paper item | Configuration / script | Result files |
|---|---|---|
| Table I, Fig. 3 (degradation sweep, LP-test, 40 frames) | `scripts/degradation_sweep.py` with `configs/robustness.yaml`; baselines via `scripts/baselines/` | `results/paper/sweep/robustness_lp_*.json` |
| Section VII kernel generalization (unseen OpenCV kernels) | `scripts/kernel_generalization_lr.py` | `results/paper/sweep/kernel_generalization.json` |
| Table II reconstruction (LP-test, 23 clips) | `scripts/test.py --config configs/test_lp.yaml`; baselines: `scripts/evaluate_outputs.py --extra lpips tof` on saved frames | `results/paper/lp_test/*.json` |
| Table II recognition (LP-rec, 51 clips) | `scripts/lp/plate_boxes.py`, `scripts/lp/ocr_eval.py --engine {fast_plate_ocr,easyocr} --reference Bicubic`, `scripts/lp/paired_stats.py` | `results/paper/lp_rec/` |
| Table III (RLPR) | `scripts/lp/rlpr_prepare.py`, then as for LP-rec | `results/paper/rlpr/` |
| Table IV (variants) | `configs/finetune_mixed_from_paper.yaml`, `configs/finetune_perceptual_from_kernelmix.yaml` (`forknet/aux_losses.py`), `configs/lp/lp_from_kernelmix.yaml`, `configs/lp/lp_text_from_kernelmix.yaml` | `results/paper/lp_test/Ours-*.json`, `lp_rec/` |
| Table V (REDS4, Vid4) | `scripts/test.py --config configs/test_reds4.yaml` / `test_vid4.yaml`; whole-clip baselines: `runpod/run_reds4_full.py` | `results/paper/reds4/` |
| Table VI (ablations) | `configs/finetune_kernelmix_nocv2_from_paper.yaml`, `configs/finetune_kernelmix7_from_paper.yaml`, `configs/seeds/*.yaml`, `configs/pilot/kernel_mix.yaml`, `configs/size/km_w{32,64}_full.yaml` | `results/paper/lp_test/Ours-kernelmix-nocv2.json`; others in the training logs of each run |
| Kernel-mix fine-tune (Section IV) | `configs/finetune_kernelmix_from_paper.yaml`; control: `configs/finetune_corrected_from_paper.yaml`; degradations in `forknet/data/degradations.py` | |
| BasicVSR++ with the kernel mix | `scripts/baselines/train_basicvsrpp.py` (driver: `runpod/run_bvpp_km.py`) | `results/paper/lp_test/BasicVSR++-kernelmix.json`, `reds4/reds4_BasicVSR++-kernelmix*.json` |
| Figs. 4 and 5 (qualitative crops) | `scripts/qualitative_crops.py` | |
| Efficiency (Section VI-F) | `scripts/complexity.py --runtime` | |

Example, level 3:

```bash
python scripts/train.py --config configs/finetune_kernelmix_from_paper.yaml --set pretrained=weights/forknet_reds.pth
python scripts/test.py  --config configs/test_lp.yaml --checkpoint experiments/forknet_kernelmix_ft/checkpoints/final.pth
```

## 3. Released weights

Attached to the GitHub release `v1.0` with SHA-256 hashes:

| File | Model in the paper |
|---|---|
| `forknet_reds.pth` | Fork-Net (REDS-trained, the starting point of every fine-tune) |
| `forknet_kernelmix.pth` | Fork-Net, kernel mix |
| `forknet_benchmark_ft.pth` | Fork-Net, benchmark fine-tune (control) |
| `basicvsrpp_kernelmix.pth` | BasicVSR++, kernel mix |
| `forknet_lp_from_kernelmix.pth` | Fork-Net, LP fine-tune from kernel mix |
| `forknet_mixed.pth`, `forknet_percep.pth`, `forknet_lp_text.pth` | Table IV variants |
| `forknet_kernelmix_nocv2.pth`, `forknet_kernelmix_7frame.pth` (28.3k iterations), `forknet_kernelmix_seed1.pth` (29.7k iterations) | Table VI ablations |

Download them into `weights/` and check them with `sha256sum -c SHA256SUMS.txt`. Each file holds `params`, `config`
and `iter` only (no optimizer state); `forknet_reds.pth` uses the original propagation (`legacy_*` switches on), as
it was trained, and every fine-tune the corrected one. SpyNet weights ship in `weights/`. Baselines use their official weights (BasicVSR, BasicVSR++, RVRT, MIA-VSR,
RealBasicVSR).

## 4. Data

* **REDS** and **Vid4**: from their authors; `scripts/prepare_reds.py`.
* **RLPR**: from its authors, CC BY-NC-ND 4.0, non-commercial benchmarking only.
* **License plate video**: the clips are crops of publicly available traffic videos and show readable plates, so the
  frames and the verified plate strings are not redistributed openly. They are available for non-commercial research
  under the [data-use agreement](DATA_USE_AGREEMENT.md), which prohibits re-identification of vehicles or persons;
  datasheet in [DATASET.md](DATASET.md). The splits, including the leak-free `test_clean` and `val_clean` subsets and the reasons for
  each exclusion, are in `splits/lp_lpsr.json`.
