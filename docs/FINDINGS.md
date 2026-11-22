# Code audit: original research implementation

This document records what was found when the original research code was consolidated into this
repository. Each finding lists the evidence, the effect, and how this repository handles it.
Findings F1–F5 are behaviours of the original code. They are reproduced exactly by the `legacy_*`
switches, so existing checkpoints keep their behaviour, and fixed in the default configuration.

| ID | Severity | Summary | Config switch | Default |
|----|----------|---------|---------------|---------|
| F1 | **critical** | Optical flows used in the wrong direction | `model.legacy_flow_direction` | fixed |
| F2 | **critical** | Backward branch reads the wrong time steps | `model.legacy_backward_index` | fixed |
| F3 | major | Flow vectors not rescaled at the ½× level | `model.legacy_flow_scaling` | fixed |
| F4 | major | Latest original commit is an ablation variant | — | Fork-Net |
| F5 | minor | "Horizontal flip" augmentation swapped RGB↔BGR | `data.train.augment: legacy` | fixed |
| F6 | **critical for comparisons** | Non-standard evaluation protocol | `scripts/test.py --protocol` | standard |

`tests/test_legacy_equivalence.py` loads the original modules and checks that this repository's
`ForkNet` produces **identical outputs** from identical weights (max. abs. difference 0 to 1e-7).
Run it with `PIXEL_LEVEL_VSR_ROOT=<checkout of the original code>`.
`scripts/inspect_checkpoint.py` identifies which variant any original checkpoint is.

---

## F1: Optical flows are applied in the wrong direction (critical)

`ForkNetModule.compute_flows` computed `forward_flows_1 = spynet(x[:, :-1], x[:, 1:])`, i.e.
`SpyNet(ref = x_{t-1}, supp = x_t)`. SpyNet returns the flow that warps `supp` onto `ref`, so
this field maps frame *t* to frame *t−1*. The Tempo Fuser then used it to warp the propagated
feature of frame *t−1* towards frame *t*. Backward and second-order flows are swapped the same
way. BasicVSR uses `spynet(x[:, 1:], x[:, :-1])` for the forward branch.

Evidence (`tests/test_flow.py::test_forward_flow_direction`, real SpyNet weights, 4-pixel
horizontal motion):

| Warping frame *t−1* onto frame *t* | mean abs. error |
|---|---|
| no warping | 0.0438 |
| with the original `forward_1` | **0.0696** (worse than no warping) |
| with the corrected flow | 0.0050 |

Effect: instead of compensating motion, every warp roughly doubles the misalignment. The
network can only learn to down-weight the propagated features.

## F2: The backward branch reads the wrong time steps (critical)

The backward loop appends propagated features to a list in reverse time order and then reads
`processed_feats_backward[t - T + 2]` (first order) and `[t - T + 3]` (second order). These
indices are only correct for the last two frames. For every `t ≤ T−4` they resolve to the same
two entries, the propagated features of frames **T−2** and **T−3**:

```
T = 7: t=3 reads frames 5,4 (should be 4,5); t=2 reads 5,4 (should be 3,4); t=0 reads 5,4 (should be 1,2)
```

With the 15-frame training clips, frames 0–11 all receive frame 13's and 12's features, warped
with flows that belong to other frame pairs, so the backward direction carries almost no useful
temporal information.
`tests/test_forknet.py::test_legacy_backward_indexing_skips_next_frame` demonstrates that with the
original indexing frame 0 is completely independent of frame 1. The fix reads `[-1]` and `[-2]`,
as BasicVSR++ does.

## F3: Flow magnitudes are not rescaled at the ½× level (major)

`interpolate_flow` bilinearly resizes the LR-resolution flow to the ½× feature map
(`tempo_fuser_32`) without multiplying the vectors by ½, so displacements there are 2× too large.
Fixed by `resize_flow(..., rescale_values=True)`.

## F4: The latest original commit is an ablation variant (major)

The latest original commit trains a model with `num_grid_blocks = 5`, plain residual blocks (no
ConvNeXt), plain-convolution fusion and upsampling, fusion of `[H_fwd, F, H_bwd]` and a C-channel
bottleneck. It has **4,422,339 trainable parameters excluding SpyNet**. It is not the default
Fork-Net. In the clean code it is
`ForkNetConfig(prop_block="resblock", bsconv_fusion=False, fuse_with_input=True, bottleneck_mult=1)`.

## F5: Augmentation flipped colour channels instead of width (minor)

`REDSDataset.augment` flipped dims 2 and 1 of a `(T, C, H, W)` tensor, i.e. a vertical flip and
an **RGB↔BGR channel reversal**, but no horizontal flip. `augment: legacy` reproduces it and
`augment: flips` (default) does horizontal and vertical flips.

## F6: Evaluation protocol (critical for comparisons)

The original `validate()` did not follow the protocol used by BasicVSR, IconVSR, BasicVSR++ and
MIA-VSR, so numbers from it are not comparable with those methods:

| | Original Fork-Net code | Standard (this repo, `--protocol standard`) |
|---|---|---|
| Frames | 15-frame disjoint chunks; frames 90–99 of each REDS4 clip never evaluated | all 100 frames |
| Output | float, not clamped or quantised, fp16 autocast | clamped to [0, 1], rounded to uint8 |
| PSNR | one value per 15-frame chunk; `data_range = max(GT) − min(GT)` (torchmetrics default) | per frame, data range 255, mean over frames then clips |
| SSIM | torchmetrics (float RGB) | BasicSR / MATLAB-style (11×11 Gaussian, σ 1.5, valid window) on uint8 |

`scripts/test.py --protocol both` reports both. When comparing methods, **every method must be
scored by the same code**: generate each baseline's outputs and run `scripts/evaluate_outputs.py`.
`scripts/baselines/basicvsr.py` shows how and reproduces BasicVSR's published REDS4 result (see
REPRODUCIBILITY.md).
