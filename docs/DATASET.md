# License-plate (LP) video dataset

Datasheet for the license-plate video benchmark of the paper (format after Gebru et al., *Datasheets for Datasets*).
The data are available for non-commercial research under the [data-use agreement](DATA_USE_AGREEMENT.md); this
repository holds the splits and the per-clip results, not the frames or the plate strings.

## Composition

| | |
|---|---|
| Clips | 282: 221 training, 30 validation, 31 test; one plate track per clip |
| HR frames | 256 × 256 px RGB crops centered on the plate; 9,698 frames in total |
| LR frames | 64 × 64 px, made from the HR frames with OpenCV's cubic resize (`cv2.INTER_CUBIC`, ×4, no anti-aliasing) |
| Clip length | 10–176 frames (median 30) over the 51 evaluation clips; 11–64 (median 36) over the 31 test clips |
| Plates | Taiwan, mainland China and India |
| Labels | hand-verified plate strings for the 51 evaluation clips |

## Collection and processing

The clips were cropped by the authors from publicly available traffic and CCTV videos on YouTube. Each clip follows one
plate; frames are 256 × 256 windows centered on it. The identifiers of the source videos were not retained.

LR frames were generated once with OpenCV's cubic kernel, which aliases when downsampling. This is the kernel the paper
studies, and the released LR frames are the ones used for every LP result. To regenerate them from the HR frames, use
`scripts/prepare_lp_dataset.py --kernel cv2_bicubic`; the script's default (`matlab`) gives the anti-aliased benchmark
kernel instead.

## Splits and leakage control

`splits/lp_lpsr.json` lists the splits and two leak-free evaluation subsets:

* `test_clean` (23 clips): test clips that share no frames with any training data. Test clip 029 shares frames with
  training clip 008, and seven test clips share frames with an earlier plate set on which the original fine-tuned
  checkpoint was trained.
* `val_clean` (19 clips): validation clips with no overlap with either training set.

The paper reports reconstruction metrics on `test_clean` (LP-test) and recognition on 51 clips (LP-rec): the 23
`test_clean` clips and 28 validation clips. The two validation clips that share frames with the LP training split
(000 and 008) are excluded. No model or hyperparameter was selected on these clips.

## Labels

Plate strings were read by two recognizers with temporal consensus and then checked by hand on the HR frames; 4 of the
31 test labels were corrected in that check. They are distributed with the data, not in this repository.

## Uses, distribution and maintenance

* **Intended use:** training and evaluating video super-resolution and plate-recognition pre-processing.
* **Not permitted:** identifying, locating or tracking vehicles or people; surveillance or law-enforcement use;
  commercial use (see the agreement).
* **Personal data:** license plates can be personal data in many jurisdictions. Access is limited to researchers who
  accept the agreement, and clips are removed on request.
* **Distribution:** on request from the corresponding author, Dr. T. D. Ambegoda (thanuja@cse.mrt.ac.lk), under the
  [data-use agreement](DATA_USE_AGREEMENT.md).
* **Maintenance and removal requests:** same contact.
