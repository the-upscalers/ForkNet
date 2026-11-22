Model-size study under the kernel-mix degradation (research/13 "smaller model"): does a narrower Fork-Net
generalise better from REDS to the LP benchmark? All runs from scratch, same schedule and data, only `num_feat`
differs. Pilot scale (7-frame clips, 20k iterations, seeds 0 and 1 via `--set seed=1 name=...`) for the curve;
full scale (15-frame clips, 40k iterations) for widths 32 and 64. Compare in-domain REDS4 vs shifted LP test_clean
(whole clip and 15-frame chunks) and plate recognition; the "generalisation gap" is the in-domain minus the shifted
score relative to bicubic.
