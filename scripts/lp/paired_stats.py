"""Paired statistics for per-clip results: bootstrap 95% CIs, exact McNemar (plate read / not read) and a sign-flip
permutation test on per-clip PSNR differences. Inputs are the JSON files written by ocr_eval.py and
evaluate_outputs.py; no GPU or torch needed.

    python scripts/lp/paired_stats.py --ocr results/lp_ocr_fast_plate_ocr.json \\
        --quality results/lp_test_clean_full_{method}.json --reference Bicubic --out stats.json
"""

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(0)
B = 10000


def boot_ci(x):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for b fixes vs c breaks."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def signflip_p(d):
    d = np.asarray(d, float)
    signs = RNG.choice([-1.0, 1.0], (B, len(d)))
    null = np.abs((signs * d).mean(1))
    return float((np.sum(null >= abs(d.mean())) + 1) / (B + 1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ocr", nargs="+", default=[], help="ocr_eval.py outputs (one per engine)")
    p.add_argument("--quality", default=None, help="glob of evaluate_outputs.py outputs (one per method)")
    p.add_argument("--reference", default="Bicubic")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = {"ocr": {}, "quality": {}}

    for f in args.ocr:
        d = json.loads(Path(f).read_text())
        methods = d["methods"]
        clips = sorted(methods[args.reference]["clips"])
        res = {}
        for name, m in methods.items():
            corr = [float(m["clips"][c]["correct"]) for c in clips if c in m["clips"]]
            r = {"clip_exact": boot_ci(corr), "n": len(corr)}
            for other in methods:
                if other == name or other == "HR":
                    continue
                o = methods[other]["clips"]
                common = [c for c in clips if c in m["clips"] and c in o]
                b = sum(m["clips"][c]["correct"] and not o[c]["correct"] for c in common)
                cc = sum(o[c]["correct"] and not m["clips"][c]["correct"] for c in common)
                r[f"vs_{other}"] = {"fixes": b, "breaks": cc, "mcnemar_p": mcnemar_exact(b, cc)}
            res[name] = r
        out["ocr"][d["engine"]] = res

    if args.quality:
        files = sorted(glob.glob(args.quality))
        per = {json.loads(Path(f).read_text())["method"]: json.loads(Path(f).read_text())["standard"]["clips"] for f in files}
        for name, clips in per.items():
            r = {k: boot_ci([v[k] for v in clips.values()]) for k in ("psnr_rgb", "ssim_rgb", "lpips", "tof")}
            for other, oc in per.items():
                if other == name:
                    continue
                common = sorted(set(clips) & set(oc))
                d = [clips[c]["psnr_rgb"] - oc[c]["psnr_rgb"] for c in common]
                r[f"psnr_vs_{other}"] = {"mean_diff": boot_ci(d), "signflip_p": signflip_p(d),
                                         "clips_better": int(sum(x > 0 for x in d)), "n": len(d)}
            out["quality"][name] = r

    Path(args.out).write_text(json.dumps(out, indent=1))
    for eng, res in out["ocr"].items():
        print(f"[{eng}]")
        for name, r in res.items():
            m, lo, hi = r["clip_exact"]
            print(f"  {name:22s} exact {m:.3f} [{lo:.3f}, {hi:.3f}]  " + "  ".join(
                f"vs {o[3:]}: +{v['fixes']}/-{v['breaks']} p={v['mcnemar_p']:.3f}" for o, v in r.items() if o.startswith("vs_")))
    for name, r in out["quality"].items():
        m, lo, hi = r["psnr_rgb"]
        print(f"{name:22s} PSNR {m:.2f} [{lo:.2f}, {hi:.2f}]  " + "  ".join(
            f"vs {o[8:]}: {v['mean_diff'][0]:+.2f} [{v['mean_diff'][1]:+.2f},{v['mean_diff'][2]:+.2f}] p={v['signflip_p']:.4f} ({v['clips_better']}/{v['n']})"
            for o, v in r.items() if o.startswith("psnr_vs_")))


if __name__ == "__main__":
    main()
