"""Generalisation analysis from a robustness sweep (scripts/degradation_sweep.py output).

For every method m and degradation level d (with "clean" = the training degradation):
    gain(m, d)  = PSNR(m, d) - PSNR(Bicubic, d)        benefit over doing nothing
    gap(m, d)   = gain(m, clean) - gain(m, d)          how much of the benefit is lost under shift
    below(m, d) = gain(m, d) < 0                       the model is worse than bicubic interpolation
Across levels: mean / worst gain, mean gap, number of levels below bicubic, and Kendall's tau between the
in-domain ranking and the ranking under each shift (tau < 1: the benchmark ranking does not transfer).
Optionally per-clip paired Wilcoxon tests of method A vs. B on each level.

    python scripts/generalization_report.py results/robustness_lp.json --params ForkNet-paper=4.37 BasicVSR=6.3 \\
        --pair ForkNet-paper BasicVSR --out results/generalization_lp.md
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def kendall_tau(a, b):
    """Kendall's tau-a between two equally ordered score lists."""
    pairs = list(combinations(range(len(a)), 2))
    if not pairs:
        return float("nan")
    s = sum(np.sign(a[i] - a[j]) * np.sign(b[i] - b[j]) for i, j in pairs)
    return float(s / len(pairs))


def wilcoxon_p(x, y):
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(x, y).pvalue)
    except Exception:  # scipy missing or all differences zero
        return float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep", help="JSON written by scripts/degradation_sweep.py")
    p.add_argument("--clean", default="clean")
    p.add_argument("--reference", default="Bicubic")
    p.add_argument("--metric", default="psnr_rgb")
    p.add_argument("--params", nargs="*", default=[], metavar="METHOD=MILLIONS", help="model sizes for the size column")
    p.add_argument("--pair", nargs=2, metavar=("A", "B"), help="per-clip paired test of A vs. B on every level")
    p.add_argument("--out")
    args = p.parse_args()

    res = json.loads(Path(args.sweep).read_text())
    levels = [lv for lv in res if lv != args.clean]
    methods = [m for m in res[args.clean] if m != args.reference]
    size = dict(kv.split("=") for kv in args.params)
    score = lambda m, lv: res[lv][m][args.metric]  # noqa: E731
    gain = {m: {lv: score(m, lv) - score(args.reference, lv) for lv in [args.clean] + levels} for m in methods}

    lines = [f"# Generalisation report ({args.metric}, reference {args.reference})", "",
             "| Method | Params (M) | gain clean | mean gain shifted | worst gain | mean gap | levels below ref. |",
             "|---|---|---|---|---|---|---|"]
    for m in sorted(methods, key=lambda m: -np.mean([gain[m][lv] for lv in levels])):
        g = [gain[m][lv] for lv in levels]
        gap = [gain[m][args.clean] - x for x in g]
        lines.append(f"| {m} | {size.get(m, '-')} | {gain[m][args.clean]:+.2f} | {np.mean(g):+.2f} | {min(g):+.2f} "
                     f"| {np.mean(gap):.2f} | {sum(x < 0 for x in g)}/{len(g)} |")

    lines += ["", "## Gain over the reference per level", "", "| Level | " + " | ".join(methods) + " |",
              "|" + "---|" * (len(methods) + 1)]
    for lv in [args.clean] + levels:
        lines.append(f"| {lv} | " + " | ".join(f"{gain[m][lv]:+.2f}" for m in methods) + " |")

    if len(methods) > 2:
        clean_scores = [score(m, args.clean) for m in methods]
        taus = {lv: kendall_tau(clean_scores, [score(m, lv) for m in methods]) for lv in levels}
        lines += ["", "## Does the in-domain ranking transfer? (Kendall's tau vs. clean)", "",
                  "| Level | tau |", "|---|---|"] + [f"| {lv} | {t:+.2f} |" for lv, t in taus.items()]
        lines.append(f"\nMean tau over shifted levels: {np.mean(list(taus.values())):+.2f}")

    if args.pair:
        a, b = args.pair
        lines += ["", f"## Paired per-clip comparison: {a} vs. {b}", "", "| Level | mean diff (dB) | clips A better | Wilcoxon p |",
                  "|---|---|---|---|"]
        for lv in [args.clean] + levels:
            clips = sorted(res[lv][a]["clips"])
            x = np.array([res[lv][a]["clips"][c][args.metric] for c in clips])
            y = np.array([res[lv][b]["clips"][c][args.metric] for c in clips])
            lines.append(f"| {lv} | {np.mean(x - y):+.2f} | {int((x > y).sum())}/{len(clips)} | {wilcoxon_p(x, y):.3g} |")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
