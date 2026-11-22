"""Collect metrics JSON files (from test.py / evaluate_outputs.py) into Markdown and LaTeX tables.

    python scripts/make_tables.py results/reds4/*/metrics.json results/lp/*.json --out results/tables.md
"""

import argparse
import json
from pathlib import Path


def row(path):
    with open(path) as f:
        r = json.load(f)
    name = r.get("method") or Path(r.get("checkpoint") or "bicubic").stem
    params = r.get("params", {}).get("params_total")
    out = {"method": name, "params_m": params / 1e6 if params else None, "file": str(path)}
    if "standard" in r:
        out.update(r["standard"]["average"])
    if "legacy" in r:
        out["legacy_psnr"], out["legacy_ssim"] = r["legacy"]["average"]["psnr"], r["legacy"]["average"]["ssim"]
    return out


def fmt(v, nd):
    return "-" if v is None else f"{v:.{nd}f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--out")
    args = p.parse_args()
    rows = [row(f) for f in args.files]
    cols = [("method", "Method", None), ("params_m", "Params (M)", 2), ("psnr_rgb", "PSNR-RGB", 2), ("ssim_rgb", "SSIM-RGB", 4),
            ("psnr_y", "PSNR-Y", 2), ("ssim_y", "SSIM-Y", 4), ("legacy_psnr", "PSNR (legacy)", 2), ("legacy_ssim", "SSIM (legacy)", 4)]
    cols = [c for c in cols if any(r.get(c[0]) is not None for r in rows)]
    md = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    tex = ["\\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", "\\hline", " & ".join(c[1] for c in cols) + " \\\\", "\\hline"]
    for r in rows:
        vals = [r[c[0]] if c[2] is None else fmt(r.get(c[0]), c[2]) for c in cols]
        md.append("| " + " | ".join(vals) + " |")
        tex.append(" & ".join(vals) + " \\\\")
    tex += ["\\hline", "\\end{tabular}"]
    text = "\n".join(md) + "\n\n```latex\n" + "\n".join(tex) + "\n```\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
