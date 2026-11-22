"""Check the numbers printed in the paper against the released result files in results/paper/.

Runs on CPU in seconds and needs no data, weights or GPU:

    python scripts/verify_paper_results.py

Covers Table II (LP-test reconstruction, LP-rec recognition), Table III (RLPR recognition), Table IV (variants) and
the exact McNemar tests quoted in the text. The per-clip records hold correctness only; the plate strings are
available under the data-use agreement (see docs/PAPER.md).
"""
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "paper"

# Paper value -> (file stem in results/paper/lp_test, PSNR, SSIM, LPIPS, tOF)
LP_TEST = {
    "Bicubic": ("Bicubic", 24.39, 0.834, 0.289, 1.33),
    "BasicVSR": ("BasicVSR", 17.57, 0.588, 0.393, 3.76),
    "BasicVSR++": ("BasicVSR++", 20.23, 0.708, 0.248, 1.35),
    "RVRT": ("RVRT", 20.76, 0.748, 0.216, 1.37),
    "MIA-VSR": ("MIA-VSR", 20.30, 0.725, 0.232, 1.35),
    "RealBasicVSR": ("RealBasicVSR", 22.67, 0.815, 0.187, 1.68),
    "Fork-Net": ("ForkNet-paper", 21.29, 0.786, 0.200, 1.35),
    "Fork-Net, benchmark fine-tune": ("Ours-corrected", 21.09, 0.777, 0.205, 1.35),
    "Fork-Net, kernel mix": ("Ours-kernelmix", 26.68, 0.899, 0.122, 1.09),
    "BasicVSR++, kernel mix": ("BasicVSR++-kernelmix", 29.64, 0.935, 0.075, 0.85),
    "Fork-Net, LP fine-tune": ("Ours-LPft", 29.16, 0.931, 0.088, 0.74),
    "(i) mixed degradations": ("Ours-mixed", 26.45, None, 0.129, None),
    "(ii) + LPIPS and GAN": ("Ours-percep", 25.32, None, 0.127, None),
    "(iii) + text-feature loss": ("Ours-LPft-text", 29.13, None, 0.091, None),
    "without OpenCV cubic": ("Ours-kernelmix-nocv2", 23.18, None, None, None),
}

# LP-rec, 51 clips, clip exact match (%): method -> (plate recognizer, EasyOCR)
LP_REC = {
    "HR": (86.3, 31.4), "Bicubic": (35.3, 9.8), "BasicVSR": (25.5, 9.8), "BasicVSR++": (27.5, 11.8),
    "RVRT": (33.3, 9.8), "MIA-VSR": (31.4, 7.8), "RealBasicVSR": (45.1, 15.7), "Fork-Net": (33.3, 11.8),
    "Fork-Net, benchmark fine-tune": (33.3, 7.8), "Fork-Net, kernel mix": (54.9, 21.6),
    "BasicVSR++, kernel mix": (74.5, 23.5), "Fork-Net, LP fine-tune": (56.9, 19.6),
    "Fork-Net, mixed": (52.9, None), "Fork-Net, LPIPS+GAN": (47.1, None), "Fork-Net, LP fine-tune + text": (60.8, None),
}
# (method, reference, fixes, breaks, p as printed or None)
LP_REC_TESTS = [
    ("Fork-Net, kernel mix", "Bicubic", 10, 0, 0.002), ("Fork-Net, kernel mix", "Fork-Net", 11, 0, 0.001),
    ("BasicVSR++, kernel mix", "Bicubic", 20, 0, None), ("BasicVSR++, kernel mix", "BasicVSR++", 24, 0, None),
    ("RealBasicVSR", "Bicubic", 6, 1, 0.13), ("Fork-Net, kernel mix", "RealBasicVSR", 7, 2, 0.18),
    ("Fork-Net, kernel mix", "RVRT", 11, 0, None), ("Fork-Net, kernel mix", "MIA-VSR", 12, 0, None),
    ("BasicVSR++, kernel mix", "Fork-Net, kernel mix", 11, 1, 0.006),
    ("Fork-Net, LP fine-tune", "Bicubic", 11, 0, 0.001), ("Fork-Net, LP fine-tune", "Fork-Net, kernel mix", 2, 1, None),
    ("BasicVSR++, kernel mix", "Fork-Net, LP fine-tune", 10, 1, 0.012),
    ("Fork-Net, mixed", "Fork-Net, kernel mix", 1, 2, None), ("Fork-Net, LPIPS+GAN", "Fork-Net, kernel mix", 2, 6, 0.29),
    ("Fork-Net, LP fine-tune + text", "Fork-Net, LP fine-tune", 4, 2, 0.69),
]

# RLPR, 200 sequences, plate recognizer clip exact (%), and McNemar against bicubic (fixes, breaks, p)
RLPR = {
    "Pseudo-GT": (83.0, None), "Bicubic": (9.0, None), "MF-LPR2": (5.0, (5, 13, 0.10)),
    "BasicVSR": (10.0, (8, 6, 0.79)), "RVRT": (10.0, (8, 6, 0.79)), "MIA-VSR": (7.5, (4, 7, 0.55)),
    "RealBasicVSR": (4.0, (1, 11, 0.006)), "ForkNet-paper": (9.0, (6, 6, 1.00)),
    "Ours-corrected": (9.5, (7, 6, 1.00)), "Ours-kernelmix": (8.0, (5, 7, 0.77)),
    "Ours-kernelmix-7frame": (8.0, (5, 7, 0.77)), "Ours-mixed": (11.5, (12, 7, 0.36)),
    "Ours-percep": (5.0, (2, 10, 0.039)), "Ours-LPft": (9.5, (4, 3, 1.00)), "Ours-LPft-text": (9.0, (5, 5, 1.00)),
}

failures = []


def check(name, got, want, decimals):
    if want is None:
        return
    ok = got is not None and abs(got - want) <= 0.5 * 10 ** -decimals + 1e-9  # printed value, rounded half up
    if not ok:
        failures.append(f"{name}: paper {want}, files {got}")
    print(f"  {'ok ' if ok else 'BAD'} {name:58s} paper {want:<8} files {got if got is None else round(got, decimals + 2)}")


def mcnemar(a, b):
    """Fixes and breaks of method a against reference b, and the exact two-sided McNemar p-value."""
    fixes = sum(a[c] and not b[c] for c in a)
    breaks = sum(b[c] and not a[c] for c in a)
    n, k = fixes + breaks, min(fixes, breaks)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return fixes, breaks, p


def p_decimals(p):
    return 3 if p < 0.01 else 2


def load(rel):
    return json.load(open(ROOT / rel, encoding="utf-8"))


def main():
    print("Table II / IV / VI, LP-test (23 clips): PSNR, SSIM, LPIPS, tOF")
    for name, (stem, psnr, ssim, lpips, tof) in LP_TEST.items():
        avg = load(f"lp_test/{stem}.json")["standard"]["average"]
        check(f"{name} PSNR", avg["psnr_rgb"], psnr, 2)
        check(f"{name} SSIM", avg["ssim_rgb"], ssim, 3)
        check(f"{name} LPIPS", avg.get("lpips"), lpips, 3)
        check(f"{name} tOF", avg.get("tof"), tof, 2)

    print("\nTable II / IV, LP-rec (51 clips): clip exact match (%)")
    ocr = {e: load(f"lp_rec/ocr_{e}.json")["methods"] for e in ("fast_plate_ocr", "easyocr")}
    for name, (fpo, easy) in LP_REC.items():
        check(f"{name} plate recognizer", 100 * ocr["fast_plate_ocr"][name]["clip_exact"], fpo, 1)
        if easy is not None:
            check(f"{name} EasyOCR", 100 * ocr["easyocr"][name]["clip_exact"], easy, 1)
    print("\nLP-rec exact McNemar tests (plate recognizer)")
    clips = {m: v["clips"] for m, v in ocr["fast_plate_ocr"].items()}
    for a, b, fx, br, p in LP_REC_TESTS:
        gf, gb, gp = mcnemar(clips[a], clips[b])
        check(f"{a} vs {b} fixes", gf, fx, 0)
        check(f"{a} vs {b} breaks", gb, br, 0)
        if p is not None:
            check(f"{a} vs {b} p", gp, p, p_decimals(p))

    print("\nTable III, RLPR (200 sequences): plate recognizer clip exact match (%) and McNemar vs bicubic")
    rlpr = load("rlpr/ocr_fast_plate_ocr.json")["methods"]
    for name, (acc, test) in RLPR.items():
        check(f"{name}", 100 * rlpr[name]["clip_exact"], acc, 1)
        if test is not None:
            gf, gb, gp = mcnemar(rlpr[name]["clips"], rlpr["Bicubic"]["clips"])
            check(f"{name} vs Bicubic fixes/breaks", 100 * gf + gb, 100 * test[0] + test[1], 0)
            check(f"{name} vs Bicubic p", gp, test[2], p_decimals(test[2]))

    print(f"\n{len(failures)} mismatch(es)")
    for f in failures:
        print("  " + f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
