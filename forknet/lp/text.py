"""Plate-string utilities: normalisation, edit distance and temporal consensus."""

import re
from collections import Counter, defaultdict


def normalize(text):
    """Upper-case alphanumerics only (plate separators, spaces and OCR padding removed)."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_accuracy(pred, gt):
    """1 - CER, clipped at 0. Both strings are normalised first."""
    pred, gt = normalize(pred), normalize(gt)
    if not gt:
        return float(not pred)
    return max(0.0, 1.0 - levenshtein(pred, gt) / len(gt))


def consensus(readings):
    """Clip-level plate string from per-frame OCR readings.

    ``readings``: iterable of ``(text, confidence)``. Readings are normalised; the most common
    length is kept and each character position is decided by confidence-weighted voting.
    Returns ``(text, support, agreement)``: support = readings of the winning length, agreement =
    share of those readings identical to the consensus string.
    """
    readings = [(normalize(t), float(c)) for t, c in readings]
    readings = [(t, c) for t, c in readings if t]
    if not readings:
        return "", 0, 0.0
    length = Counter(len(t) for t, _ in readings).most_common(1)[0][0]
    same = [(t, c) for t, c in readings if len(t) == length]
    votes = [defaultdict(float) for _ in range(length)]
    for t, c in same:
        for i, ch in enumerate(t):
            votes[i][ch] += max(c, 1e-3)
    text = "".join(max(v, key=v.get) for v in votes)
    agreement = sum(t == text for t, _ in same) / len(same)
    return text, len(same), agreement
