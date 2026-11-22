"""Sequence-level inference with temporal chunking."""

import torch


def temporal_windows(num_frames, chunk_size=None, overlap=0):
    """Splits ``range(num_frames)`` into windows and assigns each frame to one window.

    Returns a list of ``(start, end, keep)`` where ``keep`` lists the frame indices whose
    output is taken from that window.

    * ``chunk_size`` None or >= num_frames: one window over the whole sequence.
    * ``overlap == 0``: consecutive disjoint chunks; the last one may be shorter.
      With ``chunk_size=15`` this matches the original validation for the first
      ``15 * (T // 15)`` frames.
    * ``overlap > 0``: sliding windows with stride ``chunk_size - overlap`` (last one
      aligned to the end); each frame is taken from the window where it is farthest
      from a window border, i.e. has the most temporal context on both sides.
    """
    if not chunk_size or chunk_size >= num_frames:
        return [(0, num_frames, list(range(num_frames)))]
    if overlap == 0:
        starts = range(0, num_frames, chunk_size)
        return [(s, min(s + chunk_size, num_frames), list(range(s, min(s + chunk_size, num_frames)))) for s in starts]
    if not 0 < overlap < chunk_size:
        raise ValueError(f"overlap must be in [0, chunk_size), got {overlap}")
    stride = chunk_size - overlap
    starts = list(range(0, num_frames - chunk_size + 1, stride))
    if starts[-1] + chunk_size < num_frames:
        starts.append(num_frames - chunk_size)
    owner = {}
    for s in starts:
        for t in range(s, s + chunk_size):
            margin = min(t - s, s + chunk_size - 1 - t)
            if t not in owner or margin > owner[t][1]:
                owner[t] = (s, margin)
    return [(s, s + chunk_size, [t for t in range(s, s + chunk_size) if owner[t][0] == s]) for s in starts]


def _tile_starts(size, tile, overlap):
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, tile - 2 * overlap))
    if starts[-1] + tile < size:
        starts.append(size - tile)
    return starts


def _forward_tiled(model, lrs, tile, tile_overlap, scale=4):
    """Runs ``model`` on overlapping spatial tiles of ``lrs`` (N, T, C, h, w); each output pixel is taken
    from the tile in which it lies farthest from a tile border (no blending), so tiles only meet in
    regions that had ``tile_overlap`` LR pixels of context on both sides."""
    h, w = lrs.shape[-2:]
    ys, xs = _tile_starts(h, tile, tile_overlap), _tile_starts(w, tile, tile_overlap)
    out = None
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            sr = model(lrs[..., y : y + tile, x : x + tile]).float()
            if out is None:
                out = sr.new_empty((*sr.shape[:3], h * scale, w * scale))
            # Interior of this tile: halfway into the overlap with each neighbour.
            y0 = 0 if i == 0 else (y + (ys[i - 1] + tile)) // 2
            y1 = h if i == len(ys) - 1 else (ys[i + 1] + y + tile) // 2
            x0 = 0 if j == 0 else (x + (xs[j - 1] + tile)) // 2
            x1 = w if j == len(xs) - 1 else (xs[j + 1] + x + tile) // 2
            out[..., y0 * scale : y1 * scale, x0 * scale : x1 * scale] = \
                sr[..., (y0 - y) * scale : (y1 - y) * scale, (x0 - x) * scale : (x1 - x) * scale]
    return out


@torch.no_grad()
def upscale_sequence(model, lrs, chunk_size=None, overlap=0, amp=False, output_device="cpu", tile=None, tile_overlap=16):
    """Super-resolves a whole sequence.

    Args:
        lrs: (1, T, 3, h, w) tensor on the model's device, values in [0, 1].
        tile: if set, frames are processed in overlapping ``tile`` x ``tile`` LR tiles (for large inputs).
    Returns:
        (1, T, 3, 4h, 4w) float tensor on ``output_device`` (not clamped).
    """
    t_len = lrs.size(1)
    out = None
    device_type = lrs.device.type
    for start, end, keep in temporal_windows(t_len, chunk_size, overlap):
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp and device_type == "cuda"):
            if tile:
                sr = _forward_tiled(model, lrs[:, start:end], tile, tile_overlap)
            else:
                sr = model(lrs[:, start:end]).float()
        if out is None:
            out = torch.empty((1, t_len, *sr.shape[2:]), dtype=torch.float32, device=output_device)
        idx = torch.tensor(keep) - start
        out[:, keep] = sr[:, idx].to(output_device)
    return out
