import pytest
import torch

from forknet import ForkNet, ForkNetConfig
from forknet.inference import temporal_windows, upscale_sequence


@pytest.mark.parametrize("t,chunk,overlap", [(100, 15, 0), (100, 15, 4), (7, 15, 0), (16, 15, 2), (30, 10, 5)])
def test_every_frame_assigned_once(t, chunk, overlap):
    windows = temporal_windows(t, chunk, overlap)
    kept = sorted(k for _, _, keep in windows for k in keep)
    assert kept == list(range(t))
    for s, e, keep in windows:
        assert all(s <= k < e for k in keep) and e - s <= chunk


def test_disjoint_chunks_match_legacy_layout():
    windows = temporal_windows(100, 15, 0)
    assert [(s, e) for s, e, _ in windows] == [(i, min(i + 15, 100)) for i in range(0, 100, 15)]


def test_overlap_prefers_central_frames():
    windows = temporal_windows(20, 10, 4)
    owner = {k: s for s, _, keep in windows for k in keep}
    assert [(s, e) for s, e, _ in windows] == [(0, 10), (6, 16), (10, 20)]
    assert owner[7] == 0  # margins: 2 in [0, 10) vs 1 in [6, 16)
    assert owner[9] == 6  # margins: 0 in [0, 10) vs 3 in [6, 16)


def test_upscale_sequence_whole_equals_model():
    model = ForkNet(ForkNetConfig(num_feat=8, num_encoder_blocks=1, num_prop_blocks=2)).eval()
    x = torch.rand(1, 6, 3, 64, 64)
    with torch.no_grad():
        ref = model(x)
    torch.testing.assert_close(upscale_sequence(model, x, chunk_size=None), ref)
    chunked = upscale_sequence(model, x, chunk_size=3)
    with torch.no_grad():
        torch.testing.assert_close(chunked[:, 3:], model(x[:, 3:]))


@pytest.mark.parametrize("h, w, tile, ov", [(64, 96, 32, 4), (70, 50, 24, 6), (20, 20, 32, 4)])
def test_tiled_stitching_is_exact_for_a_local_model(h, w, tile, ov):
    def model(x):  # pixel-local x4 "SR": tiling must reproduce it exactly
        n, t, c, hh, ww = x.shape
        return torch.nn.functional.interpolate(x.flatten(0, 1), scale_factor=4, mode="nearest").view(n, t, c, hh * 4, ww * 4)

    x = torch.rand(1, 4, 3, h, w)
    torch.testing.assert_close(upscale_sequence(model, x, tile=tile, tile_overlap=ov), model(x))


def test_tiled_forknet_close_to_whole_frame():
    model = ForkNet(ForkNetConfig(num_feat=8, num_encoder_blocks=1, num_prop_blocks=2)).eval()
    x = torch.rand(1, 3, 3, 64, 96)
    whole, tiled = upscale_sequence(model, x), upscale_sequence(model, x, tile=64, tile_overlap=16)
    assert tiled.shape == whole.shape
    assert (tiled - whole).abs().mean() < 0.05
