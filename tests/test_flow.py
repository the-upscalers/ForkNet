import pytest
import torch

from conftest import SPYNET_WEIGHTS, smooth_texture
from forknet import ForkNet, ForkNetConfig
from forknet.archs.flow import flow_warp, resize_flow
from forknet.archs.spynet import SpyNet


def test_flow_warp_zero_flow_is_identity():
    x = torch.rand(2, 3, 9, 13)
    torch.testing.assert_close(flow_warp(x, torch.zeros(2, 9, 13, 2)), x)


def test_flow_warp_integer_shift():
    x = torch.rand(1, 1, 8, 8)
    flow = torch.zeros(1, 8, 8, 2)
    flow[..., 0] = 1.0  # out(p) = x(p + (1, 0))
    out = flow_warp(x, flow)
    torch.testing.assert_close(out[..., :, :-1], x[..., :, 1:])


def test_resize_flow_rescales_vectors():
    flow = torch.ones(1, 3, 2, 8, 8) * 4.0
    half = resize_flow(flow, (4, 4), rescale_values=True)
    assert half.shape == (1, 3, 2, 4, 4)
    torch.testing.assert_close(half, torch.full_like(half, 2.0))
    torch.testing.assert_close(resize_flow(flow, (4, 4), rescale_values=False), torch.full_like(half, 4.0))


@pytest.fixture(scope="module")
def shifted_pair():
    img = torch.from_numpy(smooth_texture(160, 288)).permute(2, 0, 1).float()[None] / 255
    d = 4  # content moves +4 px to the right from frame 0 to frame 1
    prev, cur = img[..., 16:-16], img[..., 16 - d : -16 - d]
    return torch.stack([prev, cur], dim=1)  # (1, 2, 3, H, W)


@pytest.mark.skipif(not SPYNET_WEIGHTS.exists(), reason="SpyNet weights missing")
def test_pretrained_spynet_estimates_known_shift(shifted_pair):
    spynet = SpyNet(str(SPYNET_WEIGHTS)).eval()
    prev, cur = shifted_pair[:, 0], shifted_pair[:, 1]
    with torch.no_grad():
        flow = spynet(cur, prev)  # warps prev onto cur: out(p) = prev(p + f) -> f_x ~ -4
    assert flow[:, 0, 16:-16, 16:-16].mean().item() == pytest.approx(-4.0, abs=0.5)


@pytest.mark.skipif(not SPYNET_WEIGHTS.exists(), reason="SpyNet weights missing")
@pytest.mark.parametrize("legacy", [False, True])
def test_forward_flow_direction(shifted_pair, legacy):
    """forward_1 must align frame t-1 to frame t. The legacy direction makes alignment worse
    than not warping at all (docs/FINDINGS.md, F1)."""
    model = ForkNet(ForkNetConfig(legacy_flow_direction=legacy), spynet_path=str(SPYNET_WEIGHTS)).eval()
    with torch.no_grad():
        flows = model.compute_flows(shifted_pair)
    prev, cur = shifted_pair[:, 0], shifted_pair[:, 1]
    crop = lambda z: z[..., 16:-16, 16:-16]  # noqa: E731
    err_none = (crop(prev) - crop(cur)).abs().mean()
    err_warp = (crop(flow_warp(prev, flows["forward_1"][:, 0].permute(0, 2, 3, 1))) - crop(cur)).abs().mean()
    if legacy:
        assert err_warp > err_none
    else:
        assert err_warp < 0.25 * err_none
