import pytest
import torch

from forknet import ForkNet, ForkNetConfig
from forknet.archs.forknet import TempoFuser

SMALL = dict(num_feat=8, num_encoder_blocks=1, num_prop_blocks=2)

# Exact parameter counts (including SpyNet's 1,440,300) for the configurations in the paper.
GOLDEN_PARAMS = {
    "main": ({}, 4_373_039),
    "first_order": ({"prop_order": 1}, 4_307_503),
    "no_convnext": ({"prop_block": "resblock"}, 6_967_855),
    "no_bsconv": ({"use_bsconv": False}, 10_362_799),
    "original_final_commit": (
        {"prop_block": "resblock", "bsconv_fusion": False, "fuse_with_input": True, "bottleneck_mult": 1},
        5_862_639,
    ),
}


@pytest.mark.parametrize("name", list(GOLDEN_PARAMS))
def test_parameter_counts(name):
    kwargs, expected = GOLDEN_PARAMS[name]
    model = ForkNet(ForkNetConfig(**kwargs))
    assert model.num_params() == expected
    assert model.num_params() - model.num_params(include_spynet=False) == 1_440_300


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("order", [1, 2])
def test_output_shape(legacy, order):
    flags = dict(legacy_flow_direction=legacy, legacy_backward_index=legacy, legacy_flow_scaling=legacy)
    model = ForkNet(ForkNetConfig(prop_order=order, **SMALL, **flags)).eval()
    x = torch.rand(1, 5, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 5, 3, 256, 256)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("t", [1, 2, 3])
def test_short_sequences(t):
    model = ForkNet(ForkNetConfig(**SMALL)).eval()
    with torch.no_grad():
        assert model(torch.rand(1, t, 3, 64, 64)).shape == (1, t, 3, 256, 256)


def test_odd_input_size_is_padded_and_cropped():
    model = ForkNet(ForkNetConfig(**SMALL)).eval()
    with torch.no_grad():
        assert model(torch.rand(1, 3, 3, 65, 67)).shape == (1, 3, 3, 260, 268)


def test_zero_residual_gives_bicubic():
    """With a zeroed final conv the network reduces to bicubic upsampling (global skip of Fig. 1)."""
    model = ForkNet(ForkNetConfig(**SMALL)).eval()
    torch.nn.init.zeros_(model.u_up.final_conv.weight)
    torch.nn.init.zeros_(model.u_up.final_conv.bias)
    x = torch.rand(1, 3, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    ref = torch.nn.functional.interpolate(x[0], scale_factor=4, mode="bicubic", align_corners=False)[None]
    torch.testing.assert_close(y, ref)


def test_batch_independence():
    model = ForkNet(ForkNetConfig(**SMALL)).eval()
    x = torch.rand(2, 3, 3, 64, 64)
    with torch.no_grad():
        both, first = model(x), model(x[:1])
    torch.testing.assert_close(both[:1], first, atol=1e-5, rtol=1e-5)


def test_gradients_reach_all_trainable_parameters():
    model = ForkNet(ForkNetConfig(**SMALL))
    model(torch.rand(1, 3, 3, 64, 64)).mean().backward()
    missing = [n for n, p in model.named_parameters() if not n.startswith("spynet.") and p.grad is None]
    assert not missing, missing


def test_config_validation():
    with pytest.raises(ValueError):
        ForkNetConfig(prop_order=3)
    with pytest.raises(ValueError):
        ForkNetConfig(prop_block="transformer")
    with pytest.raises(ValueError):
        ForkNetConfig.from_dict({"unknown_key": 1})
    assert ForkNetConfig(use_bsconv=False).bsconv_fusion is False
    cfg = ForkNetConfig(prop_order=1)
    assert ForkNetConfig.from_dict(cfg.to_dict()) == cfg


def _backward_dependency(legacy, t_out=0, t_in=1, t_len=6):
    """Does the Tempo Fuser output at t_out depend on the input feature at t_in? (zero flows)"""
    cfg = ForkNetConfig(**SMALL, legacy_backward_index=legacy)
    fuser = TempoFuser(8, cfg).eval()
    feats = torch.rand(1, t_len, 8, 8, 8)
    flows = {f"{d}_{k}": torch.zeros(1, t_len - k, 2, 8, 8) for d in ("forward", "backward") for k in (1, 2)}
    with torch.no_grad():
        base = fuser(feats, flows)
        feats2 = feats.clone()
        feats2[:, t_in] += 1.0
        changed = fuser(feats2, flows)
    return not torch.allclose(base[:, t_out], changed[:, t_out])


def test_backward_propagation_reaches_next_frame():
    """Corrected: frame 0 receives information from frame 1 via the backward branch."""
    assert _backward_dependency(legacy=False)


def test_legacy_backward_indexing_skips_next_frame():
    """Legacy: frame 0 reads the propagated features of frames T-2 and T-3 instead of 1 and 2,
    so it never sees frame 1 (docs/FINDINGS.md, F2)."""
    assert not _backward_dependency(legacy=True)


def test_forward_propagation_is_causal():
    cfg = ForkNetConfig(**SMALL, legacy_backward_index=False)
    fuser = TempoFuser(8, cfg).eval()
    for p in fuser.res_backward.parameters():
        torch.nn.init.zeros_(p)  # silence the backward branch
    feats = torch.rand(1, 5, 8, 8, 8)
    flows = {f"{d}_{k}": torch.zeros(1, 5 - k, 2, 8, 8) for d in ("forward", "backward") for k in (1, 2)}
    with torch.no_grad():
        base = fuser(feats, flows)
        feats[:, 3] += 1.0
        changed = fuser(feats, flows)
    torch.testing.assert_close(base[:, :3], changed[:, :3])
    assert not torch.allclose(base[:, 3:], changed[:, 3:])
