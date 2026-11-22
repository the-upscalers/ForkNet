import pytest
import torch

from forknet.archs.blocks import BSConvU, ConvNeXtBlock, ResidualBlock, UNetBlock, make_unet_layer


def test_bsconv_structure_and_params():
    m = BSConvU(8, 16, 3)
    shapes = {n: tuple(p.shape) for n, p in m.named_parameters()}
    assert shapes == {"pw.weight": (16, 8, 1, 1), "dw.weight": (16, 1, 3, 3), "dw.bias": (16,)}
    assert m(torch.rand(2, 8, 10, 12)).shape == (2, 16, 10, 12)


def test_bsconv_matches_reference_package():
    ref_mod = pytest.importorskip("bsconv.pytorch")
    ref = ref_mod.BSConvU(8, 16, 3, 1, padding=1, dilation=1)
    ours = BSConvU(8, 16, 3)
    ours.load_state_dict(ref.state_dict(), strict=True)
    x = torch.rand(2, 8, 9, 11)
    torch.testing.assert_close(ours(x), ref(x))


@pytest.mark.parametrize("block", [ConvNeXtBlock, ResidualBlock])
def test_prop_blocks_are_residual_and_shape_preserving(block):
    m = block(16)
    for p in m.parameters():
        torch.nn.init.zeros_(p)
    x = torch.rand(1, 16, 8, 8)
    torch.testing.assert_close(m(x), x)  # zero weights -> identity


def test_unet_block_residual_only_when_channels_match():
    assert UNetBlock(8, 8).residual
    assert not UNetBlock(3, 8).residual
    layer = make_unet_layer(3, 8, 3)
    assert len(layer) == 3 and layer(torch.rand(1, 3, 8, 8)).shape == (1, 8, 8, 8)


def test_rep_dw_fuses_exactly_and_restores_plain_shapes():
    from forknet import ForkNet, ForkNetConfig

    torch.manual_seed(0)
    cfg = dict(num_feat=8, num_encoder_blocks=1, num_prop_blocks=1)
    rep = ForkNet(ForkNetConfig(rep_dw=True, **cfg)).eval()
    for m in rep.modules():  # non-trivial training branches
        if m.__class__.__name__ == "RepDWConv":
            torch.nn.init.normal_(m.conv1.weight, std=0.1)
    x = torch.rand(1, 3, 3, 16, 16)
    with torch.no_grad():
        y = rep(x)
        rep.reparameterize()
        torch.testing.assert_close(rep(x), y, rtol=1e-4, atol=1e-4)
    plain = ForkNet(ForkNetConfig(**cfg))
    assert rep.cfg.rep_dw is False
    assert {k: v.shape for k, v in rep.state_dict().items()} == {k: v.shape for k, v in plain.state_dict().items()}
