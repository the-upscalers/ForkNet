"""Numerical equivalence with the original research code.

Requires a checkout of the private repository https://github.com/the-upscalers/pixel-level-vsr (access needed) at branch ``fork-net``
(commit 70baa1f) and the ``bsconv`` package:

    git clone -b fork-net https://github.com/the-upscalers/pixel-level-vsr /tmp/pixel-level-vsr
    PIXEL_LEVEL_VSR_ROOT=/tmp/pixel-level-vsr pytest tests/test_legacy_equivalence.py

Skipped otherwise.
"""

import os
import sys
import types
from pathlib import Path

import pytest
import torch

from forknet import ForkNet, ForkNetConfig

LEGACY_ROOT = os.environ.get("PIXEL_LEVEL_VSR_ROOT")
pytestmark = pytest.mark.skipif(
    not LEGACY_ROOT or not (Path(LEGACY_ROOT) / "models").is_dir(),
    reason="set PIXEL_LEVEL_VSR_ROOT to a checkout of the (private) original repository",
)
LEGACY = dict(legacy_flow_direction=True, legacy_backward_index=True, legacy_flow_scaling=True)


@pytest.fixture(scope="module")
def legacy():
    pytest.importorskip("bsconv")
    sys.path.insert(0, LEGACY_ROOT)
    from models.optical_flow_model.spynet import SpyNet

    def load_script(name, subs=()):
        path = Path(LEGACY_ROOT) / name
        src = path.read_text()
        for a, b in subs:
            assert a in src
            src = src.replace(a, b)
        mod = types.ModuleType(name)
        exec(compile(src, str(path), "exec"), mod.__dict__)
        return mod

    yield SpyNet, load_script
    sys.path.remove(LEGACY_ROOT)


def _compare(ref, cfg):
    new = ForkNet(cfg)
    new.load_state_dict(ref.state_dict(), strict=True)
    ref.eval(), new.eval()
    x = torch.rand(1, 6, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(new(x), ref(x), atol=1e-5, rtol=1e-5)


def test_final_commit_modular_model(legacy):
    SpyNet, _ = legacy
    from models.denoising_fork_net.fork_net_module import ForkNetModule
    from utils.options import Options

    opts = Options()
    opts.num_grid_blocks, opts.debug = 5, False
    cfg = ForkNetConfig(prop_block="resblock", bsconv_fusion=False, fuse_with_input=True, bottleneck_mult=1, **LEGACY)
    _compare(ForkNetModule(SpyNet(None), opts), cfg)


@pytest.mark.parametrize("bsconv", [True, False])
def test_first_order_script(legacy, bsconv):
    SpyNet, load_script = legacy
    mod = load_script("first_order.py", [] if bsconv else [("bs = True", "bs = False")])
    opts = mod.Options()
    opts.num_grid_blocks = 5
    _compare(mod.ForkNetModule(SpyNet(None), opts), ForkNetConfig(prop_order=1, use_bsconv=bsconv, **LEGACY))


@pytest.mark.parametrize("level,half", [("tempo_fuser_32", False), ("tempo_fuser_64", True)])
def test_second_order_convnext_tempo_fuser(legacy, level, half):
    _, load_script = legacy
    mod = load_script("train_vim.py")
    ref = mod.TempoFuser(mid_filters=128, num_grid_blocks=5, half_filters=half).eval()
    new = getattr(ForkNet(ForkNetConfig(**LEGACY)).u_up, level).eval()
    new.load_state_dict(ref.state_dict(), strict=True)
    feats = torch.rand(1, 6, 128, 8, 12)
    flows = {f"{d}_{k}": torch.randn(1, 6 - k, 2, 8, 12) * 2 for d in ("forward", "backward") for k in (1, 2)}
    with torch.no_grad():
        torch.testing.assert_close(new(feats, flows), ref(flows, feats))
