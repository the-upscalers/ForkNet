from pathlib import Path

import pytest
import torch

from forknet import ForkNet, ForkNetConfig
from forknet.utils import build_model_from_checkpoint, infer_config, load_checkpoint, load_config, save_checkpoint
from test_forknet import GOLDEN_PARAMS

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", list(GOLDEN_PARAMS))
def test_infer_config_from_original_format(name, tmp_path):
    kwargs, _ = GOLDEN_PARAMS[name]
    cfg = ForkNetConfig(**kwargs, legacy_flow_direction=True, legacy_backward_index=True, legacy_flow_scaling=True)
    model = ForkNet(cfg)
    assert infer_config(model.state_dict()) == cfg
    # checkpoint as written by the original train/train.py, including a DataParallel prefix
    path = tmp_path / "best_checkpoint.pth"
    torch.save({"epoch": 3, "psnr": 29.0, "model_state_dict": {f"module.{k}": v for k, v in model.state_dict().items()}}, path)
    loaded, meta = build_model_from_checkpoint(path)
    assert loaded.cfg == cfg and meta["format"] == "original" and meta["epoch"] == 3
    for (k, a), b in zip(model.state_dict().items(), loaded.state_dict().values()):
        assert torch.equal(a, b), k


def test_release_roundtrip_and_overrides(tmp_path):
    cfg = ForkNetConfig(num_feat=8, num_encoder_blocks=1, num_prop_blocks=2, legacy_flow_direction=False)
    model = ForkNet(cfg)
    save_checkpoint(tmp_path / "m.pth", model, cfg, iter=10)
    sd, loaded_cfg, meta = load_checkpoint(tmp_path / "m.pth")
    assert loaded_cfg == cfg and meta == {"iter": 10, "format": "release"}
    m2, _ = build_model_from_checkpoint(tmp_path / "m.pth", {"legacy_backward_index": False})
    assert m2.cfg.legacy_backward_index is False
    with pytest.raises(ValueError):
        build_model_from_checkpoint(tmp_path / "m.pth", {"num_feat": 16})


def test_config_inheritance_and_overrides(tmp_path):
    (tmp_path / "a.yaml").write_text("x: 1\nnested: {a: 1, b: 2}\n")
    (tmp_path / "b.yaml").write_text("base: a.yaml\nnested: {b: 3}\n")
    cfg = load_config(tmp_path / "b.yaml", ["nested.c=[1, 2]", "x=0.5"])
    assert cfg == {"x": 0.5, "nested": {"a": 1, "b": 3, "c": [1, 2]}}


TRAIN_CONFIGS = sorted((ROOT / "configs").glob("train*.yaml")) + sorted((ROOT / "configs").glob("ablation/*.yaml")) + sorted((ROOT / "configs").glob("tradeoff/*.yaml")) + sorted((ROOT / "configs").glob("pilot/*.yaml")) + sorted((ROOT / "configs").glob("finetune_*_from_paper.yaml"))


@pytest.mark.parametrize("path", TRAIN_CONFIGS + [ROOT / "configs/finetune_lp.yaml"], ids=lambda p: p.stem)
def test_shipped_training_configs_are_valid(path):
    cfg = load_config(path)
    ForkNetConfig.from_dict(cfg["model"])
    assert {"train", "val"} <= set(cfg["data"]) and cfg["train"]["total_iters"] > 0


def test_telemetry_is_a_noop_without_wandb_key(tmp_path, monkeypatch):
    from forknet.utils.telemetry import Telemetry

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    t = Telemetry({"name": "x"}, tmp_path)
    assert t.run is None
    t.log({"a": 1.0}, step=1)
    t.finish()
    assert not list(tmp_path.iterdir())


def test_telemetry_failure_never_raises(tmp_path, monkeypatch):
    """A broken wandb (API change, no network) must disable telemetry, not stop training."""
    import sys
    import types

    from forknet.utils.telemetry import Telemetry

    fake = types.ModuleType("wandb")

    def init(**kwargs):
        raise RuntimeError("wandb unavailable")

    fake.init = init
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    with pytest.warns(UserWarning, match="telemetry disabled"):
        t = Telemetry({"name": "x"}, tmp_path)
    assert t.run is None
    t.log({"a": 1.0}, step=1)
    t.finish()
