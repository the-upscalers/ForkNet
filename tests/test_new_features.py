"""Ablation switches, degradations, LP text utilities and the trainer's time budget."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import smooth_texture
from forknet import ForkNet, ForkNetConfig
from forknet.data import imresize
from forknet.data.degradations import RealVideoDegradation, _ffmpeg_exe, degrade_clip, video_compress
from forknet.lp.text import char_accuracy, consensus, levenshtein, normalize
from forknet.utils import infer_config

SMALL = dict(num_feat=8, num_encoder_blocks=1, num_prop_blocks=2)
VARIANTS = {
    "no_propagation": {"prop_order": 0},
    "forward_only": {"prop_direction": "forward"},
    "forward_only_first_order": {"prop_direction": "forward", "prop_order": 1},
    "no_flow": {"use_flow": False},
    "no_lr_injection": {"lr_injection": False},
}


@pytest.mark.parametrize("name", list(VARIANTS))
def test_ablation_variants_run_and_roundtrip(name):
    cfg = ForkNetConfig(**SMALL, **VARIANTS[name])
    model = ForkNet(cfg).eval()
    with torch.no_grad():
        assert model(torch.rand(1, 4, 3, 64, 64)).shape == (1, 4, 3, 256, 256)
    inferred = infer_config(model.state_dict())
    for key in ("prop_order", "prop_direction", "lr_injection", "num_encoder_blocks"):
        assert getattr(inferred, key) == getattr(cfg, key), key


def test_no_propagation_is_framewise():
    model = ForkNet(ForkNetConfig(**SMALL, prop_order=0)).eval()
    x = torch.rand(1, 4, 3, 64, 64)
    with torch.no_grad():
        full = model(x)
        single = model(x[:, 2:3])
    torch.testing.assert_close(full[:, 2:3], single, atol=1e-5, rtol=1e-5)


def test_forward_only_is_causal():
    model = ForkNet(ForkNetConfig(**SMALL, prop_direction="forward", use_flow=False)).eval()
    x = torch.rand(1, 5, 3, 64, 64)
    with torch.no_grad():
        a = model(x)
        x[:, 4] = torch.rand_like(x[:, 4])
        b = model(x)
    torch.testing.assert_close(a[:, :4], b[:, :4])


def test_degrade_clip_clean_is_matlab_bicubic():
    hr = [smooth_texture(64, 96, seed=s) for s in range(2)]
    lr = degrade_clip(hr)
    for h, l in zip(hr, lr):
        np.testing.assert_array_equal(l, imresize(h, 0.25))


def test_degrade_clip_cv2_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    hr = [smooth_texture(64, 64)]
    lr = degrade_clip(hr, downsample="cv2_bicubic")[0]
    ref = cv2.resize(hr[0], (16, 16), interpolation=cv2.INTER_CUBIC)
    assert np.abs(lr.astype(int) - ref.astype(int)).max() <= 1


def test_degrade_clip_levels_are_deterministic_and_harmful():
    hr = [smooth_texture(64, 64, seed=s) for s in range(3)]
    clean = degrade_clip(hr)
    for kwargs in ({"blur_sigma": 2.0}, {"noise_sigma": 10}, {"jpeg_quality": 30}):
        a, b = degrade_clip(hr, seed=1, **kwargs), degrade_clip(hr, seed=1, **kwargs)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
        assert all(x.shape == (16, 16, 3) for x in a)
        assert not all(np.array_equal(x, y) for x, y in zip(a, clean))


@pytest.mark.skipif(_ffmpeg_exe() is None, reason="ffmpeg not available")
def test_video_compression_roundtrip():
    frames = [smooth_texture(32, 48, seed=s) for s in range(4)]
    err = {}
    for crf in (18, 45):
        out = video_compress(frames, crf=crf)
        assert len(out) == 4 and out[0].shape == (32, 48, 3)
        err[crf] = np.mean([np.abs(o.astype(int) - f.astype(int)).mean() for o, f in zip(out, frames)])
    assert err[18] < err[45] and err[18] < 20  # 4:2:0 chroma subsampling limits fidelity on saturated textures


@pytest.mark.parametrize("video_prob", [0.0, 1.0])
def test_real_degradation_shapes(video_prob):
    if video_prob and _ffmpeg_exe() is None:
        pytest.skip("ffmpeg not available")
    deg = RealVideoDegradation({"video_prob": video_prob})
    hr = [smooth_texture(128, 128, seed=s) for s in range(3)]
    lr = deg(hr, np.random.default_rng(0))
    assert len(lr) == 3 and all(f.shape == (32, 32, 3) and f.dtype == np.uint8 for f in lr)


def test_degraded_training_dataset(make_video_dataset):
    from forknet.data import VideoTrainDataset

    _, hr = make_video_dataset(num_clips=2, num_frames=4)
    ds = VideoTrainDataset(None, hr, num_frames=3, patch_size=32, degradation={"video_prob": 0.0})
    x, y = ds[0]
    assert x.shape == (3, 3, 32, 32) and y.shape == (3, 3, 128, 128)


def test_lp_text_utilities():
    assert normalize(" ab-12 c_") == "AB12C"
    assert levenshtein("KITTEN", "SITTING") == 3
    assert char_accuracy("AB12C", "AB12C") == 1.0
    assert char_accuracy("AB1XC", "AB12C") == pytest.approx(0.8)
    text, support, agree = consensus([("AB12C", 0.9), ("AB12C", 0.8), ("A812C", 0.4), ("AB1", 0.9)])
    assert text == "AB12C" and support == 3 and agree == pytest.approx(2 / 3)
    assert consensus([]) == ("", 0, 0.0)


def test_trainer_time_budget(make_video_dataset, tmp_path):
    from forknet.trainer import Trainer
    from forknet.utils import load_config

    lr, hr = make_video_dataset(num_clips=2, num_frames=4)
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/train_reds.yaml", [
        f"output_dir={tmp_path}", "name=budget", "spynet_path=null", f"model={json.dumps(SMALL)}",
        f"data.train.lr_root={lr}", f"data.train.hr_root={hr}", "data.train.exclude=null", "data.train.num_frames=2",
        "data.train.patch_size=32", "data.val=null", "train.total_iters=1000", "train.batch_size=1",
        "train.num_workers=0", "train.max_hours=0.0000001", "train.val_every=0",
    ])
    it = Trainer(cfg).fit()
    assert it < 1000
    assert (tmp_path / "budget" / "checkpoints" / "latest.pth").exists()
    assert not (tmp_path / "budget" / "checkpoints" / "final.pth").exists()


def test_grad_checkpointing_gives_identical_gradients():
    torch.manual_seed(0)
    model = ForkNet(ForkNetConfig(**SMALL, legacy_flow_direction=False)).train()
    x = torch.rand(1, 3, 3, 64, 64)
    grads = []
    for enabled in (False, True):
        model.zero_grad()
        model.set_grad_checkpointing(enabled)
        model(x).mean().backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    for n in grads[0]:
        torch.testing.assert_close(grads[0][n], grads[1][n], atol=1e-6, rtol=1e-5)


def test_kernel_mix_shapes_and_kernel_choice():
    from forknet.data.degradations import KernelMixDegradation, build_degradation

    hr = [np.random.default_rng(i).integers(0, 255, (32, 48, 3), dtype=np.uint8) for i in range(3)]
    deg = build_degradation({"type": "kernel_mix", "video_prob": 0.0, "kernels": ["cv2_bicubic"]})
    assert isinstance(deg, KernelMixDegradation)
    lr = deg(hr, np.random.default_rng(0))
    assert len(lr) == 3 and lr[0].shape == (8, 12, 3) and lr[0].dtype == np.uint8
    import cv2

    ref = np.clip(np.round(cv2.resize(hr[0].astype(np.float32), (12, 8), interpolation=cv2.INTER_CUBIC)), 0, 255)
    assert np.abs(lr[0].astype(int) - ref.astype(int)).max() <= 1
    with pytest.raises(ValueError):
        build_degradation({"type": "nope"})


def test_mixed_degradation_uses_both_pipelines():
    from forknet.data.degradations import MixedDegradation, build_degradation

    hr = [np.random.default_rng(i).integers(0, 255, (64, 64, 3), dtype=np.uint8) for i in range(3)]
    deg = build_degradation({"type": "mixed", "real_prob": 0.5, "real": {"video_prob": 0.0},
                             "kernel_mix": {"video_prob": 0.0}})
    assert isinstance(deg, MixedDegradation)
    picks = set()
    for seed in range(20):
        rng = np.random.default_rng(seed)
        picks.add(rng.random() < 0.5)
        lr = deg(hr, np.random.default_rng(seed))
        assert len(lr) == 3 and lr[0].shape == (16, 16, 3) and lr[0].dtype == np.uint8
    assert picks == {True, False}


def test_plate_masks_follow_crop_and_flips(tmp_path):
    import json

    from forknet.data import VideoTrainDataset, write_image

    for c in ("a",):
        for i in range(4):
            hr = np.zeros((32, 32, 3), np.uint8)
            hr[4:12, 8:24] = 255  # the "plate" is the only bright region
            write_image(tmp_path / "hr" / c / f"{i:08d}.png", hr)
            write_image(tmp_path / "lr" / c / f"{i:08d}.png", hr[::4, ::4])
    boxes = tmp_path / "boxes.json"
    boxes.write_text(json.dumps({"a": {f"{i:08d}": [8, 4, 24, 12] for i in range(4)}}))
    ds = VideoTrainDataset(tmp_path / "lr", tmp_path / "hr", num_frames=2, patch_size=None, plate_boxes=boxes)
    for _ in range(8):
        lr, hr, mask = ds[0]
        assert mask.shape == (2, 1, 32, 32) and hr.shape == (2, 3, 32, 32)
        assert torch.equal((hr[:, :1] > 0.5), (mask > 0.5))  # the mask stays on the plate after flips


def test_aux_gan_loss_trains_discriminator():
    from forknet.aux_losses import AuxLosses

    aux = AuxLosses({"gan_weight": 0.05, "aux_frames": 2}, torch.device("cpu"))
    assert aux.enabled and aux.disc is not None
    sr = torch.rand(1, 3, 3, 32, 32, requires_grad=True)
    hr = torch.rand(1, 3, 3, 32, 32)
    loss, logs = aux.generator_loss(sr, hr)
    loss.backward()
    assert sr.grad is not None and "gan_g" in logs
    logs.update(aux.discriminator_step(sr, hr))
    assert "gan_d" in logs and set(aux.state_dict()) == {"disc", "d_opt"}
