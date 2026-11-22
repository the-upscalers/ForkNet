import json

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import smooth_texture
from forknet.data import VideoTestDataset, VideoTrainDataset, imresize, resolve_clips


def test_imresize_constant_and_shape():
    img = np.full((40, 52, 3), 77, np.uint8)
    out = imresize(img, 1 / 4)
    assert out.shape == (10, 13, 3) and (out == 77).all()


def test_imresize_close_to_pil_bicubic_in_interior():
    img = smooth_texture(128, 128)
    ours = imresize(img, 1 / 4).astype(np.float64)
    pil = np.asarray(Image.fromarray(img).resize((32, 32), Image.BICUBIC)).astype(np.float64)
    assert np.abs(ours - pil)[2:-2, 2:-2].mean() < 1.0


def test_imresize_upscale_preserves_linear_ramp():
    ramp = np.tile(np.arange(16, dtype=np.float64), (16, 1))[..., None]
    up = imresize(ramp, 2)
    np.testing.assert_allclose(np.diff(up[8, 4:-4, 0]), 0.5, atol=1e-9)


def test_train_dataset_fixed_chunks(make_video_dataset):
    lr, hr = make_video_dataset(num_clips=3, num_frames=8)
    ds = VideoTrainDataset(lr, hr, num_frames=3, patch_size=32, temporal_sampling="fixed_chunks", exclude=["001"])
    assert ds.clips == ["000", "002"]
    assert len(ds) == 2 * (8 // 3)
    x, y = ds[0]
    assert x.shape == (3, 3, 32, 32) and y.shape == (3, 3, 128, 128)
    assert 0 <= x.min() and x.max() <= 1


def test_train_crop_alignment(make_video_dataset):
    """HR patch must be exactly the region corresponding to the LR patch."""
    from forknet.data import read_image

    lr, hr = make_video_dataset(num_clips=1, num_frames=4)
    ds = VideoTrainDataset(lr, hr, num_frames=2, patch_size=16, augment="none")
    x, y = ds[0]
    down = imresize((y[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8), 1 / 4)
    full_lr = read_image(sorted((lr / "000").iterdir())[0])
    # locate the LR patch in the full LR frame, then compare with downscaled HR patch interior
    patch = (x[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    assert any(
        np.array_equal(full_lr[i : i + 16, j : j + 16], patch) for i in range(49) for j in range(49)
    )
    assert np.abs(down.astype(int) - patch.astype(int))[2:-2, 2:-2].mean() < 2.0


def test_legacy_augment_swaps_channels(make_video_dataset, monkeypatch):
    lr, hr = make_video_dataset(num_clips=1, num_frames=4)
    plain = VideoTrainDataset(lr, hr, num_frames=2, patch_size=None, augment="none")
    legacy = VideoTrainDataset(lr, hr, num_frames=2, patch_size=None, augment="legacy")
    draws = iter([0.9, 0.1])  # skip vertical flip, take the second ("horizontal") flip
    monkeypatch.setattr("forknet.data.video_dataset.random.random", lambda: next(draws))
    x_legacy, _ = legacy[0]
    x_plain, _ = plain[0]
    torch.testing.assert_close(x_legacy, x_plain.flip(1))  # RGB -> BGR, not a spatial flip


def test_test_dataset_and_split_file(make_video_dataset, tmp_path):
    lr, hr = make_video_dataset(num_clips=3, num_frames=5)
    split = tmp_path / "splits.json"
    split.write_text(json.dumps({"train": ["000", "001"], "test": ["002"]}))
    ds = VideoTestDataset(lr, hr, split_file=split, split="test")
    item = ds[0]
    assert item["clip"] == "002" and item["lr"].shape == (5, 3, 64, 64) and item["hr"].shape == (5, 3, 256, 256)
    assert VideoTestDataset(lr, None, clips=["000"])[0]["hr"].numel() == 0
    with pytest.raises(ValueError):
        resolve_clips(lr, clips=["000"], exclude=["000"])
