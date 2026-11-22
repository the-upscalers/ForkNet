"""Train for a few iterations on synthetic data, then run every CLI script on the result."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from forknet.trainer import Trainer
from forknet.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
SMALL_MODEL = {"num_feat": 8, "num_encoder_blocks": 1, "num_prop_blocks": 2}


def run(*args):
    proc = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout + proc.stderr


@pytest.fixture
def trained(make_video_dataset, tmp_path):
    lr, hr = make_video_dataset(num_clips=3, num_frames=6)
    overrides = [
        f"output_dir={tmp_path / 'exp'}", "name=smoke", "spynet_path=null",
        f"model={json.dumps(SMALL_MODEL)}",
        f"data.train.lr_root={lr}", f"data.train.hr_root={hr}", "data.train.exclude=['002']",
        "data.train.num_frames=3", "data.train.patch_size=32",
        f"data.val={{lr_root: {lr}, hr_root: {hr}, clips: ['002']}}",
        "train.total_iters=4", "train.batch_size=2", "train.num_workers=0", "train.fix_flow_iters=2",
        "train.log_every=2", "train.val_every=4", "train.save_every=2", "val.chunk_size=3",
    ]
    cfg = load_config(ROOT / "configs/train_reds.yaml", overrides)
    trainer = Trainer(cfg)
    assert trainer.fit() == 4
    exp = tmp_path / "exp" / "smoke"
    return exp, lr, hr, cfg, overrides


def test_training_writes_logs_and_checkpoints(trained):
    exp, *_ = trained
    for f in ("config.yaml", "train.log", "train_log.jsonl", "val_log.jsonl", "checkpoints/latest.pth", "checkpoints/final.pth", "checkpoints/best.pth"):
        assert (exp / f).exists(), f
    logs = [json.loads(line) for line in (exp / "train_log.jsonl").read_text().splitlines()]
    assert [r["iter"] for r in logs] == [2, 4] and logs[0]["flow_trainable"] is False and logs[1]["flow_trainable"] is True


def test_resume_continues_iterations(trained):
    exp, _, _, cfg, _ = trained
    cfg = dict(cfg, train={**cfg["train"], "total_iters": 6})
    trainer = Trainer(cfg, resume=exp / "checkpoints" / "latest.pth")
    assert trainer.start_iter == 4
    assert trainer.fit() == 6


def test_cli_scripts(trained, tmp_path):
    exp, lr, hr, _, _ = trained
    ckpt = exp / "checkpoints" / "final.pth"
    test_cfg = tmp_path / "test.yaml"
    test_cfg.write_text(yaml.safe_dump({
        "name": "synthetic",
        "data": {"test": {"lr_root": str(lr), "hr_root": str(hr), "clips": ["002"]}},
        "test": {"chunk_size": 3, "overlap": 1},
    }))
    out = tmp_path / "res"
    run("scripts/test.py", "--config", test_cfg, "--checkpoint", ckpt, "--out", out, "--save-images")
    report = json.loads((out / "metrics.json").read_text())
    assert report["standard"]["clips"]["002"]["num_frames"] == 6
    assert len(list((out / "images" / "002").glob("*.png"))) == 6

    run("scripts/test.py", "--config", test_cfg, "--bicubic", "--out", tmp_path / "bic")
    run("scripts/evaluate_outputs.py", "--sr", out / "images", "--gt", hr, "--clips", "002", "--name", "smoke", "--out", tmp_path / "eval.json")
    again = json.loads((tmp_path / "eval.json").read_text())["standard"]["average"]
    assert again["psnr_rgb"] == pytest.approx(report["standard"]["average"]["psnr_rgb"], abs=1e-9)

    printed = run("scripts/inspect_checkpoint.py", ckpt, "--export", tmp_path / "release.pth")
    assert "format      : release" in printed
    run("scripts/infer.py", "--checkpoint", tmp_path / "release.pth", "--input", lr / "000", "--out", tmp_path / "infer", "--chunk-size", "3")
    assert len(list((tmp_path / "infer" / "000").glob("*.png"))) == 6
    run("scripts/make_tables.py", out / "metrics.json", tmp_path / "eval.json", "--out", tmp_path / "tables.md")
    assert "PSNR-RGB" in (tmp_path / "tables.md").read_text()


def test_complexity_script(tmp_path):
    out = run("scripts/complexity.py", "--lr-size", "64", "64", "--num-frames", "3", "--out", tmp_path / "c.json")
    rows = json.loads((tmp_path / "c.json").read_text())
    assert rows["Fork-Net (2nd order + BSConv + ConvNeXt)"]["params_total"] == 4_373_039
    assert "MACs/frame" in out


def test_prepare_lp_dataset(tmp_path, make_video_dataset):
    _, hr = make_video_dataset(num_clips=5, num_frames=5)
    run("scripts/prepare_lp_dataset.py", "--src", hr, "--dst", tmp_path / "lp", "--test-ratio", "0.4", "--seed", "1")
    split = json.loads((tmp_path / "lp" / "splits.json").read_text())
    assert len(split["test"]) == 2 and len(split["train"]) == 3 and not set(split["test"]) & set(split["train"])
    assert len(list((tmp_path / "lp" / "LR" / "000").glob("*.png"))) == 5
