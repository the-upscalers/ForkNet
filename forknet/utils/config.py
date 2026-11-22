"""YAML configs with ``base:`` inheritance and ``key.sub=value`` command-line overrides."""

import copy
from pathlib import Path

import yaml


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path, overrides=()):
    """Loads ``path``; a top-level ``base`` key (relative path or list of paths) is merged first."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    bases = cfg.pop("base", [])
    merged = {}
    for b in [bases] if isinstance(bases, str) else bases:
        merged = deep_merge(merged, load_config(path.parent / b))
    cfg = deep_merge(merged, cfg)
    for item in overrides:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        value = yaml.safe_load(value)
        if isinstance(value, dict) and isinstance(node.get(leaf), dict):
            value = deep_merge(node[leaf], value)
        node[leaf] = value
    return cfg


def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
