"""registry：生产 datasets.yaml 与 YAML 锚点模板键。"""
from __future__ import annotations

from pathlib import Path

import yaml

from data_access.cos_mirror import DATASET_MIRROR_REGISTRY
from data_access.registry import load_registry


def test_yaml_anchor_keys_are_not_registered_datasets():
    config_path = Path(__file__).resolve().parents[2] / "config" / "datasets.yaml"
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    anchor_keys = [k for k in raw if str(k).startswith("_")]
    assert anchor_keys, "datasets.yaml 应含 _*_defaults 锚点模板"

    reg = load_registry(config_path)
    for key in anchor_keys:
        assert key not in reg._datasets


def test_mirror_registry_datasets_are_registered():
    config_path = Path(__file__).resolve().parents[2] / "config" / "datasets.yaml"
    reg = load_registry(config_path)
    missing = sorted(set(DATASET_MIRROR_REGISTRY) - set(reg._datasets))
    assert missing == [], f"cos_mirror 已配置但未在 datasets.yaml 登记: {missing}"
