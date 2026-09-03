# -*- coding: utf-8 -*-
"""R58 —— wheel package-data + importlib.resources 读取 config。

AlphaFlow review #7：``data_access/config/semantic_fields.yaml``、
``data_access/config/datasets.yaml``、``data_access/config/README.md`` 必须通过
``[tool.setuptools.package-data]`` 打进 wheel，并用 ``importlib.resources`` 读取
（不依赖 source path）。缺了这些文件，安装后的 wheel 无法 ``get_store()`` /
``get_semantic_catalog()``。

本文件用 mock 锁住「包内数据缺失 → 回退 source path」与「包内数据存在 → 直接
读取」两条路径，不触网、不真 build wheel。
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from data_access.registry.loader import load_registry
from data_access.read.semantic_catalog import SemanticFieldCatalog


def test_pyproject_declares_config_package_data():
    """pyproject.toml 必须声明 data_access/config/* 为 package-data（R58 #7）。"""
    root = Path(__file__).resolve().parent.parent.parent
    pyproject = root / "pyproject.toml"
    assert pyproject.exists(), "pyproject.toml 不存在"
    text = pyproject.read_text(encoding="utf-8")
    assert "[tool.setuptools.package-data]" in text
    assert '"data_access" = ["config/*.yaml", "config/README.md"]' in text


def test_config_files_exist_in_source():
    """config/ 三个运行时文件必须存在（wheel package-data 的来源）。"""
    root = Path(__file__).resolve().parent.parent.parent
    for name in ("datasets.yaml", "semantic_fields.yaml", "README.md"):
        assert (root / "config" / name).exists(), f"config/{name} 缺失"


def test_load_registry_via_importlib_resources(tmp_path):
    """包内数据存在 → importlib.resources 直接读取（不依赖 source path）。"""
    fake_yaml = textwrap.dedent(
        """
        ds:
          kind: static
          access_mode: staging
          layout: plain
          root: "/tmp/x"
          glob: "part-*.parquet"
          time_column: ts
          instrument_column: sym
          schema:
            ts: date
            sym: string
            val: double
        """
    )
    fake_files = type(
        "_FakeFiles",
        (),
        {
            "joinpath": lambda self, p: type(
                "_FakePath", (), {"read_bytes": lambda self: fake_yaml.encode()}
            )()
        },
    )()
    with patch("importlib.resources.files", return_value=fake_files):
        reg = load_registry(None)
    assert "ds" in reg._datasets


def test_load_registry_falls_back_to_source_path(tmp_path):
    """包内数据缺失 → 回退 source path（source checkout 兼容）。"""
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            ds:
              kind: static
              access_mode: staging
              layout: plain
              root: "/tmp/x"
              glob: "part-*.parquet"
              time_column: ts
              instrument_column: sym
              schema:
                ts: date
                sym: string
                val: double
            """
        ),
        encoding="utf-8",
    )
    with patch("importlib.resources.files", side_effect=FileNotFoundError):
        reg = load_registry(str(cfg))
    assert "ds" in reg._datasets


def test_semantic_catalog_via_importlib_resources():
    """包内数据存在 → importlib.resources 直接读取 semantic_fields.yaml。"""
    fake_yaml = textwrap.dedent(
        """
        close:
          dataset: ashare_stock_daily
          physical_name: Close
          market: ashare
          canonical_unit: price
        """
    )
    fake_files = type(
        "_FakeFiles",
        (),
        {
            "joinpath": lambda self, p: type(
                "_FakePath", (), {"read_bytes": lambda self: fake_yaml.encode()}
            )()
        },
    )()
    with patch("importlib.resources.files", return_value=fake_files):
        cat = SemanticFieldCatalog.from_yaml(None)
    assert "close" in cat._fields


def test_semantic_catalog_falls_back_to_source_path(tmp_path):
    """包内数据缺失 → 回退 source path。"""
    cfg = tmp_path / "semantic_fields.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            close:
              dataset: ashare_stock_daily
              physical_name: Close
              market: ashare
              canonical_unit: price
            """
        ),
        encoding="utf-8",
    )
    with patch("importlib.resources.files", side_effect=FileNotFoundError):
        cat = SemanticFieldCatalog.from_yaml(str(cfg))
    assert "close" in cat._fields
