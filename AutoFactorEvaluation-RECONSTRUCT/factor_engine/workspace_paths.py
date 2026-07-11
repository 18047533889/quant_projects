"""factor_engine 路径解析（monorepo 与各组员 home 通用）。"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """``factor_engine/`` 包根目录。"""
    return Path(__file__).resolve().parent


def quant_projects_root() -> Path:
    """``quant_projects/`` monorepo 根（``factor_engine`` 的上一级）。"""
    override = os.getenv("QUANT_PROJECTS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root().parent


def resolve_path(value: str | Path, *, base_dir: Path | None = None) -> Path:
    """展开 ``~`` / 环境变量；相对路径基于 ``base_dir`` 或 ``quant_projects_root()``。"""
    text = os.path.expandvars(str(value).strip())
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = base_dir if base_dir is not None else quant_projects_root()
    return (root / path).resolve()


def workspace_data_root() -> Path:
    override = os.getenv("QUANTSOCIETY_WORKSPACE_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    monorepo_data = quant_projects_root() / "data"
    if monorepo_data.is_dir():
        return monorepo_data.resolve()
    return (repo_root() / "workspace_data").resolve()


def default_factor_lake_root() -> Path:
    override = os.getenv("FACTOR_LAKE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return workspace_data_root() / "factors" / "lake"
