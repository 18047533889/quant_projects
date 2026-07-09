"""factor_engine 路径解析（monorepo 与各组员 home 通用）。"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """``factor_engine/`` 包根目录。"""
    return Path(__file__).resolve().parents[1]


def quant_projects_root() -> Path:
    """``quant_projects/`` monorepo 根（``factor_engine`` 的上一级）。"""
    override = os.getenv("QUANT_PROJECTS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root().parent


_ENV_LOADED = False


def load_env_file(path: Path | None = None, *, override: bool = False) -> None:
    """从 ``.env`` 加载 ``KEY=VALUE`` 到 ``os.environ``（进程内只执行一次）。"""
    global _ENV_LOADED
    if _ENV_LOADED and not override:
        return
    env_path = path or (quant_projects_root() / ".env")
    if not env_path.is_file():
        _ENV_LOADED = True
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = val
    _ENV_LOADED = True


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
