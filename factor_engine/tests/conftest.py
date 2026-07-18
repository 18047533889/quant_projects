"""pytest 全局：确保 factor_engine 与 quant_projects 根目录在 import 路径中。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_FE_ROOT = Path(__file__).resolve().parents[1]
_QUANT_ROOT = _FE_ROOT.parent

# ``data_access`` is installed as an editable package in the project venv.
# Do not add its source directory directly: that shadows the factor_engine
# ``tests`` package when the monorepo is collected from its root.
for _path in (str(_FE_ROOT), str(_QUANT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


@pytest.fixture(autouse=True)
def _reset_production_env_leaks():
    """清理 FactorEngine._sync_production_env 在进程内遗留的生产环境变量。"""
    yield
    os.environ.pop("QUANT_PRODUCTION_MODE", None)
    if os.environ.get("FACTOR_ENGINE_RUN_MODE") == "production":
        os.environ.pop("FACTOR_ENGINE_RUN_MODE", None)
