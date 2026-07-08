# -*- coding: utf-8 -*-
"""运行时环境引导：加载 monorepo ``.env``，不覆盖已有环境变量。"""
from __future__ import annotations

from workspace_paths import load_env_file


def bootstrap_runtime_env() -> None:
    """进程级一次性加载 ``quant_projects/.env``。"""
    load_env_file()
