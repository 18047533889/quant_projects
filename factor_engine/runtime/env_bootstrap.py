# -*- coding: utf-8 -*-
"""运行时环境引导：加载 monorepo ``.env``，不覆盖已有环境变量。

在 ``FactorEngine`` 构造时自动调用，确保 ``QUANT_*`` / ``FACTOR_ENGINE_*``
等配置在首次执行前可用。已存在于 ``os.environ`` 的键不会被覆盖。
"""
from __future__ import annotations

from workspace_paths import load_env_file


def bootstrap_runtime_env() -> None:
    """进程级一次性加载 ``quant_projects/.env``。

    由 ``FactorEngine.__init__`` 调用；多次调用安全（``load_env_file`` 幂等）。
    """
    load_env_file()

