# -*- coding: utf-8 -*-
"""R27 大规模批量因子极速落值 / 资源调度测试 session。"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def r27_low_memory() -> None:
    """小内存服务器：压住并发，避免测试 OOM（用户约束）。"""
    os.environ.setdefault("FACTOR_ENGINE_CPU_BUDGET", "4")
    os.environ.setdefault("FACTOR_ENGINE_MAX_MEMORY_BYTES", str(8 * 1024**3))


@pytest.fixture(autouse=True)
def r27_reset_calibration():
    from runtime.runtime_calibration import reset_calibration

    reset_calibration()
    yield
    reset_calibration()
