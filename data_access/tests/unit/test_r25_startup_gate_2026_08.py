# -*- coding: utf-8 -*-
"""R25 §111 —— Production Startup Gate 测试。

    startup gate：production 下任一 critical fail → 启动失败；research 只收集。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_startup_gate_production_blocking():
    from data_access.core.exceptions import DataAccessError
    from data_access.runtime.startup_gate import run_startup_gate

    store = MagicMock()

    def _blocking(store=None):
        return ["critical: mirror layout != COS contract (R25 §23)"]

    with pytest.raises(RuntimeError, match="startup gate failed"):
        run_startup_gate(store, production=True, checks=[_blocking])


def test_startup_gate_production_clean():
    from data_access.runtime.startup_gate import run_startup_gate

    store = MagicMock()

    def _clean(store=None):
        return []

    result = run_startup_gate(store, production=True, checks=[_clean])
    assert result.passed is True
    assert list(result.problems) == []


def test_startup_gate_research_nonblocking():
    from data_access.runtime.startup_gate import run_startup_gate

    store = MagicMock()

    def _problem(store=None):
        return ["research note: cache permissions (non-blocking)"]

    result = run_startup_gate(store, production=False, checks=[_problem])
    assert result.passed is True
    assert len(result.problems) == 1


def test_startup_gate_legacy_home_fallback(monkeypatch):
    """§53：legacy /home/shw fallback 检测。"""
    from data_access.runtime.startup_gate import _legacy_home_fallback_in_use

    used = _legacy_home_fallback_in_use()
    # 开发机默认根在 /home/shw → 会报告（production 下 blocking）
    assert isinstance(used, list)
