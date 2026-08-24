# -*- coding: utf-8 -*-
"""P3 pandas-only 边界：FFT/矩阵/随机/constant 等 intentionally 保留 pandas 路径。"""
from __future__ import annotations

import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.operator_policy import INTENTIONALLY_PANDAS_ONLY
from factor_engine.cleaned_operators.registry import OperatorRegistry

# 企业级 intentionally pandas-only（与 operator_policy 单点定义同步）
def _load():
    ensure_cleaned_loaded()


@pytest.fixture(scope="module", autouse=True)
def _ensure_ops_loaded():
    _load()


def test_removed_legacy_regression_names_are_not_production_backends():
    from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS

    for name in ("regress", "ridge", "lasso", "slope", "aggr_top_n", "ACF", "pacf"):
        assert name not in DAILY_CANONICALS


def test_daily_has_no_pandas_only_evidence_gap():
    from factor_engine.backend.fastpath_evidence import polars_executed_parity_canonicals
    from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS

    assert DAILY_CANONICALS <= polars_executed_parity_canonicals()


def test_intentionally_pandas_only_still_have_pandas():
    for name in sorted(INTENTIONALLY_PANDAS_ONLY):
        if name not in OperatorRegistry.list_canonical():
            continue
        assert "pandas_numpy" in OperatorRegistry.backends_for(name), name
