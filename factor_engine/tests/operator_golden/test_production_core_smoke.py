# -*- coding: utf-8
"""PRODUCTION_CORE 算子 golden panel smoke：shape 保持 + 可运行。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.operator_spec import PRODUCTION_CORE_CANONICALS, build_operator_spec
from cleaned_operators.registry import OperatorRegistry
from tests.operator_golden.conftest import assert_panel_shape_unchanged

# 需多输入 / 特殊参数，smoke 中跳过
_SKIP_SMOKE: frozenset[str] = frozenset(
    {
        "RSI_WILDER",
        "ATR_WILDER",
        "ts_corr",
        "ts_beta",
        "rolling_beta",
        "cs_resid",
        "cs_regression",
        "group_rank",
        "group_mean",
        "group_zscore",
        "group_neutralize",
        # Two-input group aggregators (x, group): unary smoke cannot exercise
        # them; they are covered by the group parity/golden suites instead.
        "group_count",
        "group_max",
        "group_min",
        "group_std",
        "group_sum",
        "group_normalize",
        "group_winsorize",
        "where",
        "coalesce",
        "protected_div",
        "safe_div_null",
        "period_average",
        "period_change",
        "period_cagr",
        "quarter_from_cumulative",
        "ttm_from_quarterly",
        "ttm_from_cumulative",
        "yoy_by_period",
    }
)

_BINARY_TWO_PANEL: frozenset[str] = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "gt",
        "lt",
        "ge",
        "le",
        "eq",
        "ne",
        "and_",
        "or_",
        "maximum",
        "minimum",
        "ts_cov",
        "vwap",
    }
)

_UNARY_SMOKE: dict[str, dict] = {
    "ts_mean": {"d": 2},
    "ts_sum": {"d": 2},
    "ts_min": {"d": 2},
    "ts_max": {"d": 2},
    "ts_std": {"d": 2},
    "ts_var": {"d": 2},
    "ts_median": {"d": 2},
    "ts_zscore": {"d": 2},
    "ts_delay": {"d": 1},
    "ts_delta": {"d": 1},
    "ts_pct": {"d": 1},
    "ts_rank": {"d": 2},
    "ts_ema": {"d": 2},
    "ts_decay_linear": {"d": 2},
    "ts_sharpe": {"window": 2},
    "ts_autocorr": {"window": 2, "lag": 1},
    "winsorize": {"lower": 0.05, "upper": 0.95},
    "ffill": {},
    "fillna_const": {"value": 0.0},
    "protected_log": {},
    "protected_sqrt": {},
    "scale": {},
    "normalize": {},
    "cs_demean": {},
    "rank": {},
    "zscore": {},
    "abs": {},
    "log": {},
    "clip": {"lower": 0.0, "upper": 100.0},
    "exp": {},
    "sqrt": {},
    "sign": {},
    "neg": {},
    "power": {"exponent": 2.0},
    "fillna": {"value": 0.0},
    "volatility": {"window": 2},
    "log_returns": {"periods": 1},
    "winsorize": {"lower": 0.05, "upper": 0.95},
    "group_winsorize": {"lower": 0.05, "upper": 0.95},
    "vwap": {"d": 2},
    "ts_cov": {"d": 2},
}


def _smoke_canonicals() -> list[str]:
    return sorted(PRODUCTION_CORE_CANONICALS - _SKIP_SMOKE)


@pytest.mark.parametrize("canonical", _smoke_canonicals())
def test_production_core_unary_smoke(golden_panel, loaded, canonical: str):
    spec = build_operator_spec(canonical)
    assert spec is not None
    assert spec.allow_in_production, canonical

    op = OperatorRegistry.get(canonical, backend="pandas_numpy")
    assert op is not None, canonical

    kw = dict(_UNARY_SMOKE.get(canonical, {"d": 2}))
    panel = golden_panel.copy()
    if canonical in _BINARY_TWO_PANEL:
        out = op.calculate(panel, panel, **kw)
    elif canonical == "not_":
        out = op.calculate(panel > 0, **kw)
    else:
        out = op.calculate(panel, **kw)

    assert_panel_shape_unchanged(golden_panel, out)
    assert isinstance(out, pd.DataFrame)
    # 至少有一个有限值（全 NaN 输出视为异常 smoke）
    assert np.isfinite(out.to_numpy(dtype=float)).any(), canonical
