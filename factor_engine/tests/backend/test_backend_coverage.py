# -*- coding: utf-8
"""Backend 覆盖审计：P0 算子必须有 polars 和/或 sql backend。"""
from __future__ import annotations

import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from backend.sql_pushdown.sql_registry import register_sql_backends, SQL_CAPABLE_CANONICALS

# 因子流水线 P0：必须有 polars（auto 路径）
POLARS_REQUIRED = frozenset(
    {
        "ts_mean",
        "ts_std",
        "ts_delay",
        "ts_delta",
        "ts_pct",
        "ts_corr",
        "ts_var",
        "ts_median",
        "ts_ema",
        "rank",
        "zscore",
        "scale",
        "winsorize",
        "cs_demean",
        "group_rank",
        "group_mean",
        "group_zscore",
        "group_neutralize",
        "group_winsorize",
        "where",
        "if_else",
        "ADX",
        "AROON",
        "KAMA",
        "Slope",
        "sharpe_ratio",
        "ts_regression",
        "Beta",
        "Corr",
        "trade_when",
        "fillna",
        "idio_vol",
        "MACD",
        "WMA",
        "prev",
        "expanding_mean",
        "row_sum",
        "row_avg",
        "cum_sum",
        "abs",
        "log",
        "clip",
        "cs_regression",
        "cs_resid",
        "ewm_mean",
        "downside_beta",
        "vp_weighted_price",
        "real_turnover_rate",
        "hump_decay",
        "group_decay_linear",
        "cum_prod",
        "idio_skew",
        "fillna_interpolate",
        "rank_corr",
        "add",
        "ts_topk_sum",
    }
)

# P0 SQL 下推
SQL_REQUIRED = frozenset(
    {
        "ts_mean",
        "ts_std",
        "ts_delay",
        "ts_delta",
        "ts_pct",
        "ts_corr",
        "ts_var",
        "ts_median",
        "rank",
        "zscore",
        "scale",
        "cs_demean",
        "group_rank",
        "group_mean",
        "group_zscore",
        "group_neutralize",
        "where",
        "if_else",
        "winsorize",
        "ts_ema",
        "ts_rank",
        "ewm_mean",
        "sign",
        "ffill",
        "fillna_const",
        "ts_decay_linear",
        "coalesce",
        "protected_div",
        "protected_log",
        "protected_sqrt",
        "nan_to_num",
        "fillna",
        "is_nan",
        "is_finite",
        "normalize",
        "group_normalize",
        "group_percentile",
        "group_decay_linear",
    }
)


@pytest.fixture(scope="module", autouse=True)
def _load_ops():
    load_all()
    register_sql_backends()


def test_polars_p0_coverage():
    missing = []
    for name in sorted(POLARS_REQUIRED):
        if "polars" not in OperatorRegistry.backends_for(name):
            missing.append(name)
    assert not missing, f"缺少 polars 实现: {missing}"


def test_sql_p0_in_registry():
    missing = SQL_REQUIRED - SQL_CAPABLE_CANONICALS
    assert not missing, f"SQL 白名单未收录: {sorted(missing)}"


def test_sql_registry_backends():
    register_sql_backends()
    for name in SQL_REQUIRED:
        if name in {"column", "literal"}:
            continue
        assert "sql" in OperatorRegistry.backends_for(name), name


def test_coverage_counts():
    load_all()
    register_sql_backends()
    canon = [c for c in OperatorRegistry.list_canonical() if OperatorRegistry.backends_for(c)]
    polars_n = sum(1 for c in canon if "polars" in OperatorRegistry.backends_for(c))
    sql_n = sum(1 for c in canon if "sql" in OperatorRegistry.backends_for(c))
    assert polars_n >= 320, f"polars 覆盖过低: {polars_n}"
    assert sql_n >= 60, f"sql 覆盖过低: {sql_n}"
