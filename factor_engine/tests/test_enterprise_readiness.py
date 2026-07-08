# -*- coding: utf-8 -*-
"""企业级就绪门禁：覆盖、别名契约、双 backend 对称、引擎引导。

本文件为 **CI 质量门禁**，不替代单测数值对齐；阈值随 Polars 覆盖推进而上调。
失败即表示：算子注册遗漏、dedupe 别名断裂、或 SQL/Polars 路径不对称。
"""
from __future__ import annotations

import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from backend.sql_pushdown.sql_registry import register_sql_backends, SQL_CAPABLE_CANONICALS
from runtime.env_bootstrap import bootstrap_runtime_env


@pytest.fixture(scope="module", autouse=True)
def _boot():
    bootstrap_runtime_env()
    load_all()
    register_sql_backends()


def test_polars_coverage_threshold():
    canon = [c for c in OperatorRegistry.list_canonical() if OperatorRegistry.backends_for(c)]
    polars_n = sum(1 for c in canon if "polars" in OperatorRegistry.backends_for(c))
    assert polars_n >= 220, f"polars 覆盖 {polars_n} 低于企业门禁 220"


def test_sql_coverage_threshold():
    canon = [c for c in OperatorRegistry.list_canonical() if OperatorRegistry.backends_for(c)]
    sql_n = sum(1 for c in canon if "sql" in OperatorRegistry.backends_for(c))
    assert sql_n >= 37, f"sql 覆盖 {sql_n} 低于企业门禁 37"


def test_sql_registry_synced_with_emitter_whitelist():
    assert "ts_ema" in SQL_CAPABLE_CANONICALS
    for name in ("ts_mean", "group_winsorize", "ts_beta"):
        assert name in SQL_CAPABLE_CANONICALS


def test_dedupe_removed_names_not_primary_keys():
    removed = ("Sum", "Percentile", "Mad", "returns", "clamp")
    for name in removed:
        assert name not in OperatorRegistry._operators, f"{name} 不应作为 primary 注册名"


def test_p0_ops_have_polars_backend():
    p0 = {
        "ewm_mean", "vp_weighted_price", "vpmacd", "downside_beta",
        "real_turnover_rate", "intercept", "r_squared", "row_var",
        "cs_regression", "cs_resid", "hump_decay",
        "idio_skew", "residual_momentum_capm", "coskewness_to_market",
        "group_decay_linear", "cum_prod", "ewm_corr",
        "fillna_const", "fillna_interpolate", "expanding_rank",
        "cum_delta", "cum_first", "rank_corr",
        "add", "subtract", "multiply", "divide",
        "ts_topk_sum", "is_nan", "corr_test",
    }
    missing = [n for n in p0 if "polars" not in OperatorRegistry.backends_for(n)]
    assert not missing, f"P0 缺少 polars: {missing}"


def test_minimum_maximum_dual_backend():
    assert "polars" in OperatorRegistry.backends_for("maximum")
    assert "pandas_numpy" in OperatorRegistry.backends_for("maximum")
    assert "polars" in OperatorRegistry.backends_for("minimum")
    assert "pandas_numpy" in OperatorRegistry.backends_for("minimum")


def test_no_polars_only_without_pandas_except_intentional():
    polars_only = [
        c for c in OperatorRegistry.list_canonical()
        if OperatorRegistry.backends_for(c) == ["polars"]
    ]
    assert not polars_only, f"不应存在 polars-only canonical: {polars_only}"


def test_all_load_modules_importable():
    from cleaned_operators import _LOAD_MODULES

    for mod in _LOAD_MODULES:
        __import__(mod, fromlist=["*"])


def test_implemented_canonical_has_pandas_or_polars():
    missing = []
    for canon in OperatorRegistry.list_canonical():
        backends = OperatorRegistry.backends_for(canon)
        if not backends:
            continue
        if "pandas_numpy" not in backends and "polars" not in backends:
            missing.append(canon)
    assert not missing, f"已实现算子缺少 pandas/polars runtime: {missing}"


def test_dedupe_aliases_resolve_to_implemented():
    broken = []
    for alias, canon in OperatorRegistry._aliases.items():
        if alias == canon:
            continue
        if not OperatorRegistry.backends_for(canon):
            broken.append(f"{alias}->{canon}")
    assert not broken, f"别名指向未实现 canonical: {broken[:20]}"


def test_sql_arithmetic_ops_have_polars_backend():
    """SQL 下推四则运算在 hybrid 路径也应有 polars fallback。"""
    for name in ("add", "subtract", "multiply", "divide"):
        assert "polars" in OperatorRegistry.backends_for(name), name
        assert "sql" in OperatorRegistry.backends_for(name), name


def test_tier1_operators_have_explicit_policy():
    from cleaned_operators.operator_policy import TIER1_CANONICALS, _EXPLICIT_POLICIES

    missing = sorted(c for c in TIER1_CANONICALS if c not in _EXPLICIT_POLICIES)
    assert not missing, f"Tier-1 缺少显式 OperatorPolicy: {missing}"
    assert len(TIER1_CANONICALS) >= 50, f"Tier-1 数量 {len(TIER1_CANONICALS)} 低于 50"
