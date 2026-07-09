# -*- coding: utf-8 -*-
"""企业级就绪门禁：覆盖、别名契约、双 backend 对称、引擎引导。

本文件为 **CI 质量门禁**，不替代单测数值对齐；阈值随 Polars 覆盖推进而上调。
失败即表示：算子注册遗漏、dedupe 别名断裂、或 SQL/Polars 路径不对称。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from backend.sql_pushdown.sql_registry import register_sql_backends, SQL_CAPABLE_CANONICALS
from runtime.env_bootstrap import bootstrap_runtime_env

FE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def _boot():
    bootstrap_runtime_env()
    load_all()
    register_sql_backends()


def test_polars_coverage_threshold():
    canon = [c for c in OperatorRegistry.list_canonical() if OperatorRegistry.backends_for(c)]
    polars_n = sum(1 for c in canon if "polars" in OperatorRegistry.backends_for(c))
    assert polars_n >= 320, f"polars 覆盖 {polars_n} 低于企业门禁 320"


def test_sql_coverage_threshold():
    canon = [c for c in OperatorRegistry.list_canonical() if OperatorRegistry.backends_for(c)]
    sql_n = sum(1 for c in canon if "sql" in OperatorRegistry.backends_for(c))
    assert sql_n >= 95, f"sql 覆盖 {sql_n} 低于企业门禁 95"


def test_sql_registry_synced_with_emitter_whitelist():
    assert "ts_ema" in SQL_CAPABLE_CANONICALS
    for name in ("ts_mean", "group_winsorize", "ts_beta", "group_decay_linear", "ts_cov", "power", "WMA", "ts_argmax"):
        assert name in SQL_CAPABLE_CANONICALS


def test_sql_pushdown_coverage_doc():
    doc = FE_ROOT / "docs" / "sql_pushdown_coverage.md"
    assert doc.is_file(), "运行 report_backend_coverage.py --write-doc 生成清单"
    text = doc.read_text(encoding="utf-8")
    assert "group_decay_linear" in text
    assert "normalize" in text


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


def test_examples_yaml_no_legacy_data_source_types():
    """examples/ 下配置不得再使用 legacy parquet 源（企业读端统一门禁）。"""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parent.parent / "examples"
    legacy = frozenset({"cleaned_parquet", "multi_parquet", "parquet_kline"})
    violations: list[str] = []

    def _walk(node: object) -> set[str]:
        found: set[str] = set()
        if isinstance(node, dict):
            if "type" in node:
                found.add(str(node["type"]).lower())
            for v in node.values():
                found |= _walk(v)
        elif isinstance(node, list):
            for item in node:
                found |= _walk(item)
        return found

    for path in sorted(root.rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        ds = raw.get("data_source")
        if not isinstance(ds, dict):
            continue
        bad = _walk(ds) & legacy
        if bad:
            violations.append(f"{path.relative_to(root)}: {sorted(bad)}")
    assert not violations, violations


def test_storage_write_targets_public_api():
    from storage import ClickHouseWriteTarget, resolve_write_target

    assert resolve_write_target("local").name == "local"
    assert isinstance(resolve_write_target("clickhouse"), ClickHouseWriteTarget)
