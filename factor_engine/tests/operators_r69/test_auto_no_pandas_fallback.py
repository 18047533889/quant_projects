"""R69 守卫：auto→polars 原生生产接线验证。

三层守卫（不转换算子，专管接线与回归钉死）：

1. ``test_polars_udf_pandas_delegate_count_is_zero``
   全量 execution_kind 分类审计：所有已注册 polars 槽位中
   ``polars_udf_pandas_delegate`` 计数必须为 0（R69 最终目标）。
   同时打印 native / unsupported 基线计数，作为本批次的可观测基线。

2. ``test_auto_backend_paths_never_fall_back_to_pandas``
   代表性算子（数学 / 滚动 / 截面 / 技术指标各若干）在 ``build_backend("auto")``
   下 ``run_many`` 的 ``backend_paths`` 必须全部指向 polars 原生（或 SQL 全下推）；
   任何 pandas 回退 / delegate / python-rolling 路径都判失败。
   该层在 research 模式运行（hermetic 合成面板），routing 决策与 production
   共享同一 auto planner，pandas 回退在这层即可被钉死。

3. ``test_production_run_many_auto_backend_paths_are_native``
   production 模式 run_many 的同构守卫（真实 bounded DataAccess）。
   若被 R37 参数域证据门禁挡住（证据过期 / 树不满足重生成条件），显式 skip
   并在 reason 中给出可行动的修复指引；证据刷新后自动转为强制执行。

任何对本守卫的放宽（例如把 delegate 回退改回允许）都必须同步修改本文件并在
commit message 中说明理由。
"""
from __future__ import annotations

import pytest

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.perf_config import PerfConfig
from factor_engine.tests.helpers import InMemorySeriesSource

# ---------------------------------------------------------------------------
# 代表算子集合：数学 / 滚动 / 截面 / 技术指标（全部持有 POLARS_NATIVE_EXPR spec）
# ---------------------------------------------------------------------------

MATH_CALLS = (
    ("abs", ("close",), {}),
    ("log", ("close",), {}),
    ("exp", ("close",), {}),
    ("sign", ("close",), {}),
    ("neg", ("close",), {}),
    ("add", ("close", "high"), {}),
    ("multiply", ("close", "high"), {}),
    ("subtract", ("high", "close"), {}),
    ("divide", ("close", "high"), {}),
    ("clip", ("close",), {"lo": -3.0, "hi": 3.0}),
)

ROLLING_CALLS = (
    ("ts_mean", ("close",), {"window": 5}),
    ("ts_std", ("close",), {"window": 10}),
    ("ts_sum", ("close",), {"window": 7}),
    ("ts_delta", ("close",), {"n": 1}),
    ("ts_delay", ("close",), {"n": 2}),
    ("ts_max", ("close",), {"window": 5}),
    ("ts_min", ("close",), {"window": 5}),
    ("ts_zscore", ("close",), {"window": 10}),
    ("ts_skew", ("close",), {"window": 10}),
    ("ts_pct", ("close",), {"d": 1}),
)

CROSS_CALLS = (
    ("rank", ("close",), {}),
    ("cs_pct_rank", ("close",), {}),
    ("c_mean", ("close",), {}),
)

TECH_CALLS = (
    ("RSI_WILDER", ("close",), {"window": 14}),
)

REPRESENTATIVE_CALLS = MATH_CALLS + ROLLING_CALLS + CROSS_CALLS + TECH_CALLS

# backend_paths 里允许的落点：polars 原生或 SQL 全下推。
_ALLOWED_ACTUAL_BACKENDS = {
    "polars_panel", "polars_long", "polars_long_native",
    "duckdb_sql", "duckdb_sql_full", "clickhouse_sql_full",
}
_ALLOWED_PRIMARY_ROUTES = {
    "polars_panel", "polars_long", "polars_long_native",
    "duckdb_sql_full", "clickhouse_sql_full", "sql_partial_polars_long",
    "sql_partial", "sql_plan_only", "polars_expr",
}


@pytest.fixture(scope="module", autouse=True)
def _registry_loaded():
    load_all()


# ---------------------------------------------------------------------------
# 守卫 1：polars_udf_pandas_delegate 全量计数 == 0（基线钉死）
# ---------------------------------------------------------------------------

def test_polars_udf_pandas_delegate_count_is_zero():
    from factor_engine.backend.polars_backend_kind import canonical_polars_kind

    ops = OperatorRegistry._operators
    delegates: list[str] = []
    no_polars: list[str] = []
    counts: dict[str, int] = {}
    for canon, backends in ops.items():
        if "polars" not in backends:
            no_polars.append(canon)
            continue
        kind = canonical_polars_kind(canon, production_mode=True)
        value = getattr(kind, "value", str(kind))
        counts[value] = counts.get(value, 0) + 1
        if value == "polars_udf_pandas_delegate":
            delegates.append(canon)

    # 基线可观测性：native / unsupported / 无 polars 槽位数量随批次演进，
    # 在输出中留痕（报告口径）。
    native_count = counts.get("polars_native", 0)
    unsupported_count = counts.get("unsupported", 0)
    print(
        f"\n[R69 baseline] polars slots: total={len(ops)} "
        f"native={native_count} unsupported={unsupported_count} "
        f"no_polars={len(no_polars)} delegate={len(delegates)}"
    )

    assert delegates == [], (
        "R69 守卫失败：以下算子的 polars 槽位被分类为 polars_udf_pandas_delegate"
        f"（pl→pandas→pl 委托，禁止进入 auto/production）：{sorted(delegates)}"
    )


# ---------------------------------------------------------------------------
# 守卫 2：auto backend_paths 无 pandas 回退（research，hermetic 合成面板）
# ---------------------------------------------------------------------------

def _synthetic_panel():
    import numpy as np
    import pandas as pd

    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    instruments = ["A", "B", "C", "D"]
    index = pd.MultiIndex.from_product([dates, instruments],
                                       names=["timestamp", "instrument"])
    rng = np.random.default_rng(6901)
    close = pd.Series(rng.uniform(8, 30, len(index)), index=index)
    high = close * rng.uniform(1.0, 1.05, len(index))
    low = close * rng.uniform(0.95, 1.0, len(index))
    return {"close": close, "high": high, "low": low}


def _build_representative_factors() -> list[Factor]:
    factors: list[Factor] = []
    for canonical, args, kwargs in REPRESENTATIVE_CALLS:
        factory = make_cleaned_call_factory(canonical)
        expr = factory(*(col(a) for a in args), **kwargs)
        factors.append(Factor(f"r69_{canonical.lower()}", expr))
    return factors


def _assert_native_path(path, name: str) -> None:
    summary = path.get("backend_path_summary") or {}
    actual = (path.get("physical_plan", {}) or {}).get("actual_backend") \
        or summary.get("actual_backend") or path.get("backend")
    assert actual in _ALLOWED_ACTUAL_BACKENDS, (
        f"{name}: backend 落点 {actual!r} 不是 polars 原生/SQL（summary={summary}）"
    )
    route = summary.get("primary_route")
    assert route is None or route in _ALLOWED_PRIMARY_ROUTES, (
        f"{name}: primary_route={route!r} 含 pandas/python 回退（summary={summary}）"
    )
    fallback_ops = summary.get("pandas_fallback_ops") or []
    assert fallback_ops == [], (
        f"{name}: 出现 pandas 回退算子 {fallback_ops}（summary={summary}）"
    )
    fastpath = summary.get("fastpath_route")
    assert fastpath != "fallback", (
        f"{name}: fastpath_route=fallback（summary={summary}）"
    )
    final_collect = summary.get("final_collect")
    assert final_collect != "fallback_to_pandas_or_polars", (
        f"{name}: final_collect 发生 pandas 回退（summary={summary}）"
    )


def _auto_engine_panel_source():
    from types import SimpleNamespace

    class _CostedSource(InMemorySeriesSource):
        def estimate_scan_cost(self, *, fields, time_range=None, instruments=None):
            rows = len(next(iter(self.data.values())))
            return SimpleNamespace(
                selected_bytes=rows * 8 * len(fields),
                projection_bytes=rows * 8 * len(fields),
                estimated_rows=rows, instrument_count=4,
                file_count=0, remote=False,
            )

    return _CostedSource(_synthetic_panel())


def test_auto_backend_paths_never_fall_back_to_pandas():
    factors = _build_representative_factors()
    engine = FactorEngine(
        build_backend("auto"), _auto_engine_panel_source(), run_mode="research")
    result = engine.run_many(factors, perf=PerfConfig(max_workers=1))

    assert set(result["backend_paths"]) >= {f.name for f in factors}
    for factor in factors:
        _assert_native_path(result["backend_paths"][factor.name], factor.name)


# ---------------------------------------------------------------------------
# 守卫 3：production run_many 的同构守卫（真实 bounded DataAccess）
# ---------------------------------------------------------------------------

def _parameter_domain_gate_blocker() -> str:
    """production 准入的 R37 证据门禁自检；返回阻塞原因或空串。"""
    from factor_engine.runtime.exceptions import ParameterDomainError

    try:
        from factor_engine.runtime.parameter_domain_store import (
            assert_parameter_domain_ready,
        )
        assert_parameter_domain_ready()
        return ""
    except ParameterDomainError as exc:
        return str(exc)
    except Exception as exc:  # pragma: no cover - 防御性
        return f"{type(exc).__name__}: {exc}"


_GATE_BLOCKER = _parameter_domain_gate_blocker()

_PRODUCTION_SKIP_REASON = (
    "production 准入被 R37 参数域证据门禁挡住（证据过期，且当前并发工作树"
    "不满足审计脚本的干净树前置条件）——属于证据过期类，修复动作：待工作树"
    "收敛后重跑 scripts/audit_r37_parameter_domains.py 生成新证据。"
    "本守卫将在证据刷新后自动转为强制执行。门禁详情：{}"
).format(_GATE_BLOCKER)


@pytest.mark.skipif(bool(_GATE_BLOCKER), reason=_PRODUCTION_SKIP_REASON)
def test_production_run_many_auto_backend_paths_are_native(monkeypatch, tmp_path):
    import os

    from factor_engine.api.factor import FactorExecutionScopeHint
    from factor_engine.runtime.production_policy import ProductionPolicyViolation
    from factor_engine.storage.factory import DataSourceBuildContext, build_data_source

    cos_parquet_root = "/home/sunhaiwei/quant_projects/data/a_share/lqtp_data"
    symbols = ("000001.SZ", "600000.SH")
    configured_cli = os.environ.get(
        "DATA_ACCESS_COS_CLI", os.environ.get("ASHARE_COS_CLI", "clean-cos-ro"))
    monkeypatch.setenv("ASHARE_PARQUET_ROOT", cos_parquet_root)
    monkeypatch.delenv("DATA_ACCESS_SKIP_COS_MIRROR", raising=False)
    monkeypatch.setenv("DATA_ACCESS_COS_READ_MODE", "remote")
    monkeypatch.setenv("DATA_ACCESS_COS_REMOTE_BACKEND", "cli")
    monkeypatch.setenv("DATA_ACCESS_COS_CLI", configured_cli)
    cache_root = tmp_path / "cos-cache"
    cache_root.mkdir(mode=0o700)
    cache_root.chmod(0o700)
    monkeypatch.setenv("DATA_ACCESS_COS_CACHE_ROOT", str(cache_root))
    config = {
        "type": "data_access",
        "dataset": "ashare_stock_daily_adj",
        "fields": {"close": "AdjClose", "high": "AdjHigh", "low": "AdjLow"},
        "start_date": "2024-01-02",
        "end_date": "2024-02-29",
        "instrument_filter": list(symbols),
        "read_auto": True,
    }
    context = DataSourceBuildContext(
        run_mode="production", market="ashare", calendar_id="SSE", pit_enforce=True,
        timezone="Asia/Shanghai",
    )
    source = build_data_source(config, build_context=context)

    identity = FactorExecutionScopeHint(
        market="ashare",
        universe_id="ASHARE_SMOKE_2_SYMBOLS",
        frequency="1d",
        calendar_id="SSE",
        decision_time_policy="close_to_close",
    )
    factors = []
    for canonical, args, kwargs in REPRESENTATIVE_CALLS:
        factory = make_cleaned_call_factory(canonical)
        expr = factory(*(col(a) for a in args), **kwargs)
        factors.append(Factor(
            f"r69_prod_{canonical.lower()}", expr,
            source_expr=f"{canonical}({', '.join(args)})",
            semantic_identity=identity,
        ))

    engine = FactorEngine(
        backend=build_backend("auto"), data_source=source, run_mode="production",
    )
    result = engine.run_many(
        factors,
        perf=PerfConfig(max_workers=1),
        input_dq_check=True,
        auto_warmup=True,
    )

    assert set(result["backend_paths"]) >= {f.name for f in factors}
    for factor in factors:
        _assert_native_path(result["backend_paths"][factor.name], factor.name)


def test_production_gate_blocker_is_reported_when_present():
    """门禁自检一致性：若 production 被挡，skipif reason 必须与实际门禁一致。

    保证守卫 3 的 skip 不会吞掉真实回归——门禁一旦刷新（_GATE_BLOCKER 为空），
    守卫 3 强制执行；本测试只校验诊断路径本身可信。
    """
    if _GATE_BLOCKER:
        assert "ParameterDomain" in _GATE_BLOCKER or "参数域" in _GATE_BLOCKER, (
            f"未预期的门禁阻塞形态，需要人工确认：{_GATE_BLOCKER}"
        )
    else:
        # 门禁已刷新：production 守卫此刻应处于强制执行状态。
        import factor_engine.tests.operators_r69.test_auto_no_pandas_fallback as _self

        assert not getattr(_self, "_GATE_BLOCKER", "sentinel")
