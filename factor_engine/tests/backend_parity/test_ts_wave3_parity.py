# -*- coding: utf-8
"""wave3 ts path/risk family: pandas / polars-long / DuckDB SQL 三方 parity.

覆盖算子（2026-09-08，polars 分支在 ``polars_expr_emitter.py`` 的
``_rolling_monotonicity_expr`` 等 rolling_map helper 块；SQL 分支仅限可精确
表达的组合）：

* 三方（pandas == polars_long == duckdb_sql）：
  - ts_robust_zscore_inclusive（scale="std" 组合；scale="mad" 需要 center_t
    广播进窗口每一行，DuckDB 禁嵌套窗口 → SQL 诚实回退 polars）
* 双方（pandas == polars_long；DuckDB 禁嵌套窗口 / O(n²) pair 不可表达，
  诚实保持 SQL fallback，不登记 SQL_IMPLEMENTED_CANONICALS）：
  - ts_monotonicity（两两方向一致 O(n²)）
  - ts_turning_point_ratio / ts_endpoint_deviation / ts_vol_shift_score /
    ts_recovery_fraction（尾部连续段 + 窗口内再排序）
  - ts_time_under_water / ts_current_drawdown_duration（running peak 基线
    起点依赖输出行：跨窗口起点的段不可一次物化）
  - ts_mean_abs_deviation / ts_median_abs_deviation（单中心偏差：
    两遍窗口只会得到每行自己的 center，是另一个统计量）
  - ts_upside_deviation（SQL 分支历史遗留与 pandas 内核不一致，保持
    ``_SQL_FALLBACK_CANONICALS`` 黑名单，polars 走 registry 精确内核）

数据面板含 NaN 洞 / 全常数窗 / 冷启动（窗口不满）边界。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.factory import build_data_source
from tests.helpers import InMemorySeriesSource

F = make_cleaned_call_factory


def _make_panel() -> InMemorySeriesSource:
    """30 天 × 6 instruments，含 NaN 洞 / 常数段 / 冷启动边界。"""
    load_all()
    dates = pd.date_range("2024-01-02", periods=30, freq="D")
    insts = ["A", "B", "C", "D", "E", "F"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])
    n = len(idx)
    rng = np.random.default_rng(11)
    x = pd.Series(100.0 + rng.normal(0, 2.0, n).cumsum() * 0.2, index=idx)
    # NaN 洞（物理时间断口：running peak / 尾部连续段不得跨缺口重连）
    x.iloc[5] = np.nan          # (2024-01-02, F)
    x.iloc[40] = np.nan         # (2024-01-08, E)
    x.iloc[41] = np.nan         # (2024-01-08, F)
    # 全常数窗（std / spread == 0 的退化窗）
    x.iloc[60:66] = 100.0
    x.iloc[66:72] = 101.5
    # 单日骤降 + 修复路径（recovery fraction / drawdown episode 语义）
    x.iloc[100:110] = x.iloc[100:110].to_numpy() * 0.97
    x.iloc[110] = np.nan
    return InMemorySeriesSource(data={"x": x})


@pytest.fixture(scope="module")
def panel():
    return _make_panel()


def _write_duckdb_registry(path: Path, root: Path) -> None:
    path.write_text(
        f"""ts_wave3_test_daily:
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true
  root: {root}
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    X: double
""",
        encoding="utf-8",
    )


def _seed_duckdb(root: Path, source: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for (ts, sym), val in source.data["x"].items():
        rows.append(
            {
                "TradeDate": ts.date(),
                "Symbol": sym,
                "X": float(val) if pd.notna(val) else None,
            }
        )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


@pytest.fixture
def duckdb_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, panel: InMemorySeriesSource):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    root = tmp_path / "data"
    _write_duckdb_registry(tmp_path / "datasets.yaml", root)
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(tmp_path / "datasets.yaml"))
    _seed_duckdb(root, panel)
    from data_access import reset_store

    reset_store()
    return build_data_source({"type": "data_access", "dataset": "ts_wave3_test_daily"})


def _run(source, expr, backend: str):
    return FactorEngine(
        backend=build_backend(backend), data_source=source, run_mode="research"
    ).run(Factor(name="ts_wave3_parity", expr=expr))


def _assert_parity(reference: pd.Series, candidate: pd.Series) -> None:
    pd.testing.assert_series_equal(
        reference.sort_index(),
        candidate.sort_index(),
        check_names=False,
        check_dtype=False,
        rtol=1e-6,
        atol=1e-6,
    )


def _mem(name: str):
    return col(name)


def _sql(name: str):
    return col("X")


# ---------------------------------------------------------------------------
# pandas == polars_long（rolling_map 分支 / registry 内核，精确 parity）
# ---------------------------------------------------------------------------
POLARS_CASES = [
    ("ts_monotonicity", lambda c: F("ts_monotonicity")(c("x"), 15)),
    ("ts_turning_point_ratio", lambda c: F("ts_turning_point_ratio")(c("x"), 12)),
    ("ts_endpoint_deviation", lambda c: F("ts_endpoint_deviation")(c("x"), 12)),
    ("ts_vol_shift_score", lambda c: F("ts_vol_shift_score")(c("x"), 12)),
    ("ts_time_under_water", lambda c: F("ts_time_under_water")(c("x"), 12)),
    ("ts_current_drawdown_duration", lambda c: F("ts_current_drawdown_duration")(c("x"), 12)),
    ("ts_recovery_fraction", lambda c: F("ts_recovery_fraction")(c("x"), 20)),
    ("ts_mean_abs_deviation", lambda c: F("ts_mean_abs_deviation")(c("x"), 12)),
    ("ts_median_abs_deviation", lambda c: F("ts_median_abs_deviation")(c("x"), 12)),
    ("ts_upside_deviation", lambda c: F("ts_upside_deviation")(c("x"), 12)),
    ("ts_robust_zscore_inclusive", lambda c: F("ts_robust_zscore_inclusive")(c("x"), 12)),
    (
        "ts_robust_zscore_inclusive(mean,std,clip)",
        lambda c: F("ts_robust_zscore_inclusive")(c("x"), 12, center="mean", scale="std", clip=2.5),
    ),
]


@pytest.mark.parametrize(
    ("name", "expr_builder"),
    POLARS_CASES,
    ids=[name for name, _ in POLARS_CASES],
)
def test_wave3_ts_polars_matches_pandas(panel, name, expr_builder):
    pandas_out = _run(panel, expr_builder(_mem), "pandas")["result"]
    polars_out = _run(panel, expr_builder(_mem), "polars_long")["result"]
    _assert_parity(pandas_out, polars_out)


# ---------------------------------------------------------------------------
# pandas == polars_long == duckdb_sql（可精确表达的 SQL 组合）
# 注意：positional-only 调用（attrs 为空）默认 scale="mad"，SQL 不可表达 →
# 回退 polars；仅显式 scale="std" 组合走真实 SQL 下推。
# ---------------------------------------------------------------------------
SQL_CASES = [
    (
        "ts_robust_zscore_inclusive(mean,std,clip)",
        lambda c: F("ts_robust_zscore_inclusive")(c("x"), 12, center="mean", scale="std", clip=2.5),
    ),
]


@pytest.mark.parametrize(
    ("name", "expr_builder"),
    SQL_CASES,
    ids=[name for name, _ in SQL_CASES],
)
def test_wave3_ts_sql_triple_parity(panel, duckdb_source, name, expr_builder):
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    pandas_out = _run(panel, expr_builder(_mem), "pandas")["result"]
    polars_out = _run(panel, expr_builder(_mem), "polars_long")["result"]
    _assert_parity(pandas_out, polars_out)
    sql_out = _run(duckdb_source, expr_builder(_sql), "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_out)
    _assert_parity(pandas_out, sql_out["result"])


# ---------------------------------------------------------------------------
# 诚实底线：不可精确 SQL 表达的组合必须回退，不得登记 tiers。
# ---------------------------------------------------------------------------
def test_wave3_ts_honest_sql_fallback():
    from factor_engine.backend.sql_pushdown.emitter import compile_plan_to_sql
    from factor_engine.backend.sql_pushdown.plan_fixtures import minimal_plan
    from factor_engine.backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    # 不可表达 → 不登记 SQL_IMPLEMENTED（两两 O(n²) / 尾部连续段 / 单中心偏差 /
    # running peak 基线起点依赖输出行）
    not_implemented = [
        "ts_monotonicity",
        "ts_turning_point_ratio",
        "ts_endpoint_deviation",
        "ts_vol_shift_score",
        "ts_recovery_fraction",
        "ts_mean_abs_deviation",
        "ts_median_abs_deviation",
        "ts_time_under_water",
        "ts_current_drawdown_duration",
        "ts_upside_deviation",
    ]
    for name in not_implemented:
        assert name not in SQL_IMPLEMENTED_CANONICALS, f"{name} 不应登记 SQL tiers"

    # scale="mad" 组合：DuckDB 禁嵌套窗口 → emitter 返回 None（诚实回退）
    plan = PlanNodeFactory.robust_zscore_mad()
    assert compile_plan_to_sql(
        plan, dataset="test_panel", time_column="ts", instrument_column="inst"
    ) is None


class PlanNodeFactory:
    """按名构造 minimal plan 的小工厂（供 fallback 断言复用）。"""

    @staticmethod
    def robust_zscore_mad():
        from factor_engine.backend.sql_pushdown.plan_fixtures import (
            column as _column,
            literal as _literal,
        )
        from factor_engine.planner.logical_plan import PlanNode

        return PlanNode(
            op="ts_robust_zscore_inclusive",
            inputs=[
                _column("close"),
                _literal(20.0),
                _literal("median"),
                _literal("mad"),
            ],
            attrs={"window": 20, "center": "median", "scale": "mad"},
        )


def test_wave3_ts_tier_classification():
    from factor_engine.backend.polars_long_policy import (
        POLARS_LONG_NATIVE,
        POLARS_LONG_PYTHON_ROLLING,
        infer_polars_long_tier,
    )

    python_rolling = [
        "ts_monotonicity",
        "ts_turning_point_ratio",
        "ts_endpoint_deviation",
        "ts_vol_shift_score",
        "ts_time_under_water",
        "ts_current_drawdown_duration",
        "ts_recovery_fraction",
    ]
    for name in python_rolling:
        assert name in POLARS_LONG_PYTHON_ROLLING, name
        assert name not in POLARS_LONG_NATIVE, name
        assert infer_polars_long_tier(name) == "python_rolling", name

    # MAD pair / robust zscore / upside deviation 保持 rolling_map 之前的既有路径
    for name in (
        "ts_mean_abs_deviation",
        "ts_median_abs_deviation",
    ):
        assert infer_polars_long_tier(name) == "python_rolling", name
    for name in (
        "ts_robust_zscore_inclusive",
        "ts_upside_deviation",
    ):
        assert infer_polars_long_tier(name) == "registry", name
