"""算子代价模型：复杂度 / 内存 / 执行 Tier 路由 + backend-aware 成本。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CostContext:
    """算子代价估值的运行时上下文（行数、窗口、特征维等规模参数）。

    用于把参数规模（window / feature_dim）反映到 O(NW)/O(NK^2) 算子成本上，
    避免 window=5 与 window=500 估出相同成本。字段全为可选：缺省 None 表示
    调用方未提供该维度信息，保持无 ctx 时行为不变。
    """

    rows: int = 500_000
    instruments: int = 3000
    window: int | None = None
    k: int | None = None
    feature_dim: int | None = None
    regressors: int | None = None
    group_count: int | None = None
    average_group_size: int | None = None
    session_bars: int | None = None
    expected_density: float | None = None


@dataclass(frozen=True)
class CostSpec:
    """算子的显式代价规格：复杂度串 + 参数缩放维度 + 物化放大系数。

    参数:
        time_complexity: 大 O 复杂度串（``O(N)``、``O(NW)``、``O(NK^2)`` …）。
        memory_complexity: 内存档位（``low``/``medium``/``high``）。
        parameter_scaling: 该算子成本随哪个参数缩放：``window`` | ``feature_dim`` | ``none``。
        materialization_multiplier: 峰值物化放大系数（默认 1.0）。
    """

    time_complexity: str
    memory_complexity: str
    parameter_scaling: str = "none"
    materialization_multiplier: float = 1.0


@dataclass(frozen=True)
class OperatorCost:
    """单算子复杂度、内存与执行 tier 元数据。"""

    complexity: str
    memory: str
    supports_incremental: bool
    tier: int  # 0=bottleneck/vectorized, 1=numba, 2=polars, 3=pandas
    supports_polars: bool = True
    supports_numba: bool = False

    def to_dict(self) -> dict[str, object]:
        """序列化为字典。

        返回:
            含 complexity、memory、tier 等字段的字典。
        """
        return {
            "complexity": self.complexity,
            "memory": self.memory,
            "supports_incremental": self.supports_incremental,
            "tier": self.tier,
            "supports_polars": self.supports_polars,
            "supports_numba": self.supports_numba,
        }


_DEFAULT = OperatorCost("O(N)", "medium", False, 3, False, False)

_COSTS: dict[str, OperatorCost] = {
    "col": OperatorCost("O(1)", "low", True, 0, True, False),
    "literal": OperatorCost("O(1)", "low", True, 0, True, False),
    "add": OperatorCost("O(N)", "low", True, 2, True, False),
    "subtract": OperatorCost("O(N)", "low", True, 2, True, False),
    "multiply": OperatorCost("O(N)", "low", True, 2, True, False),
    "divide": OperatorCost("O(N)", "low", True, 2, True, False),
    "rank": OperatorCost("O(N log N)", "medium", False, 2, True, False),
    "zscore": OperatorCost("O(N)", "medium", False, 2, True, False),
    "ts_mean": OperatorCost("O(N)", "low", True, 0, True, True),
    "ts_sum": OperatorCost("O(N)", "low", True, 0, True, True),
    "ts_std": OperatorCost("O(N)", "medium", True, 0, True, True),
    "ts_std_dev": OperatorCost("O(N)", "medium", True, 0, True, True),
    "ts_min": OperatorCost("O(N)", "low", True, 0, True, True),
    "ts_max": OperatorCost("O(N)", "low", True, 0, True, True),
    "ts_delta": OperatorCost("O(N)", "low", True, 2, True, False),
    "ts_pct": OperatorCost("O(N)", "low", True, 2, True, False),
    "ts_delay": OperatorCost("O(N)", "low", True, 0, True, False),
    "ts_rank": OperatorCost("O(NW log W)", "high", False, 1, False, True),
    "ts_corr": OperatorCost("O(NW)", "high", False, 1, True, True),
    "ts_correlation": OperatorCost("O(NW)", "high", False, 1, True, True),
    "neutralize": OperatorCost("O(NK^2)", "high", False, 3, False, False),
    "group_rank": OperatorCost("O(N log N)", "high", False, 3, False, False),
    "ts_ema": OperatorCost("O(N)", "low", True, 0, True, True),
    "ts_beta": OperatorCost("O(NW)", "high", False, 1, True, True),
    "rolling_beta": OperatorCost("O(NW)", "high", False, 1, True, True),
    "RSI_WILDER": OperatorCost("O(NW)", "medium", True, 1, False, False),
    "ATR_WILDER": OperatorCost("O(NW)", "medium", True, 1, False, False),
    "ts_decay_linear": OperatorCost("O(NW)", "medium", True, 1, True, True),
    "rank_corr": OperatorCost("O(NW)", "high", False, 1, True, True),
    "winsorize": OperatorCost("O(N)", "medium", False, 2, True, False),
    "quantile": OperatorCost("O(N log N)", "medium", False, 2, True, False),
    "ffill": OperatorCost("O(N)", "low", True, 2, True, False),
    "micro_realized_vol": OperatorCost("O(NW)", "medium", True, 2, True, False),
    "real_turnover_rate": OperatorCost("O(N)", "low", True, 2, True, False),
    "WMA": OperatorCost("O(NW)", "medium", True, 1, True, True),
    "corr_test": OperatorCost("O(NW)", "high", False, 3, False, False),
    "cs_regression": OperatorCost("O(NK^2)", "high", False, 3, False, False),
    "cs_resid": OperatorCost("O(NK^2)", "high", False, 3, False, False),
    "size_neutralize": OperatorCost("O(NK^2)", "high", False, 3, False, False),
    "industry_size_neutralize": OperatorCost("O(NK^2)", "high", False, 3, False, False),
    "cum_delta": OperatorCost("O(N)", "low", True, 2, True, False),
    "cum_first": OperatorCost("O(N)", "low", True, 2, True, False),
    "cum_prod": OperatorCost("O(N)", "low", True, 2, True, False),
    "ewm_corr": OperatorCost("O(NW)", "high", False, 1, True, True),
    "ts_ema": OperatorCost("O(N)", "low", True, 0, True, True),
    "expanding_rank": OperatorCost("O(N log N)", "medium", False, 2, True, False),
    "fillna_const": OperatorCost("O(N)", "low", True, 2, True, False),
    "causal_linear_extrapolate": OperatorCost("O(N)", "medium", False, 3, False, False),
    "group_decay_linear": OperatorCost("O(NW)", "medium", True, 1, True, True),
    "group_mean": OperatorCost("O(N)", "medium", False, 2, True, False),
    "group_neutralize": OperatorCost("O(NK^2)", "high", False, 3, False, False),
    "group_winsorize": OperatorCost("O(N log N)", "medium", False, 2, True, False),
    "group_zscore": OperatorCost("O(N)", "medium", False, 2, True, False),
    "hump_decay": OperatorCost("O(NW)", "medium", True, 1, True, True),
    "micro_amihud_hf": OperatorCost("O(NW)", "medium", True, 2, True, False),
    "micro_bipower_var": OperatorCost("O(NW)", "medium", True, 2, True, False),
    "micro_jump_indicator": OperatorCost("O(NW)", "medium", True, 2, True, False),
    "micro_kyle_lambda": OperatorCost("O(NW)", "high", False, 2, True, False),
    "micro_mid_return": OperatorCost("O(N)", "low", True, 2, True, False),
    "micro_spread": OperatorCost("O(N)", "low", True, 2, True, False),
    "micro_trade_imbalance": OperatorCost("O(NW)", "medium", True, 2, True, False),
    "micro_vpin": OperatorCost("O(NW)", "high", False, 2, True, False),
    "normalize": OperatorCost("O(N)", "medium", False, 2, True, False),
    "scale": OperatorCost("O(N)", "medium", False, 2, True, False),
    "standardize": OperatorCost("O(N)", "medium", False, 2, True, False),
    "ts_topk_sum": OperatorCost("O(NW log W)", "high", False, 1, True, True),
    "vp_weighted_price": OperatorCost("O(NW)", "medium", True, 1, True, True),
    "abs": OperatorCost("O(N)", "low", True, 2, True, False),
    "log": OperatorCost("O(N)", "low", True, 2, True, False),
    "clip": OperatorCost("O(N)", "low", True, 2, True, False),
    "neg": OperatorCost("O(N)", "low", True, 2, True, False),
    "power": OperatorCost("O(N)", "low", True, 2, True, False),
    "floor": OperatorCost("O(N)", "low", True, 2, True, False),
    "ceil": OperatorCost("O(N)", "low", True, 2, True, False),
    "inverse": OperatorCost("O(N)", "low", True, 2, True, False),
    "maximum": OperatorCost("O(N)", "low", True, 2, True, False),
    "minimum": OperatorCost("O(N)", "low", True, 2, True, False),
    "gt": OperatorCost("O(N)", "low", True, 2, True, False),
    "lt": OperatorCost("O(N)", "low", True, 2, True, False),
    "ge": OperatorCost("O(N)", "low", True, 2, True, False),
    "le": OperatorCost("O(N)", "low", True, 2, True, False),
    "eq": OperatorCost("O(N)", "low", True, 2, True, False),
    "ne": OperatorCost("O(N)", "low", True, 2, True, False),
    "and_": OperatorCost("O(N)", "low", True, 2, True, False),
    "or_": OperatorCost("O(N)", "low", True, 2, True, False),
    "not_": OperatorCost("O(N)", "low", True, 2, True, False),
    "is_nan": OperatorCost("O(N)", "low", True, 2, True, False),
    "is_finite": OperatorCost("O(N)", "low", True, 2, True, False),
    "nan_to_num": OperatorCost("O(N)", "low", True, 2, True, False),
    "fillna": OperatorCost("O(N)", "low", True, 2, True, False),
    "cs_mean": OperatorCost("O(N)", "medium", False, 2, True, False),
    "cs_std": OperatorCost("O(N)", "medium", False, 2, True, False),
    "cs_sum": OperatorCost("O(N)", "medium", False, 2, True, False),
    "cs_count": OperatorCost("O(N)", "low", False, 2, True, False),
    "cs_mad": OperatorCost("O(N log N)", "medium", False, 2, True, False),
    "cs_mad_zscore": OperatorCost("O(N log N)", "medium", False, 2, True, False),
    "group_std": OperatorCost("O(N)", "medium", False, 2, True, False),
    "group_normalize": OperatorCost("O(N)", "medium", False, 2, True, False),
    "group_percentile": OperatorCost("O(N log N)", "medium", False, 2, True, False),
    "ts_cov": OperatorCost("O(NW)", "high", False, 1, True, True),
    "exp": OperatorCost("O(N)", "low", True, 2, True, False),
    "sqrt": OperatorCost("O(N)", "low", True, 2, True, False),
    "sign": OperatorCost("O(N)", "low", True, 2, True, False),
    "where": OperatorCost("O(N)", "low", True, 2, True, False),
    "if_else": OperatorCost("O(N)", "low", True, 2, True, False),
    "coalesce": OperatorCost("O(N)", "low", True, 2, True, False),
    "cs_demean": OperatorCost("O(N)", "medium", False, 2, True, False),
    "ts_var": OperatorCost("O(N)", "medium", True, 0, True, True),
    "ts_median": OperatorCost("O(NW log W)", "medium", False, 1, True, True),
    "ts_sharpe": OperatorCost("O(NW)", "medium", True, 1, True, True),
    "ts_autocorr": OperatorCost("O(NW)", "medium", True, 1, True, True),
    "protected_div": OperatorCost("O(N)", "low", True, 2, True, False),
    "safe_div_null": OperatorCost("O(N)", "low", True, 2, True, False),
    "protected_log": OperatorCost("O(N)", "low", True, 2, True, False),
    "protected_sqrt": OperatorCost("O(N)", "low", True, 2, True, False),
    "ts_zscore": OperatorCost("O(NW)", "medium", True, 1, True, True),
    "log_returns": OperatorCost("O(N)", "low", True, 2, True, False),
    "volatility": OperatorCost("O(NW)", "medium", True, 1, True, True),
    "rank_pct": OperatorCost("O(N log N)", "medium", False, 2, True, False),
    "cs_pct_rank": OperatorCost("O(N log N)", "medium", False, 2, True, False),
    "cs_quantile": OperatorCost("O(N log N)", "medium", False, 2, True, False),
    "vwap": OperatorCost("O(NW)", "medium", True, 1, True, True),
    "quarter": OperatorCost("O(N)", "low", True, 2, False, False),
    "quarter_from_cumulative": OperatorCost("O(N)", "low", True, 2, False, False),
    "ttm": OperatorCost("O(N)", "low", True, 2, False, False),
    "ttm_from_quarterly": OperatorCost("O(N)", "low", True, 2, False, False),
    "ttm_from_cumulative": OperatorCost("O(N)", "medium", False, 2, False, False),
    "yoy": OperatorCost("O(N)", "low", True, 2, False, False),
    "yoy_by_period": OperatorCost("O(N)", "low", True, 2, False, False),
    "avg2": OperatorCost("O(N)", "low", True, 2, False, False),
    "downside_beta": OperatorCost("O(NW)", "high", False, 1, False, False),
    "tail_beta": OperatorCost("O(NW)", "high", False, 1, False, False),
    "idio_vol": OperatorCost("O(NW)", "high", False, 1, False, False),
    "idio_skew": OperatorCost("O(NW)", "high", False, 1, False, False),
    "residual_momentum_capm": OperatorCost("O(NW)", "high", False, 1, False, False),
    "coskewness_to_market": OperatorCost("O(NW)", "high", False, 1, False, False),
    "rolling_beta_to_market": OperatorCost("O(NW)", "high", False, 1, True, False),
    "ts_poly2_coeff": OperatorCost("O(NW)", "high", False, 1, False, False),
    "ts_poly2_resid": OperatorCost("O(NW)", "high", False, 1, False, False),
    "ts_count_if": OperatorCost("O(N)", "low", True, 2, True, False),
    "ts_sum_if": OperatorCost("O(N)", "low", True, 2, True, False),
    "ts_mean_if": OperatorCost("O(N)", "low", True, 2, True, False),
    "ts_std_if": OperatorCost("O(N)", "medium", True, 2, True, False),
    "ts_last_if": OperatorCost("O(NW)", "medium", True, 2, True, False),
    "ts_days_since": OperatorCost("O(N)", "low", True, 2, True, False),
    "ts_true_streak": OperatorCost("O(N)", "low", True, 2, True, False),
    "cs_bucket": OperatorCost("O(N log N)", "medium", False, 2, True, False),
    "cs_multi_resid": OperatorCost("O(NK^2)", "high", False, 2, True, False),
    "cs_wls_resid": OperatorCost("O(NK^2)", "high", False, 2, True, False),
    "group_multi_resid": OperatorCost("O(NK^2)", "high", False, 2, True, False),
    "period_lag": OperatorCost("O(N)", "medium", True, 2, True, False),
    "ts_regression_tstat": OperatorCost("O(NW)", "high", False, 2, True, False),
    "ts_trend_tstat": OperatorCost("O(NW)", "high", False, 2, True, False),
    "ts_max_drawdown": OperatorCost("O(NW)", "medium", True, 2, True, False),
    "ts_partial_corr": OperatorCost("O(NW)", "high", False, 2, True, False),
    "ts_nth_value": OperatorCost("O(NW log W)", "medium", False, 2, True, False),
}


# 成本随参数缩放的高危算子族（审计 #350/#351）。
#   - window 族：O(NW) 系列，window 越大成本越高（ts_corr/ts_beta/rank_corr …）。
#   - feature_dim 族：group/neutralize/panel regression（cs_regression/neutralize …），
#     回归器数量 K 越大成本越高。
_FEATURE_DIM_OPS = frozenset({
    "neutralize",
    "group_neutralize",
    "size_neutralize",
    "industry_size_neutralize",
    "cs_regression",
    "cs_resid",
    "cs_multi_resid",
    "group_multi_resid",
    "cs_wls_resid",
})


def _resolve_canonical_name(canon: str) -> str:
    """将算子名解析为 canonical；解析失败时原样返回（兼容未 load_all 环境）。"""
    try:
        from cleaned_operators.registry import OperatorRegistry

        return OperatorRegistry.resolve_canonical(str(canon))
    except Exception:
        return str(canon)


def _cost_spec_for_registered(name: str) -> CostSpec | None:
    """查 ``_COSTS`` 表构建 ``CostSpec``；未登记返回 ``None``（不记 warning）。"""
    cost = _COSTS.get(name)
    if cost is None:
        return None
    if "NW" in cost.complexity:
        scaling = "window"
    elif "K" in cost.complexity or name in _FEATURE_DIM_OPS:
        scaling = "feature_dim"
    else:
        scaling = "none"
    return CostSpec(cost.complexity, cost.memory, scaling, 1.0)


@dataclass(frozen=True)
class BackendCost:
    """单算子 × backend 的运行成本估计（毫秒量级相对值）。"""

    backend: str
    fixed_overhead_ms: float
    per_million_rows_ms: float
    memory_factor: float
    requires_conversion: bool


_DEFAULT_BACKEND_COST = BackendCost("pandas_numpy", 1.0, 80.0, 1.0, False)

_BENCHMARK_JSON = Path(__file__).resolve().parents[1] / "benchmarks" / "backend_cost_baseline.json"
# 缓存 (table, status)：status ∈ {"missing", "corrupt", "stale", "ok"}（审计 #354）。
_BENCHMARK_CACHE: tuple[dict[str, dict[str, BackendCost]] | None, str] | None = None

# 视为「有效 measured baseline」必须携带的 provenance 字段（审计 #353）。
# 缺失任一字段即 stale——避免把无出处/无实现 hash 的旧 benchmark 当权威成本。
_BENCHMARK_PROVENANCE_REQUIRED = (
    "implementation_hash",
    "backend_versions",
    "cpu_model",
    "ram_gb",
    "rows",
    "columns",
    "window_distribution",
    "null_rate",
    "group_cardinality",
)


def _load_benchmark_costs() -> tuple[dict[str, dict[str, BackendCost]] | None, str]:
    """从 benchmark JSON 加载并缓存 backend 成本表，区分 missing/stale/corrupt。

    返回 ``(table, status)``：
      - 文件不存在 → ``(None, "missing")``
      - JSON 解析失败 → ``(None, "corrupt")``（warning）
      - provenance 不满足（非 measured / 缺字段 / seed 占位）→ ``(None, "stale")``（warning）
      - 满足 → ``(table, "ok")``

    调用方（``get_backend_cost``）在 ``table is None`` 时全走静态
    ``_BACKEND_COST_TABLE``/默认值。
    """
    global _BENCHMARK_CACHE
    if _BENCHMARK_CACHE is not None:
        return _BENCHMARK_CACHE
    if not _BENCHMARK_JSON.is_file():
        _BENCHMARK_CACHE = (None, "missing")
        return _BENCHMARK_CACHE
    try:
        import json

        data = json.loads(_BENCHMARK_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        _LOGGER.warning(
            "backend cost benchmark JSON corrupt (%s): %s", _BENCHMARK_JSON, exc
        )
        _BENCHMARK_CACHE = (None, "corrupt")
        return _BENCHMARK_CACHE
    if str(data.get("generated_by") or "") == "seed_defaults":
        _LOGGER.warning(
            "backend cost benchmark is seed_defaults placeholder, treated as stale (%s)",
            _BENCHMARK_JSON,
        )
        _BENCHMARK_CACHE = (None, "stale")
        return _BENCHMARK_CACHE
    provenance = data.get("provenance") if isinstance(data.get("provenance"), dict) else {}
    if not provenance.get("measured"):
        _LOGGER.warning(
            "backend cost benchmark provenance.measured is not true, treated as stale (%s)",
            _BENCHMARK_JSON,
        )
        _BENCHMARK_CACHE = (None, "stale")
        return _BENCHMARK_CACHE
    missing = [key for key in _BENCHMARK_PROVENANCE_REQUIRED if key not in provenance]
    if missing:
        _LOGGER.warning(
            "backend cost benchmark provenance missing fields %s, treated as stale (%s)",
            missing,
            _BENCHMARK_JSON,
        )
        _BENCHMARK_CACHE = (None, "stale")
        return _BENCHMARK_CACHE
    table: dict[str, dict[str, BackendCost]] = {}
    for canon, backends in (data.get("operators") or {}).items():
        if not isinstance(backends, dict):
            continue
        row: dict[str, BackendCost] = {}
        for backend, spec in backends.items():
            if not isinstance(spec, dict) or "error" in spec:
                continue
            bname = str(backend)
            if bname.startswith("polars_panel"):
                bname = "polars"
            elif bname.startswith("polars_long"):
                bname = "polars_long"
            elif bname == "polars_long":
                bname = "polars_long"
            elif "@" in bname:
                bname = bname.split("@", 1)[0]
                if bname == "polars_panel":
                    bname = "polars"
                elif bname == "polars_long":
                    bname = "polars_long"
            row[bname] = BackendCost(
                backend=bname,
                fixed_overhead_ms=float(spec.get("fixed_overhead_ms", 1.0)),
                per_million_rows_ms=float(spec.get("per_million_rows_ms", 80.0)),
                memory_factor=float(spec.get("memory_factor", 1.0)),
                requires_conversion=bname in {"polars", "clickhouse_sql"},
            )
        if row:
            table[str(canon)] = row
    _BENCHMARK_CACHE = (table, "ok")
    return _BENCHMARK_CACHE


def benchmark_status() -> str:
    """最近一次 benchmark 成本表加载状态：``missing`` | ``corrupt`` | ``stale`` | ``ok``。

    ``estimate_plan_cost`` 的 routing_basis 依赖该状态：只有 ``ok`` 时才视为
    measured baseline 可路由（审计 #354）。
    """
    return _load_benchmark_costs()[1]

_BACKEND_COST_TABLE: dict[str, dict[str, BackendCost]] = {
    "ts_mean": {
        "pandas_numpy": BackendCost("pandas_numpy", 1.0, 80.0, 1.0, False),
        "polars": BackendCost("polars", 3.0, 25.0, 0.8, True),
        "duckdb_sql": BackendCost("duckdb_sql", 10.0, 18.0, 0.5, False),
        "clickhouse_sql": BackendCost("clickhouse_sql", 20.0, 8.0, 0.3, False),
    },
    "rank": {
        "pandas_numpy": BackendCost("pandas_numpy", 1.0, 120.0, 1.0, False),
        "polars": BackendCost("polars", 3.0, 40.0, 0.9, True),
        "duckdb_sql": BackendCost("duckdb_sql", 12.0, 22.0, 0.5, False),
        "clickhouse_sql": BackendCost("clickhouse_sql", 22.0, 10.0, 0.35, False),
    },
    "log_returns": {
        "pandas_numpy": BackendCost("pandas_numpy", 1.0, 60.0, 1.0, False),
        "polars": BackendCost("polars", 2.0, 18.0, 0.8, True),
    },
    "volatility": {
        "pandas_numpy": BackendCost("pandas_numpy", 1.0, 90.0, 1.0, False),
        "polars": BackendCost("polars", 3.0, 28.0, 0.85, True),
    },
    "rank_pct": {
        "pandas_numpy": BackendCost("pandas_numpy", 1.0, 120.0, 1.0, False),
        "polars": BackendCost("polars", 3.0, 45.0, 0.9, True),
    },
}


def get_backend_cost(canon: str, backend: str) -> BackendCost:
    """查询单算子在指定 backend 上的运行成本估计。

    参数:
        canon: 算子 canonical 名称。
        backend: backend 名称，如 ``pandas_numpy``、``polars``。

    返回:
        含固定开销与每百万行耗时的 ``BackendCost``。

    注意:
        benchmark 表仅在 provenance 有效（``_load_benchmark_costs`` 返回 ``ok``）时
        参与查询；``stale``/``corrupt``/``missing`` 时全走静态
        ``_BACKEND_COST_TABLE``/``_DEFAULT_BACKEND_COST``。
    """
    canon = str(canon)
    bench, _status = _load_benchmark_costs()
    if bench:
        row = bench.get(canon, {})
        if backend in row:
            return row[backend]
    table = _BACKEND_COST_TABLE.get(canon, {})
    return table.get(backend, _DEFAULT_BACKEND_COST)


def cost_spec_for(canon: str) -> CostSpec:
    """返回算子的显式代价规格 ``CostSpec``。

    参数:
        canon: 算子 canonical 名称。

    返回:
        已登记则解析 ``_COSTS`` 复杂度/内存字段与参数缩放维度；未登记返回
        ``CostSpec("O(N)", "medium", "none", 1.0)`` 并打 warning 标记
        「research-only: no explicit CostSpec」（生产矿上未登记成本的算子必须显式声明，
        见 ``production_requires_cost_spec``）。
    """
    name = _resolve_canonical_name(canon)
    spec = _cost_spec_for_registered(name)
    if spec is None:
        _LOGGER.warning(
            "research-only: no explicit CostSpec for canonical %r; "
            "defaulting to O(N)/medium/none/1.0",
            name,
        )
        return CostSpec("O(N)", "medium", "none", 1.0)
    return spec


def production_requires_cost_spec(canon: str) -> bool:
    """生产模式要求显式 CostSpec 的判定。

    当 canonical 不在 ``_COSTS`` 显式登记、且当前 run_mode 为 ``production`` 时
    返回 ``True``——生产矿上未登记成本的算子必须显式声明 CostSpec，不允许静默落到
    ``_DEFAULT``（O(N)/medium）低估风险。

    参数:
        canon: 算子 canonical 名称。

    返回:
        是否要求显式 CostSpec。
    """
    if _resolve_canonical_name(canon) in _COSTS:
        return False
    try:
        from runtime.production_policy import resolve_run_mode

        return resolve_run_mode() == "production"
    except Exception:
        return False


def default_backend_speedup(
    canon: str,
    backend: str,
    status: str,
) -> float:
    """估算相对 pandas 的粗粒度加速比（用于 capability 导出）。

    参数:
        canon: 算子 canonical 名称。
        backend: 目标 backend 名称。
        status: 能力状态；``unsupported`` 时返回 ``1.0``。

    返回:
        相对 pandas 的加速比（>= 1.0）。
    """
    if status == "unsupported":
        return 1.0
    base = get_backend_cost(canon, "pandas_numpy")
    other = get_backend_cost(canon, backend)
    if other.per_million_rows_ms <= 0:
        return 1.0
    return max(1.0, base.per_million_rows_ms / other.per_million_rows_ms)


def estimate_backend_cost(
    canon: str,
    backend: str,
    *,
    row_count_estimate: int | None = None,
    requires_conversion: bool | None = None,
    cost_ctx: CostContext | None = None,
) -> float:
    """估算相对成本（越小越快）。

    参数:
        canon: 算子 canonical 名称。
        backend: 目标 backend 名称。
        row_count_estimate: 可选行数估计，默认 50 万行。
        requires_conversion: 是否需格式转换开销；为 ``None`` 时使用成本表默认值。
        cost_ctx: 运行时规模上下文（审计 #350）。提供 ``window`` 时对 O(NW) 算子
            乘 ``max(1, window/20)``（clamp 下限 1，上限 100）；提供 ``feature_dim``
            时对 group/neutralize/panel regression 族乘 ``max(1, feature_dim/10)``。
            为 ``None`` 时行为与旧版完全一致。

    返回:
        相对成本浮点数。
    """
    bc = get_backend_cost(canon, backend)
    conv = bc.requires_conversion if requires_conversion is None else requires_conversion
    rows = row_count_estimate or 500_000
    millions = max(rows / 1_000_000.0, 0.001)
    cost = bc.fixed_overhead_ms + bc.per_million_rows_ms * millions
    if conv:
        cost += 2.0 + 0.05 * millions
    cost *= bc.memory_factor
    if cost_ctx is not None:
        name = _resolve_canonical_name(canon)
        op_cost = _COSTS.get(name)
        if cost_ctx.window is not None and op_cost is not None and "NW" in op_cost.complexity:
            window_factor = max(1.0, min(100.0, cost_ctx.window / 20.0))
            cost *= window_factor
        if cost_ctx.feature_dim is not None and (
            name in _FEATURE_DIM_OPS or (op_cost is not None and "K" in op_cost.complexity)
        ):
            dim_factor = max(1.0, cost_ctx.feature_dim / 10.0)
            cost *= dim_factor
    return cost


def tier1_has_explicit_cost(canon: str) -> bool:
    """判断 Tier-1 算子是否在 cost model 中显式登记。

    参数:
        canon: 算子 canonical 名称。

    返回:
        是否在 ``_COSTS`` 字典中有专属条目（非仅 ``_DEFAULT``）。
    """
    try:
        from cleaned_operators.registry import OperatorRegistry

        name = OperatorRegistry.resolve_canonical(str(canon))
    except Exception:
        name = str(canon)
    return name in _COSTS


def get_operator_cost(op: str) -> OperatorCost:
    """查询算子复杂度与 tier 元数据。

    参数:
        op: 算子名称。

    返回:
        已登记则返回对应 ``OperatorCost``，否则返回默认 ``_DEFAULT``。

    注意:
        生产模式（run_mode=production）下，未在 ``_COSTS`` 登记成本的算子必须通过
        ``production_requires_cost_spec`` 显式声明 CostSpec，不允许静默落到 ``_DEFAULT``。
    """
    return _COSTS.get(_resolve_canonical_name(op), _DEFAULT)


_TIER_WORK_WEIGHT = {0: 1.0, 1: 10.0, 2: 3.0, 3: 8.0}
_MEMORY_FACTOR = {"high": 3.0, "medium": 2.0, "low": 1.0}
_DEFAULT_PLAN_ROWS = 500_000


def estimate_plan_cost(plan: object, *, rows: int | None = None) -> dict[str, object]:
    """计算逻辑计划子树代价摘要（节点数 + 最大 tier + 估真实 Work/峰值内存）。

    参数:
        plan: 逻辑计划根节点或子树。
        rows: 可选行数估计；缺省 50 万行（用于 total_work 行数系数与峰值内存估算）。

    返回:
        含 ``node_count``、``max_tier``、``expensive_ops``（兼容字段）以及
        ``total_work``、``peak_live_memory_bytes``、``shared_node_count``、
        ``routing_basis`` 的字典（审计 #352/#354）。
    """
    max_tier = 0
    expensive: list[str] = []
    node_count = 0
    shared_ids: set[int] = set()
    total_work = 0.0
    max_mem_factor = 1.0
    max_mat_multiplier = 1.0

    row_count = rows or _DEFAULT_PLAN_ROWS
    rows_coeff = max(1.0, row_count / float(_DEFAULT_PLAN_ROWS))

    def walk(node: object) -> None:
        """递归遍历计划子树并累计代价统计。"""
        nonlocal max_tier, node_count, total_work, max_mem_factor, max_mat_multiplier
        node_count += 1
        shared_ids.add(id(node))
        op = str(getattr(node, "op", "") or "")
        if op:
            cost = get_operator_cost(op)
            max_tier = max(max_tier, cost.tier)
            if cost.tier >= 3 or cost.memory == "high":
                expensive.append(op)
            total_work += _TIER_WORK_WEIGHT.get(cost.tier, 3.0) * _complexity_extra(cost.complexity)
            max_mem_factor = max(max_mem_factor, _MEMORY_FACTOR.get(cost.memory, 1.0))
            spec = _cost_spec_for_registered(_resolve_canonical_name(op))
            if spec is not None:
                max_mat_multiplier = max(max_mat_multiplier, spec.materialization_multiplier)
        for child in getattr(node, "inputs", []) or []:
            walk(child)

    walk(plan)
    peak_bytes = max(1, row_count) * 8 * max_mem_factor * max_mat_multiplier
    return {
        "node_count": node_count,
        "max_tier": max_tier,
        "expensive_ops": sorted(set(expensive)),
        "total_work": round(total_work * rows_coeff, 3),
        "peak_live_memory_bytes": int(peak_bytes),
        "shared_node_count": len(shared_ids),
        "routing_basis": benchmark_status(),
    }


def _complexity_extra(complexity: str) -> float:
    """O(N log N)/O(NW) 相对 O(N) 的额外 work 系数（审计 #352）。"""
    if "N log N" in complexity or "NW" in complexity:
        return 5.0
    return 1.0


def record_plan_actual(
    plan: object,
    *,
    actual_elapsed_ms: float,
    rss_delta_bytes: int,
    rows: int = 0,
    instruments: int = 0,
    backend: str = "pandas_numpy",
    market: str = "",
    frequency: str = "",
    window: int | None = None,
) -> None:
    """R27-016/172：task 运行后把实际成本喂给在线校准（EMA）。

    由 AdaptiveBatchScheduler 在每个 task 完成后调用；后续同一
    (operator, backend, shape, window) 的预测会乘上校准因子。
    """
    try:
        from runtime.runtime_calibration import record_task_actual

        summary = estimate_plan_cost(plan, rows=rows or None)
        predicted_ms = float(summary.get("total_work", 0.0))
        predicted_peak = int(summary.get("peak_live_memory_bytes", 0))
        record_task_actual(
            operator=str(getattr(plan, "op", "") or "plan"),
            backend=backend,
            actual_elapsed_ms=actual_elapsed_ms,
            rss_delta_bytes=rss_delta_bytes,
            rows=rows,
            instruments=instruments,
            window=window,
            market=market,
            frequency=frequency,
            predicted_ms=predicted_ms,
            predicted_peak_bytes=predicted_peak,
        )
    except Exception:  # noqa: BLE001 - calibration 是尽力而为
        pass


def calibrated_plan_peak_bytes(
    plan: object,
    *,
    rows: int | None = None,
    instruments: int = 0,
    backend: str = "pandas_numpy",
    market: str = "",
    frequency: str = "",
    window: int | None = None,
) -> tuple[int, float]:
    """R27-019/172：``predicted_peak = static_peak * calibrated_memory_factor *
    uncertainty``——数值峰值 + 校准，不再是 low/medium/high 档。

    返回 ``(predicted_peak_bytes, uncertainty)``。
    """
    summary = estimate_plan_cost(plan, rows=rows)
    static_peak = int(summary.get("peak_live_memory_bytes", 0))
    try:
        from runtime.runtime_calibration import (
            calibration_key,
            calibrated_peak_bytes,
        )
        from runtime.runtime_calibration import _shape_bucket, _window_bucket

        key = calibration_key(
            operator=str(getattr(plan, "op", "") or "plan"),
            backend=backend,
            shape_bucket=_shape_bucket(rows or 0, instruments),
            window_bucket=_window_bucket(window),
            market=market,
            frequency=frequency,
        )
        return calibrated_peak_bytes(static_peak, key)
    except Exception:
        return static_peak, 1.30
