"""算子代价模型：复杂度 / 内存 / 执行 Tier 路由 + backend-aware 成本。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    "delay": OperatorCost("O(N)", "low", True, 0, True, False),
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
    "bfill": OperatorCost("O(N)", "low", True, 2, True, False),
    "corr_test": OperatorCost("O(NW)", "high", False, 3, False, False),
    "cs_regression": OperatorCost("O(NK^2)", "high", False, 3, False, False),
    "cs_resid": OperatorCost("O(NK^2)", "high", False, 3, False, False),
    "cum_delta": OperatorCost("O(N)", "low", True, 2, True, False),
    "cum_first": OperatorCost("O(N)", "low", True, 2, True, False),
    "cum_prod": OperatorCost("O(N)", "low", True, 2, True, False),
    "ewm_corr": OperatorCost("O(NW)", "high", False, 1, True, True),
    "ewm_mean": OperatorCost("O(N)", "low", True, 0, True, True),
    "expanding_rank": OperatorCost("O(N log N)", "medium", False, 2, True, False),
    "fillna_const": OperatorCost("O(N)", "low", True, 2, True, False),
    "fillna_interpolate": OperatorCost("O(N)", "medium", False, 3, False, False),
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
    "c_mean": OperatorCost("O(N)", "medium", False, 2, True, False),
    "c_std": OperatorCost("O(N)", "medium", False, 2, True, False),
    "c_sum": OperatorCost("O(N)", "medium", False, 2, True, False),
    "c_count": OperatorCost("O(N)", "low", False, 2, True, False),
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
    "causal_bfill": OperatorCost("O(N)", "low", True, 2, True, False),
    "downside_beta": OperatorCost("O(NW)", "high", False, 1, False, False),
    "tail_beta": OperatorCost("O(NW)", "high", False, 1, False, False),
    "idio_vol": OperatorCost("O(NW)", "high", False, 1, False, False),
    "idio_skew": OperatorCost("O(NW)", "high", False, 1, False, False),
    "residual_momentum_capm": OperatorCost("O(NW)", "high", False, 1, False, False),
    "coskewness_to_market": OperatorCost("O(NW)", "high", False, 1, False, False),
    "rolling_beta_to_market": OperatorCost("O(NW)", "high", False, 1, True, False),
    "ts_poly2_coeff": OperatorCost("O(NW)", "high", False, 1, False, False),
    "ts_poly2_resid": OperatorCost("O(NW)", "high", False, 1, False, False),
}


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
_BENCHMARK_CACHE: dict[str, dict[str, BackendCost]] | None = None


def _load_benchmark_costs() -> dict[str, dict[str, BackendCost]]:
    """从 benchmark JSON 加载并缓存 backend 成本表。"""
    global _BENCHMARK_CACHE
    if _BENCHMARK_CACHE is not None:
        return _BENCHMARK_CACHE
    table: dict[str, dict[str, BackendCost]] = {}
    if _BENCHMARK_JSON.is_file():
        try:
            import json

            data = json.loads(_BENCHMARK_JSON.read_text(encoding="utf-8"))
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
        except Exception:
            table = {}
    _BENCHMARK_CACHE = table
    return table

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
    """
    canon = str(canon)
    bench = _load_benchmark_costs().get(canon, {})
    if backend in bench:
        return bench[backend]
    table = _BACKEND_COST_TABLE.get(canon, {})
    return table.get(backend, _DEFAULT_BACKEND_COST)


def default_backend_speedup(
    canon: str,
    backend: str,
    status: str,
) -> float:
    """相对 pandas 的粗粒度加速比（用于 capability 导出）。"""
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
) -> float:
    """估算相对成本（越小越快）。"""
    bc = get_backend_cost(canon, backend)
    conv = bc.requires_conversion if requires_conversion is None else requires_conversion
    rows = row_count_estimate or 500_000
    millions = max(rows / 1_000_000.0, 0.001)
    cost = bc.fixed_overhead_ms + bc.per_million_rows_ms * millions
    if conv:
        cost += 2.0 + 0.05 * millions
    return cost * bc.memory_factor


def tier1_has_explicit_cost(canon: str) -> bool:
    """Tier-1 算子是否在 cost model 中显式登记（非仅 _DEFAULT）。"""
    return canon in _COSTS


def get_operator_cost(op: str) -> OperatorCost:
    """查询算子复杂度与 tier 元数据。

    参数:
        op: 算子名称。

    返回:
        已登记则返回对应 ``OperatorCost``，否则返回默认 ``_DEFAULT``。
    """
    return _COSTS.get(str(op), _DEFAULT)


def estimate_plan_cost(plan: object) -> dict[str, object]:
    """逻辑计划子树代价摘要（节点数 + 最大 tier）。"""
    max_tier = 0
    expensive: list[str] = []
    node_count = 0

    def walk(node: object) -> None:
        nonlocal max_tier, node_count
        node_count += 1
        op = str(getattr(node, "op", "") or "")
        if op:
            cost = get_operator_cost(op)
            max_tier = max(max_tier, cost.tier)
            if cost.tier >= 3 or cost.memory == "high":
                expensive.append(op)
        for child in getattr(node, "inputs", []) or []:
            walk(child)

    walk(plan)
    return {
        "node_count": node_count,
        "max_tier": max_tier,
        "expensive_ops": sorted(set(expensive)),
    }
