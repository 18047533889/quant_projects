"""时序模块 schema 常量定义。

对应文档：evaluation/timeseries/docs/FID_timeseries_performance_series.md
职责：统一列名常量与输入必需列契约。
"""

from __future__ import annotations

COL_DATETIME = "datetime"
COL_ASSET = "asset"
COL_FACTOR_ID = "factor_id"
COL_FACTOR_VALUE = "factor_value"
COL_KNOWLEDGE_TS = "knowledge_ts"
COL_DECISION_TS = "decision_ts"
COL_HORIZON = "horizon"
COL_FORWARD_RETURN = "forward_return"
COL_IS_ACTIVE = "is_active"
COL_IS_TRADABLE = "is_tradable"
COL_INDUSTRY = "industry"

COL_EVAL_RUN_ID = "eval_run_id"
COL_VALID_ASSET_COUNT = "valid_asset_count"
COL_UNIVERSE_COUNT = "universe_count"
COL_COVERAGE = "coverage_t"
COL_IC = "ic_t"
COL_RANK_IC = "rank_ic_t"
COL_KENDALL = "kendall_tau_t"
COL_QUANTILE_ID = "quantile_id"
COL_ASSET_COUNT = "asset_count"
COL_GROSS_RETURN = "gross_return"
COL_NET_RETURN = "net_return"
COL_TOP_RETURN = "top_return"
COL_BOTTOM_RETURN = "bottom_return"
COL_TOP_MINUS_BOTTOM = "top_minus_bottom_t"
COL_CUM_RETURN = "cum_return"
COL_DRAWDOWN = "drawdown"
COL_TURNOVER = "turnover_t"

REQUIRED_PURE_FACTOR_COLUMNS = (
    COL_DATETIME,
    COL_ASSET,
    COL_FACTOR_ID,
    COL_FACTOR_VALUE,
    COL_KNOWLEDGE_TS,
)

REQUIRED_MARKET_COLUMNS_BASE = (
    COL_DATETIME,
    COL_ASSET,
)

REQUIRED_UNIVERSE_COLUMNS = (
    COL_DATETIME,
    COL_ASSET,
    COL_IS_ACTIVE,
    COL_IS_TRADABLE,
)

