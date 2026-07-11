"""点时一致性对齐与时区处理。

对应文档：evaluation/timeseries/docs/FID_timeseries_performance_series.md
职责：PureFactor 的 PIT 过滤，并与 forward return、universe 合并（FID §2.2）。

说明：仅处理 PureFactor 的 `knowledge_ts`/`decision_ts`（见 purification 产出），
与 market_data 字段无关。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import (
    COL_ASSET,
    COL_DATETIME,
    COL_DECISION_TS,
    COL_FACTOR_VALUE,
    COL_FORWARD_RETURN,
    COL_IS_ACTIVE,
    COL_IS_TRADABLE,
    COL_KNOWLEDGE_TS,
)
from .validation import Severity, ValidationBuffer


def _parse_timestamps_strict(series: pd.Series, policy: str, vb: ValidationBuffer, col_name: str) -> pd.Series:
    """严格解析 PureFactor 时间戳列，不隐式将 naive 当作 UTC（FID §2.2）。"""
    parsed: list[pd.Timestamp] = []
    has_naive = False
    has_aware = False

    for raw in series.tolist():
        if pd.isna(raw):
            parsed.append(pd.NaT)
            continue
        try:
            ts_obj = pd.Timestamp(raw)
        except Exception:
            parsed.append(pd.NaT)
            continue
        if ts_obj.tzinfo is None:
            has_naive = True
            parsed.append(ts_obj)
        else:
            has_aware = True
            parsed.append(ts_obj.tz_convert("UTC").tz_localize(None))

    if has_naive and has_aware and policy == "reject_mixed":
        vb.add(
            Severity.ERROR,
            "mixed_timezone_not_allowed",
            f"{col_name} contains mixed naive and tz-aware timestamps",
        )
        return pd.Series(pd.NaT, index=series.index)

    out = pd.Series(parsed, index=series.index)
    bad = int(out.isna().sum())
    if bad:
        vb.add(
            Severity.WARNING,
            "timestamp_parse_failed",
            f"{col_name} has unparsable timestamps",
            count=bad,
        )
    return out


def pit_align(
    factor_exposure: pd.DataFrame,
    forward_return: pd.DataFrame,
    universe: pd.DataFrame,
    timezone_policy: str,
    vb: ValidationBuffer,
) -> pd.DataFrame:
    """合并 PureFactor、forward return 与 universe（FID §2.2）。"""
    fe = factor_exposure.copy()
    fr = forward_return.copy()
    u = universe.copy()

    fe[COL_KNOWLEDGE_TS] = _parse_timestamps_strict(fe[COL_KNOWLEDGE_TS], timezone_policy, vb, COL_KNOWLEDGE_TS)
    fe[COL_DECISION_TS] = _parse_timestamps_strict(fe[COL_DECISION_TS], timezone_policy, vb, COL_DECISION_TS)
    fe = fe.dropna(subset=[COL_KNOWLEDGE_TS, COL_DECISION_TS]).copy()

    pit_viol = fe[COL_KNOWLEDGE_TS] > fe[COL_DECISION_TS]
    n_viol = int(pit_viol.sum())
    if n_viol:
        vb.pit_dropped_rows += n_viol
        vb.add(
            Severity.WARNING,
            "pit_dropped",
            "Dropped rows with knowledge_ts > decision_ts",
            count=n_viol,
        )
    fe = fe.loc[~pit_viol].copy()

    u_cols = [COL_DATETIME, COL_ASSET]
    for opt_col in ("industry", "mcap"):
        if opt_col in u.columns:
            u_cols.append(opt_col)
    u_sub = u.loc[
        u[COL_IS_ACTIVE] & u[COL_IS_TRADABLE],
        u_cols,
    ].drop_duplicates([COL_DATETIME, COL_ASSET])

    merged = fe.merge(fr, on=[COL_DATETIME, COL_ASSET], how="inner")
    merged = merged.merge(u_sub, on=[COL_DATETIME, COL_ASSET], how="left", indicator="_u_match")
    dates_with_universe = set(u_sub[COL_DATETIME].dropna().unique().tolist())
    has_universe = merged[COL_DATETIME].isin(dates_with_universe)
    in_universe = merged["_u_match"] == "both"
    merged = merged.loc[(~has_universe) | in_universe].drop(columns="_u_match")
    finite_mask = np.isfinite(merged[COL_FACTOR_VALUE].astype(float)) & np.isfinite(
        merged[COL_FORWARD_RETURN].astype(float)
    )
    merged = merged.loc[finite_mask].copy()
    return merged
