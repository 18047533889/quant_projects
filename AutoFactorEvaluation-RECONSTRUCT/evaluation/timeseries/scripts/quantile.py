"""分层收益、头尾差与覆盖率计算。

对应文档：evaluation/timeseries/docs/FID_timeseries_performance_series.md
职责：生成 quantile 面板、top-bottom 序列与多空毛收益输入（FID §2.4、§2.6）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TimeseriesConfig
from .schemas import (
    COL_ASSET,
    COL_ASSET_COUNT,
    COL_BOTTOM_RETURN,
    COL_COVERAGE,
    COL_DATETIME,
    COL_EVAL_RUN_ID,
    COL_FACTOR_ID,
    COL_FACTOR_VALUE,
    COL_FORWARD_RETURN,
    COL_GROSS_RETURN,
    COL_HORIZON,
    COL_INDUSTRY,
    COL_IS_ACTIVE,
    COL_IS_TRADABLE,
    COL_QUANTILE_ID,
    COL_TOP_MINUS_BOTTOM,
    COL_TOP_RETURN,
    COL_UNIVERSE_COUNT,
    COL_VALID_ASSET_COUNT,
)
from .validation import Severity, ValidationBuffer


def _sector_rank(values: pd.Series, industries: pd.Series) -> pd.Series:
    """行业内 rank；缺行业样本保持 NaN（FID §2.4）。"""
    out = pd.Series(np.nan, index=values.index, dtype=float)
    valid = industries.notna()
    if int(valid.sum()) == 0:
        return out
    out.loc[valid] = values.loc[valid].groupby(industries.loc[valid]).rank(method="average")
    return out


def _assign_quantile_ids(sub: pd.DataFrame, config: TimeseriesConfig) -> pd.Series:
    """单个 (date, horizon) 横截面分配 1-based 分位编号（FID §2.4）。"""
    factor = sub[COL_FACTOR_VALUE]
    if config.rank_scope == "sector_relative" and COL_INDUSTRY in sub.columns:
        ranks = _sector_rank(factor, sub[COL_INDUSTRY])
    else:
        ranks = factor.rank(method="average")

    valid = ranks.notna()
    out = pd.Series(np.nan, index=sub.index, dtype=float)
    if int(valid.sum()) < config.n_quantiles:
        return out
    try:
        out.loc[valid] = pd.qcut(ranks.loc[valid], q=config.n_quantiles, labels=False, duplicates="drop").astype(float) + 1.0
    except ValueError:
        return out
    return out


def _aggregate_quantile_returns(
    sub: pd.DataFrame,
    qid: pd.Series,
    config: TimeseriesConfig,
) -> tuple[dict[float, float], dict[float, int]]:
    """按分位聚合组收益与样本数（FID §2.4；等权 mean / 市值加权 sum(ret*mcap)/sum(mcap)）。"""
    masked = qid.notna()
    if not bool(masked.any()):
        return {}, {}

    q_values = qid.loc[masked].astype(float)
    sub_m = sub.loc[masked]
    grouped = sub_m.groupby(q_values, sort=True)
    cnt_by_q = {float(k): int(v) for k, v in grouped.size().items()}

    if config.weighting == "cap_weighted" and "mcap" in sub_m.columns:
        mcap = sub_m["mcap"].astype(float)
        valid_w = np.isfinite(mcap.to_numpy(dtype=float, copy=False)) & (mcap.to_numpy(dtype=float, copy=False) > 0)
        valid = sub_m.loc[valid_w]
        q_valid = q_values.loc[valid_w]
        if valid.empty:
            gross_by_q = {q: np.nan for q in cnt_by_q}
        else:
            wx = valid[COL_FORWARD_RETURN].astype(float) * valid["mcap"].astype(float)
            w = valid["mcap"].astype(float)
            agg = pd.DataFrame({"_wx": wx, "_w": w}, index=valid.index).groupby(q_valid, sort=True).sum()
            gross_by_q = {}
            for q_val, row in agg.iterrows():
                sw = float(row["_w"])
                gross_by_q[float(q_val)] = float(row["_wx"] / sw) if sw > config.eps else np.nan
            for q_val in cnt_by_q:
                gross_by_q.setdefault(float(q_val), np.nan)
    else:
        gross_by_q = {float(k): float(v) for k, v in grouped[COL_FORWARD_RETURN].mean().items()}

    return gross_by_q, cnt_by_q


def _side_gross_from_weights(
    sub: pd.DataFrame,
    mask: pd.Series,
    asset_key: pd.Series,
    weights: dict[str, float],
) -> float:
    """按资产权重字典计算单边加权收益（用于多空毛收益）。"""
    if not weights:
        return np.nan
    sel = mask & asset_key.isin(weights)
    if not bool(sel.any()):
        return np.nan
    rets = sub.loc[sel, COL_FORWARD_RETURN].astype(float)
    w = asset_key.loc[sel].map(weights)
    valid = w.notna()
    if not bool(valid.any()):
        return np.nan
    return float((rets.loc[valid] * w.loc[valid]).sum())


def _side_weights(group: pd.DataFrame, config: TimeseriesConfig) -> dict[str, float]:
    """单边权重 asset -> 正权重（和为 1）（FID §2.4）。"""
    if group.empty:
        return {}
    if config.weighting == "cap_weighted" and "mcap" in group.columns:
        w = group[[COL_ASSET, "mcap"]].copy()
        w["mcap"] = w["mcap"].astype(float)
        w = w.loc[np.isfinite(w["mcap"]) & (w["mcap"] > 0)]
        if w.empty:
            return {}
        sw = float(w["mcap"].sum())
        if sw <= config.eps:
            return {}
        w["weight"] = w["mcap"] / sw
        return {str(a): float(v) for a, v in zip(w[COL_ASSET], w["weight"])}
    n = int(group[COL_ASSET].nunique())
    if n <= 0:
        return {}
    eq = 1.0 / float(n)
    return {str(a): eq for a in group[COL_ASSET].astype(str).unique().tolist()}


def compute_quantile_panels(
    aligned_panel: pd.DataFrame,
    universe: pd.DataFrame,
    config: TimeseriesConfig,
    factor_id: str,
    eval_run_id: str,
    vb: ValidationBuffer,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """分位收益、头尾差、多空毛收益、coverage、换手权重（FID §2.4–§2.6）。"""
    u_counts = (
        universe.loc[universe[COL_IS_ACTIVE] & universe[COL_IS_TRADABLE]]
        .groupby(COL_DATETIME)[COL_ASSET]
        .nunique()
        .rename(COL_UNIVERSE_COUNT)
    )

    q_rows: list[dict] = []
    tb_rows: list[dict] = []
    ls_rows: list[dict] = []
    cov_rows: list[dict] = []
    weight_rows: list[dict] = []

    grouped = aligned_panel.groupby([COL_DATETIME, COL_HORIZON], sort=True)
    for (date, horizon), sub0 in grouped:
        asset_key = sub0[COL_ASSET].astype(str)
        valid_n = len(sub0)
        uni_n = u_counts.get(date, np.nan)
        coverage = np.nan if np.isnan(uni_n) else valid_n / max(int(uni_n), 1)
        if np.isnan(uni_n):
            vb.add(
                Severity.WARNING,
                "universe_missing_for_date",
                f"universe count missing for date={date}; coverage set NaN",
            )

        cov_rows.append(
            {
                "date": date,
                COL_HORIZON: int(horizon),
                COL_FACTOR_ID: factor_id,
                COL_EVAL_RUN_ID: eval_run_id,
                COL_VALID_ASSET_COUNT: int(valid_n),
                COL_UNIVERSE_COUNT: np.nan if np.isnan(uni_n) else int(uni_n),
                COL_COVERAGE: coverage,
            }
        )

        if valid_n < config.min_assets:
            tb_rows.append(
                {
                    "date": date,
                    COL_HORIZON: int(horizon),
                    COL_FACTOR_ID: factor_id,
                    COL_EVAL_RUN_ID: eval_run_id,
                    COL_TOP_RETURN: np.nan,
                    COL_BOTTOM_RETURN: np.nan,
                    COL_TOP_MINUS_BOTTOM: np.nan,
                }
            )
            ls_rows.append(
                {
                    "date": date,
                    COL_HORIZON: int(horizon),
                    COL_FACTOR_ID: factor_id,
                    COL_EVAL_RUN_ID: eval_run_id,
                    COL_GROSS_RETURN: np.nan,
                }
            )
            continue

        qid = _assign_quantile_ids(sub0, config)
        if qid.isna().all():
            vb.add(Severity.WARNING, "quantile_assignment_failed", f"date={date} horizon={horizon}")
            tb_rows.append(
                {
                    "date": date,
                    COL_HORIZON: int(horizon),
                    COL_FACTOR_ID: factor_id,
                    COL_EVAL_RUN_ID: eval_run_id,
                    COL_TOP_RETURN: np.nan,
                    COL_BOTTOM_RETURN: np.nan,
                    COL_TOP_MINUS_BOTTOM: np.nan,
                }
            )
            ls_rows.append(
                {
                    "date": date,
                    COL_HORIZON: int(horizon),
                    COL_FACTOR_ID: factor_id,
                    COL_EVAL_RUN_ID: eval_run_id,
                    COL_GROSS_RETURN: np.nan,
                }
            )
            continue

        gross_by_q, cnt_by_q = _aggregate_quantile_returns(sub0, qid, config)
        for q in sorted(gross_by_q):
            q_rows.append(
                {
                    "date": date,
                    COL_HORIZON: int(horizon),
                    COL_FACTOR_ID: factor_id,
                    COL_EVAL_RUN_ID: eval_run_id,
                    COL_QUANTILE_ID: int(q),
                    COL_GROSS_RETURN: gross_by_q[q],
                    COL_ASSET_COUNT: cnt_by_q[q],
                }
            )

        # 按"实际存在的分位"取 Qmax/Q1（FID §2.4）。qcut(duplicates="drop") 在秩重复时
        # 实际分位桶数可能少于 n_quantiles；硬编码 top=n_quantiles 会使头尾差/多空在并列值
        # 较多时静默退化为 NaN，故改为从已聚合的分位集合取最大/最小分位。
        present_q = sorted(float(q) for q in gross_by_q)
        if not present_q:
            top_q = bottom_q = None
        else:
            top_q, bottom_q = present_q[-1], present_q[0]
            if config.factor_direction < 0:
                top_q, bottom_q = bottom_q, top_q

        tr = gross_by_q.get(top_q, np.nan) if top_q is not None else np.nan
        br = gross_by_q.get(bottom_q, np.nan) if bottom_q is not None else np.nan
        spread = tr - br if np.isfinite(tr) and np.isfinite(br) else np.nan
        tb_rows.append(
            {
                "date": date,
                COL_HORIZON: int(horizon),
                COL_FACTOR_ID: factor_id,
                COL_EVAL_RUN_ID: eval_run_id,
                COL_TOP_RETURN: tr,
                COL_BOTTOM_RETURN: br,
                COL_TOP_MINUS_BOTTOM: spread,
            }
        )

        top_mask = qid == top_q
        bottom_mask = qid == bottom_q
        top_assets = set(asset_key.loc[top_mask].tolist())
        bottom_assets = set(asset_key.loc[bottom_mask].tolist())
        overlap = top_assets & bottom_assets
        if overlap:
            vb.add(
                Severity.WARNING,
                "top_bottom_overlap",
                f"date={date} horizon={horizon} top/bottom overlap exists",
                count=len(overlap),
            )
            bottom_assets -= overlap
            bottom_mask = asset_key.isin(bottom_assets)

        long_w = _side_weights(sub0.loc[top_mask], config)
        short_w = _side_weights(sub0.loc[bottom_mask], config)
        ls_gross = np.nan
        if long_w and short_w:
            long_ret = _side_gross_from_weights(sub0, top_mask, asset_key, long_w)
            short_ret = _side_gross_from_weights(sub0, bottom_mask, asset_key, short_w)
            if np.isfinite(long_ret) and np.isfinite(short_ret):
                ls_gross = long_ret - short_ret
        ls_rows.append(
            {
                "date": date,
                COL_HORIZON: int(horizon),
                COL_FACTOR_ID: factor_id,
                COL_EVAL_RUN_ID: eval_run_id,
                COL_GROSS_RETURN: ls_gross,
            }
        )

        weights: dict[str, float] = {str(a): 0.0 for a in asset_key.unique().tolist()}
        for asset, ww in long_w.items():
            weights[asset] = weights.get(asset, 0.0) + float(ww)
        for asset, ww in short_w.items():
            weights[asset] = weights.get(asset, 0.0) - float(ww)
        for asset, weight in weights.items():
            weight_rows.append(
                {
                    "date": date,
                    COL_HORIZON: int(horizon),
                    COL_FACTOR_ID: factor_id,
                    COL_ASSET: asset,
                    "weight": float(weight),
                }
            )

    return (
        pd.DataFrame(q_rows),
        pd.DataFrame(tb_rows),
        pd.DataFrame(ls_rows),
        pd.DataFrame(cov_rows),
        weight_rows,
    )
