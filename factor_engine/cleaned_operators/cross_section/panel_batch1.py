# -*- coding: utf-8 -*-
"""Panel / TRUE_GAP operators batch 1 (2026-08-13).

Five panel/factor operators leveraging cross-sectional and time-series patterns:

  1. ``panel_day_night_beta_gap``      -- day-session vs night-session beta gap
  2. ``pastor_stambaugh_beta``         -- Pastor-Stambaugh liquidity beta
  3. ``price_delay_score``             -- Hou-Moskowitz price delay measure
  4. ``report_asof``                   -- days since last report/event
  5. ``event_window_return_asof``      -- return in window around last event

All causal, trailing-only, deterministic.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    ParamSpec,
    SeriesOperator,
    register_operator,
)

_EPS = 1e-12
_CANONICALS: list[str] = []

_INT_SPECS: dict[str, ParamSpec] = {
    "window": ParamSpec(dtype=int, min=2),
    "min_periods": ParamSpec(dtype=int, min=2),
    "lag": ParamSpec(dtype=int, min=0),
    "pre_window": ParamSpec(dtype=int, min=0),
    "post_window": ParamSpec(dtype=int, min=0),
    "max_lookback": ParamSpec(dtype=int, min=1),
}


def _meta(name: str, description: str, params: list[str], *, unit: str = "level", cost: int = 8) -> OperatorMetadata:
    param_specs = {n: _INT_SPECS[n] for n in params if n in _INT_SPECS}
    return OperatorMetadata(
        name=name,
        category="panel_gap",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "panel_gap", "daily", "pit_safe", "causal", "typed_v2", "deterministic",
            f"signature:{','.join(params)}->series", "domain:panel_gap",
            f"unit:{unit}", f"cost:{cost}",
        ],
        param_specs=param_specs,
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


# ---------------------------------------------------------------------------
# 1. panel_day_night_beta_gap
# ---------------------------------------------------------------------------
def _rolling_beta(y: np.ndarray, x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Rolling OLS beta with intercept."""
    n = len(y)
    out = np.full(n, np.nan)
    for t in range(n):
        start = max(0, t - window + 1)
        ys = y[start : t + 1]
        xs = x[start : t + 1]
        valid = np.isfinite(ys) & np.isfinite(xs)
        if int(valid.sum() < min_periods:
            continue
        yv = ys[valid]
        xv = xs[valid]
        xbar = float(np.mean(xv))
        varx = float(np.mean((xv - xbar) ** 2))
        if varx <= _EPS:
            continue
        ybar = float(np.mean(yv))
        out[t] = float(np.mean((xv - xbar) * (yv - ybar)) / varx)
    return out


def _day_night_beta_gap(
    day_ret: pd.DataFrame,
    night_ret: pd.DataFrame,
    mkt_ret: pd.DataFrame,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    dv = day_ret.to_numpy(dtype=float)
    nv = night_ret.to_numpy(dtype=float)
    mv = mkt_ret.to_numpy(dtype=float)
    rows, cols = dv.shape
    out = np.full((rows, cols), np.nan)
    for c in range(cols):
        beta_day = _rolling_beta(dv[:, c], mv[:, c], window, min_periods)
        beta_night = _rolling_beta(nv[:, c], mv[:, c], window, min_periods)
        out[:, c] = beta_day - beta_night
    return _frame_like(day_ret, out)


@register_operator(
    name="panel_day_night_beta_gap",
    category="panel_gap",
    business_category="panel_gap",
    canonical="panel_day_night_beta_gap",
    source="cross_section.panel_batch1",
    backend="pandas_numpy",
    status="implemented",
)
class PanelDayNightBetaGap(SeriesOperator):
    """日盘 beta 与夜盘 beta 的差值（日盘 - 夜盘）。

    分别对日盘收益和夜盘收益在市场收益上做滚动 OLS beta 估计（带截距项），
    输出日盘 beta 减去夜盘 beta。正值 = 日盘对市场更敏感；负值 = 夜盘更敏感。
    causal, deterministic。
    """

    metadata = _meta(
        "panel_day_night_beta_gap",
        "日盘 beta - 夜盘 beta（滚动 OLS，带截距）。",
        ["day_ret", "night_ret", "mkt_ret", "window", "min_periods"],
        unit="beta_diff",
    )

    def _calculate_series(
        self,
        day_ret: pd.DataFrame,
        night_ret: pd.DataFrame,
        mkt_ret: pd.DataFrame,
        window: int = 63,
        min_periods: int = 40,
        **_: Any,
    ) -> pd.DataFrame:
        window = max(2, int(window))
        min_periods = max(2, int(min_periods))
        return _day_night_beta_gap(day_ret, night_ret, mkt_ret, window, min_periods)


# ---------------------------------------------------------------------------
# 2. pastor_stambaugh_beta
# ---------------------------------------------------------------------------
def _pastor_stambaugh_beta(
    ret: pd.DataFrame,
    mkt_ret: pd.DataFrame,
    liquidity_innov: pd.DataFrame,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    """Pastor-Stambaugh liquidity beta: beta on orthogonalized liquidity innovation."""
    rv = ret.to_numpy(dtype=float)
    mv = mkt_ret.to_numpy(dtype=float)
    lv = liquidity_innov.to_numpy(dtype=float)
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan)

    # Orthogonalize liquidity innovation w.r.t. market return per window
    for c in range(cols):
        for t in range(rows):
            start = max(0, t - window + 1)
            seg_m = mv[start : t + 1, c] if cols == mv.shape[1] else mv[start : t + 1, 0]
            seg_l = lv[start : t + 1, c] if cols == lv.shape[1] else lv[start : t + 1, 0]
            seg_r = rv[start : t + 1, c]

            valid = np.isfinite(seg_m) & np.isfinite(seg_l) & np.isfinite(seg_r)
            if int(valid.sum() < min_periods:
                continue

            m_v = seg_m[valid]
            l_v = seg_l[valid]
            r_v = seg_r[valid]

            # Orthogonalize: liquidity_orth = liquidity - beta_lm * market
            mbar = float(np.mean(m_v))
            var_m = float(np.mean((m_v - mbar) ** 2))
            if var_m <= _EPS:
                continue

            lbar = float(np.mean(l_v))
            beta_lm = float(np.mean((m_v - mbar) * (l_v - lbar)) / var_m)
            l_orth = l_v - beta_lm * (m_v - mbar)

            # Beta on orthogonalized liquidity
            l_orth_bar = float(np.mean(l_orth))
            var_l = float(np.mean((l_orth - l_orth_bar) ** 2))
            if var_l <= _EPS:
                continue

            rbar = float(np.mean(r_v))
            out[t, c] = float(np.mean((l_orth - l_orth_bar) * (r_v - rbar)) / var_l)

    return _frame_like(ret, out)


@register_operator(
    name="pastor_stambaugh_beta",
    category="panel_gap",
    business_category="panel_gap",
    canonical="pastor_stambaugh_beta",
    source="cross_section.panel_batch1",
    backend="pandas_numpy",
    status="implemented",
)
class PastorStambaughBeta(SeriesOperator):
    """Pastor-Stambaugh 流动性 beta（对正交化流动性冲击的 beta）。

    流动性创新先对市场收益回归取残差（正交化），再估计个股收益对正交化流动性
    的 beta。衡量个股对市场无关流动性风险的敏感度。causal, deterministic。
    """

    metadata = _meta(
        "pastor_stambaugh_beta",
        "Pastor-Stambaugh 流动性 beta（正交化流动性创新）。",
        ["ret", "mkt_ret", "liquidity_innov", "window", "min_periods"],
        unit="beta",
    )

    def _calculate_series(
        self,
        ret: pd.DataFrame,
        mkt_ret: pd.DataFrame,
        liquidity_innov: pd.DataFrame,
        window: int = 126,
        min_periods: int = 60,
        **_: Any,
    ) -> pd.DataFrame:
        window = max(2, int(window))
        min_periods = max(2, int(min_periods))
        return _pastor_stambaugh_beta(ret, mkt_ret, liquidity_innov, window, min_periods)


# ---------------------------------------------------------------------------
# 3. price_delay_score
# ---------------------------------------------------------------------------
def _price_delay_score(
    ret: pd.DataFrame,
    mkt_ret: pd.DataFrame,
    window: int,
    lag: int,
    min_periods: int,
) -> pd.DataFrame:
    """Hou-Moskowitz price delay: unrestricted R² - restricted R² from AR."""
    rv = ret.to_numpy(dtype=float)
    mv = mkt_ret.to_numpy(dtype=float)
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan)

    for c in range(cols):
        for t in range(rows):
            start = max(0, t - window + 1)
            seg_r = rv[start : t + 1, c]
            seg_m = mv[start : t + 1, c] if cols == mv.shape[1] else mv[start : t + 1, 0]

            if len(seg_r) < min_periods + lag:
                continue

            # Build lagged market returns matrix [m_t, m_{t-1}, ..., m_{t-lag}]
            valid_idx = []
            X_restricted = []
            X_unrestricted = []
            y = []

            for i in range(lag, len(seg_r)):
                if not np.isfinite(seg_r[i]) or not np.isfinite(seg_m[i]):
                    continue

                # Check all lags are finite
                lags_ok = True
                lag_vals = []
                for d in range(1, lag + 1):
                    if i - d < 0 or not np.isfinite(seg_m[i - d]):
                        lags_ok = False
                        break
                    lag_vals.append(seg_m[i - d])

                if not lags_ok:
                    continue

                valid_idx.append(i)
                y.append(seg_r[i])
                X_restricted.append([seg_m[i]])  # only contemporaneous
                X_unrestricted.append([seg_m[i]] + lag_vals)  # contemp + lags

            if len(y) < min_periods:
                continue

            y = np.array(y)
            X_res = np.array(X_restricted)
            X_unres = np.array(X_unrestricted)

            # Restricted R²: only contemporaneous market
            ybar = float(np.mean(y))
            sst = float(np.sum((y - ybar) ** 2))
            if sst <= _EPS:
                continue

            # OLS for restricted
            X_res_centered = X_res - X_res.mean(axis=0)
            y_centered = y - ybar
            XtX_res = X_res_centered.T @ X_res_centered
            if np.abs(XtX_res).max() <= _EPS:
                continue

            beta_res = np.linalg.lstsq(X_res_centered, y_centered, rcond=None)[0]
            y_pred_res = X_res_centered @ beta_res + ybar
            ssr_res = float(np.sum((y - y_pred_res) ** 2))
            r2_res = max(0.0, 1.0 - ssr_res / sst)

            # OLS for unrestricted
            X_unres_centered = X_unres - X_unres.mean(axis=0)
            XtX_unres = X_unres_centered.T @ X_unres_centered
            if np.linalg.matrix_rank(XtX_unres) < X_unres.shape[1]:
                continue

            beta_unres = np.linalg.lstsq(X_unres_centered, y_centered, rcond=None)[0]
            y_pred_unres = X_unres_centered @ beta_unres + ybar
            ssr_unres = float(np.sum((y - y_pred_unres) ** 2))
            r2_unres = max(0.0, 1.0 - ssr_unres / sst)

            # Delay score = unrestricted - restricted R²
            out[t, c] = max(0.0, r2_unres - r2_res)

    return _frame_like(ret, out)


@register_operator(
    name="price_delay_score",
    category="panel_gap",
    business_category="panel_gap",
    canonical="price_delay_score",
    source="cross_section.panel_batch1",
    backend="pandas_numpy",
    status="implemented",
)
class PriceDelayScore(SeriesOperator):
    """Hou-Moskowitz 价格延迟度量（未限制 R² - 限制 R²）。

    限制模型：ret_t ~ mkt_t（仅同期市场收益）；未限制模型：ret_t ~ mkt_t +
    mkt_{t-1} + ... + mkt_{t-lag}（含滞后市场收益）。延迟得分 = 未限制 R² -
    限制 R²，衡量个股对市场信息的吸收滞后程度。causal, deterministic。
    """

    metadata = _meta(
        "price_delay_score",
        "Hou-Moskowitz 价格延迟得分（滞后市场收益的增量 R²）。",
        ["ret", "mkt_ret", "window", "lag", "min_periods"],
        unit="r_squared_diff",
    )

    def _calculate_series(
        self,
        ret: pd.DataFrame,
        mkt_ret: pd.DataFrame,
        window: int = 126,
        lag: int = 4,
        min_periods: int = 60,
        **_: Any,
    ) -> pd.DataFrame:
        window = max(2, int(window))
        lag = max(0, int(lag))
        min_periods = max(2, int(min_periods))
        return _price_delay_score(ret, mkt_ret, window, lag, min_periods)


# ---------------------------------------------------------------------------
# 4. report_asof
# ---------------------------------------------------------------------------
def _report_asof(
    event: pd.DataFrame,
    max_lookback: int,
) -> pd.DataFrame:
    """Days since last event (report/announcement), NaN if beyond max_lookback."""
    ev = event.to_numpy(dtype=float)
    rows, cols = ev.shape
    out = np.full((rows, cols), np.nan)

    for c in range(cols):
        last_event = -10**9
        for t in range(rows):
            if np.isfinite(ev[t, c]) and ev[t, c] != 0.0:
                last_event = t

            if last_event >= 0:
                days_since = t - last_event
                if days_since <= max_lookback:
                    out[t, c] = float(days_since)

    return _frame_like(event, out)


@register_operator(
    name="report_asof",
    category="panel_gap",
    business_category="panel_gap",
    canonical="report_asof",
    source="cross_section.panel_batch1",
    backend="pandas_numpy",
    status="implemented",
)
class ReportAsof(SeriesOperator):
    """自上次报告/事件以来的天数（超过 max_lookback 为 NaN）。

    输入 ``event`` 为 0/1 指示器（或非零值均视为事件），输出当前距离最近一次
    事件的天数。事件当日输出 0，次日输出 1。超过 max_lookback 天无事件时输出
    NaN。causal, deterministic。
    """

    metadata = _meta(
        "report_asof",
        "距离上次事件的天数（超过 max_lookback 为 NaN）。",
        ["event", "max_lookback"],
        unit="days",
        cost=3,
    )

    def _calculate_series(
        self,
        event: pd.DataFrame,
        max_lookback: int = 365,
        **_: Any,
    ) -> pd.DataFrame:
        max_lookback = max(1, int(max_lookback))
        return _report_asof(event, max_lookback)


# ---------------------------------------------------------------------------
# 5. event_window_return_asof
# ---------------------------------------------------------------------------
def _event_window_return_asof(
    ret: pd.DataFrame,
    event: pd.DataFrame,
    pre_window: int,
    post_window: int,
    max_lookback: int,
) -> pd.DataFrame:
    """Cumulative return in [event-pre, event+post] for the most recent event."""
    rv = ret.to_numpy(dtype=float)
    ev = event.to_numpy(dtype=float)
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan)

    for c in range(cols):
        last_event = -10**9
        for t in range(rows):
            if np.isfinite(ev[t, c]) and ev[t, c] != 0.0:
                last_event = t

            if last_event < 0:
                continue

            days_since = t - last_event
            if days_since > max_lookback:
                continue

            # Window: [event - pre_window, event + post_window]
            start_idx = max(0, last_event - pre_window)
            end_idx = min(rows - 1, last_event + post_window)

            # Only compute if the window is fully in the past (no forward-looking)
            if end_idx > t:
                continue

            # Cumulative return in window
            window_rets = rv[start_idx : end_idx + 1, c]
            if not np.all(np.isfinite(window_rets)):
                continue

            cum_ret = float(np.prod(1.0 + window_rets) - 1.0)
            out[t, c] = cum_ret

    return _frame_like(ret, out)


@register_operator(
    name="event_window_return_asof",
    category="panel_gap",
    business_category="panel_gap",
    canonical="event_window_return_asof",
    source="cross_section.panel_batch1",
    backend="pandas_numpy",
    status="implemented",
)
class EventWindowReturnAsof(SeriesOperator):
    """最近事件窗口 [event-pre, event+post] 的累积收益（无前视）。

    对于每个日期 t，找到最近的事件日（不超过 max_lookback 天前），计算该事件
    前后窗口 [event-pre_window, event+post_window] 的累积收益。只在窗口完全
    位于 t 之前时才输出（无前视）。事件窗口未完成或超过 max_lookback 时输出
    NaN。causal, deterministic。
    """

    metadata = _meta(
        "event_window_return_asof",
        "最近事件窗口的累积收益（[event-pre, event+post]，无前视）。",
        ["ret", "event", "pre_window", "post_window", "max_lookback"],
        unit="return",
        cost=5,
    )

    def _calculate_series(
        self,
        ret: pd.DataFrame,
        event: pd.DataFrame,
        pre_window: int = 5,
        post_window: int = 10,
        max_lookback: int = 252,
        **_: Any,
    ) -> pd.DataFrame:
        pre_window = max(0, int(pre_window))
        post_window = max(0, int(post_window))
        max_lookback = max(1, int(max_lookback))
        return _event_window_return_asof(ret, event, pre_window, post_window, max_lookback)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_CANONICALS))


_CANONICALS.extend(
    [
        "panel_day_night_beta_gap",
        "pastor_stambaugh_beta",
        "price_delay_score",
        "report_asof",
        "event_window_return_asof",
    ]
)
_register_surface()
