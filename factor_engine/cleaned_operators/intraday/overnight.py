# -*- coding: utf-8 -*-
"""Overnight / intraday return decomposition and gap behaviour (P1).

Daily panels in, daily panels out.  ``overnight_return = open/pre_close - 1``
and ``intraday_return = close/open - 1`` are derived inside the operator from
``close``, ``open`` and ``pre_close`` daily panels.  All kernels are causal.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator

_EPS = 1e-12
_CANONICALS: list[str] = []


def _meta(name: str, description: str, params: list[str], *, unit: str = "level") -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="return_decomposition",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "return_decomposition", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:return_decomp",
            f"unit:{unit}", "cost:2",
        ],
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _mk(name: str, description: str, params: list[str], fn, *, unit: str = "level"):
    metadata = _meta(name, description, params, unit=unit)

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"Overnight_{name}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=name,
        category="return_decomposition",
        business_category="return_decomposition",
        canonical=name,
        source="intraday.overnight",
        backend="pandas_numpy",
        status="experimental",
    )(cls)
    _CANONICALS.append(name)
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({name})
    return cls


def _decomp(close: pd.DataFrame, open_px: pd.DataFrame, pre_close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overnight = _safe_ret(open_px, pre_close)
    intraday = _safe_ret(close, open_px)
    return overnight, intraday


def _safe_ret(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a / b.replace(0, np.nan) - 1.0


def _rolling_cov(a: pd.DataFrame, b: pd.DataFrame, window: int) -> pd.DataFrame:
    return a.rolling(int(window), min_periods=5).cov(b)


def _ts_overnight_intraday_cov(close, open_px, pre_close, window=60):
    o, i = _decomp(close, open_px, pre_close)
    return _rolling_cov(o, i, int(window))


_mk(
    "ts_overnight_intraday_cov",
    "隔夜与日内收益滚动协方差。",
    ["close", "open", "pre_close", "window"],
    _ts_overnight_intraday_cov,
)


def _ts_overnight_intraday_spread(close, open_px, pre_close, window=60):
    o, i = _decomp(close, open_px, pre_close)
    w = int(window)
    return o.rolling(w).mean() - i.rolling(w).mean()


_mk(
    "ts_overnight_intraday_spread",
    "隔夜收益滚动均值 - 日内收益滚动均值。",
    ["close", "open", "pre_close", "window"],
    _ts_overnight_intraday_spread,
)


def _ts_overnight_intraday_sign_agreement(close, open_px, pre_close, window=60):
    o, i = _decomp(close, open_px, pre_close)
    agree = (np.sign(o) == np.sign(i)).astype(float)
    return agree.rolling(int(window)).mean()


_mk(
    "ts_overnight_intraday_sign_agreement",
    "隔夜与日内收益同号比例（滚动）。",
    ["close", "open", "pre_close", "window"],
    _ts_overnight_intraday_sign_agreement,
)


def _ts_gap_reversion_ratio(close, open_px, pre_close, threshold=0.01):
    o, i = _decomp(close, open_px, pre_close)
    ratio = -i / o.replace(0, np.nan).abs()
    return ratio.where(o.abs() > float(threshold))


_mk(
    "ts_gap_reversion_ratio",
    "-日内收益/隔夜收益，仅隔夜跳空超过阈值时定义。",
    ["close", "open", "pre_close", "threshold"],
    _ts_gap_reversion_ratio,
)


def _ts_gap_fill_ratio(close, open_px, pre_close, window=60):
    o, i = _decomp(close, open_px, pre_close)
    # gap filled on day t when close crosses pre_close
    up_gap = (o > 0).astype(float)
    filled = ((close >= pre_close) & (o > 0) & (i >= 0)) | ((close <= pre_close) & (o < 0) & (i <= 0))
    filled_f = filled.astype(float).fillna(0.0)
    return filled_f.rolling(int(window)).mean() * up_gap.rolling(int(window)).mean().replace(0, np.nan)


_mk(
    "ts_gap_fill_ratio",
    "窗口内跳空缺口被当日完全回补的比例。",
    ["close", "open", "pre_close", "window"],
    _ts_gap_fill_ratio,
)


def _ts_gap_survival_duration(close, open_px, pre_close):
    o, _i = _decomp(close, open_px, pre_close)
    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)
    for col in close.columns:
        ov = o[col].to_numpy(dtype=float)
        cv = close[col].to_numpy(dtype=float)
        pv = pre_close[col].to_numpy(dtype=float)
        arr = np.full(len(cv), np.nan, dtype=float)
        gap_age = np.nan
        gap_dir = 0.0
        for t in range(len(cv)):
            if not (np.isfinite(ov[t]) and np.isfinite(cv[t]) and np.isfinite(pv[t])):
                arr[t] = gap_age
                continue
            if abs(ov[t]) > _EPS:
                gap_dir = 1.0 if ov[t] > 0 else -1.0
                gap_age = 0.0
            else:
                if gap_age is not None and np.isfinite(gap_age):
                    gap_age = gap_age + 1.0
                # filled: close back across pre_close from the gap side
                if gap_dir > 0 and cv[t] <= pv[t]:
                    gap_dir, gap_age = 0.0, np.nan
                elif gap_dir < 0 and cv[t] >= pv[t]:
                    gap_dir, gap_age = 0.0, np.nan
            arr[t] = gap_age
        out[col] = arr
    return out


_mk(
    "ts_gap_survival_duration",
    "当前未回补跳空缺口持续的交易日数。",
    ["close", "open", "pre_close"],
    _ts_gap_survival_duration,
    unit="count",
)


def _ts_opening_mispricing_score(close, open_px, pre_close, window=60):
    o, i = _decomp(close, open_px, pre_close)
    ov = o.to_numpy(dtype=float)
    iv = i.to_numpy(dtype=float)
    rows, cols = ov.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    mp = max(6, w // 5)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - w + 1)
            a, b = ov[start : row + 1, col], iv[start : row + 1, col]
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() < mp or np.var(a[valid]) <= _EPS:
                continue
            beta = float(np.cov(a[valid], b[valid])[0, 1] / np.var(a[valid]))
            expected = beta * ov[row, col]
            out[row, col] = float(ov[row, col] - expected)
    return _frame_like(close, out)


_mk(
    "ts_opening_mispricing_score",
    "隔夜收益 - 历史滚动回归的预期日内响应。",
    ["close", "open", "pre_close", "window"],
    _ts_opening_mispricing_score,
)
