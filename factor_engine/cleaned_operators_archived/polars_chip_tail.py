# -*- coding: utf-8 -*-
"""Genuine Polars backends for the turnover-survival / weighted-tail / CPT families.

These kernels are per-instrument sequential (survival weights are a suffix
product over the trailing window; CPT ranks and probability-weights a rolling
sample), which plain ``pl.Expr`` window functions cannot express.  The backends
therefore run the *shared* numpy kernel per instrument column via Polars-native
column UDFs (``Series.to_numpy()`` → kernel → ``pl.Series``) — no pandas
DataFrame round-trip, no per-row Python loop in the data plumbing, and exact
pandas-reference parity by construction.  Production admission stays on the
certified ``pandas_numpy`` reference; these slots accelerate the Polars runtimes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.turnover_survival import (
    _column_stats,
    _default_min_periods,
)
from cleaned_operators.weighted_tail import (
    _stratified_stratum_mean,
    _weighted_es_tail,
)
from cleaned_operators.prospect_theory import (
    _MIN_COVERAGE,
    _PRESETS,
    _column_cpt,
)

_SKIP_PANEL = frozenset({"date", "stock_code"})
_EPS = 1e-12


def _cols(fr: pl.DataFrame) -> list[str]:
    return [c for c in fr.columns if c not in _SKIP_PANEL]


def _col(fr: pl.DataFrame, name: str) -> np.ndarray:
    return fr.select(name).to_series().to_numpy().astype(np.float64)


def _rebuild(base: pl.DataFrame, data: dict[str, np.ndarray]) -> pl.DataFrame:
    return base.with_columns(
        [pl.Series(name=c, values=np.asarray(v, dtype=np.float64)) for c, v in data.items()]
    )


def _mk(canonical: str, description: str, params: list[str], fn):
    metadata = OperatorMetadata(
        name=canonical,
        category="time_series_chip_cost",
        description=description,
        param_names=params,
        return_type="series",
        tags=["polars", "daily", "native_udf", "typed_v2"],
    )

    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)

    cls = type(
        f"PolarsChipTail_{canonical}",
        (SeriesOperator,),
        {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
    )
    register_operator(
        name=canonical,
        category="time_series_chip_cost",
        business_category="time_series_chip_cost",
        canonical=canonical,
        source="polars_chip_tail",
    )(cls)
    return cls


# ---------------------------------------------------------------------------
# Turnover survival family (shared kernel).
# ---------------------------------------------------------------------------

def _survival_family(price, turnover, window, band, q_high, q_low, key):
    w = max(2, int(window))
    band = max(0.0, float(band))
    qh = float(q_high)
    ql = float(q_low)
    if not 0.0 < ql < qh < 1.0:
        raise ValueError("need 0 < q_low < q_high < 1")
    mp = _default_min_periods(w)
    cols = _cols(price)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        p = _col(price, c)
        t = _col(turnover, c)
        stats = _column_stats(p, t, w, mp, band, qh, ql)
        out[c] = stats[key]
    return _rebuild(price, out)


for _name, _desc, _key in (
    ("ts_turnover_reference_price", "换手存活加权平均持仓成本(仅用 t-1 及以前)（Polars）。", "ref"),
    ("ts_turnover_cost_dispersion", "换手存活筹码的对数成本离散度（Polars）。", "disp"),
    ("ts_turnover_profit_share", "获利筹码占比（Polars）。", "profit"),
    ("ts_turnover_holding_age", "存活筹码加权平均持仓天数（Polars）。", "age"),
):
    _mk(
        _name, _desc, ["price", "turnover", "window"],
        lambda price, turnover, window=60, _key=_key: _survival_family(
            price, turnover, window, 0.05, 0.75, 0.25, _key
        ),
    )


_mk(
    "ts_turnover_near_cost_mass",
    "当前价格附近 ±band 内的存活筹码密度（Polars）。",
    ["price", "turnover", "window", "band"],
    lambda price, turnover, window=60, band=0.05: _survival_family(
        price, turnover, window, band, 0.75, 0.25, "near"
    ),
)
_mk(
    "ts_turnover_cost_quantile_distance",
    "存活筹码成本分位间距 (Q_high-Q_low)/RP（Polars）。",
    ["price", "turnover", "window", "q_high", "q_low"],
    lambda price, turnover, window=60, q_high=0.75, q_low=0.25: _survival_family(
        price, turnover, window, 0.05, q_high, q_low, "qdist"
    ),
)
_mk(
    "ts_turnover_cost_entropy",
    "筹码成本分布熵（Polars）。",
    ["price", "turnover", "window"],
    lambda price, turnover, window=60: _survival_family(price, turnover, window, 0.05, 0.75, 0.25, "cost_ent"),
)
_mk(
    "ts_turnover_cost_mode_distance",
    "最大成本峰相对当前价距离（Polars）。",
    ["price", "turnover", "window"],
    lambda price, turnover, window=60: _survival_family(price, turnover, window, 0.05, 0.75, 0.25, "mode_d"),
)
_mk(
    "ts_turnover_cost_skew",
    "筹码成本加权偏度（Polars）。",
    ["price", "turnover", "window"],
    lambda price, turnover, window=60: _survival_family(price, turnover, window, 0.05, 0.75, 0.25, "cost_sk"),
)
_mk(
    "ts_turnover_age_dispersion",
    "持仓年龄加权离散度（Polars）。",
    ["price", "turnover", "window"],
    lambda price, turnover, window=60: _survival_family(price, turnover, window, 0.05, 0.75, 0.25, "age_disp"),
)
_mk(
    "ts_turnover_old_mass",
    "窗口外残留浮筹质量 M_old（诊断输出，Polars）。",
    ["price", "turnover", "window"],
    lambda price, turnover, window=60: _survival_family(price, turnover, window, 0.05, 0.75, 0.25, "old_mass"),
)
_mk(
    "ts_turnover_cost_entropy_vol_scaled",
    "波动率标度筹码成本熵（log(P/RP)/disp 分箱，Polars）。",
    ["price", "turnover", "window"],
    lambda price, turnover, window=60: _survival_family(price, turnover, window, 0.05, 0.75, 0.25, "cost_ent_v"),
)


# ---------------------------------------------------------------------------
# Weighted tail / stratified family (shared kernels).
# ---------------------------------------------------------------------------

def _stratified_mean_spread(target, sorter, window, quantile, min_periods):
    w = max(2, int(window))
    q = float(quantile)
    if not 0.0 < q <= 0.5:
        raise ValueError("quantile must be in (0, 0.5] so the top and bottom strata are disjoint")
    mp = int(min_periods) if min_periods is not None else max(5, w // 4)
    mp = max(2, mp)
    cols = _cols(target)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        x = _col(target, c)
        s = _col(sorter, c)
        rows = x.shape[0]
        res = np.full(rows, np.nan)
        for t in range(rows):
            lo = max(0, t - w + 1)
            xv = x[lo : t + 1]
            sv = s[lo : t + 1]
            finite = np.isfinite(xv) & np.isfinite(sv)
            if int(finite.sum()) < mp:
                continue
            xs = xv[finite]
            ss = sv[finite]
            order = np.argsort(ss, kind="stable")
            xs = xs[order]
            ss = ss[order]
            k = max(1, min(int(round(q * xs.size)), xs.size - 1))
            top = _stratified_stratum_mean(xs, ss, k, top=True)
            bot = _stratified_stratum_mean(xs, ss, k, top=False)
            if not np.isfinite(top) or not np.isfinite(bot):
                continue
            res[t] = float(top - bot)
        out[c] = res
    return _rebuild(target, out)


_mk(
    "ts_stratified_mean_spread",
    "按 sorter 分层的 target 高低尾均值差（Polars）。",
    ["target", "sorter", "window", "quantile", "min_periods"],
    lambda target, sorter, window=60, quantile=0.2, min_periods=None: _stratified_mean_spread(
        target, sorter, window, quantile, min_periods
    ),
)


def _pair_window_kernel(x_frame, w_frame, window, target, min_periods, kind):
    w = max(2, int(window))
    mp = int(min_periods) if min_periods is not None else 2
    mp = max(2, mp)
    cols = _cols(x_frame)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        x = _col(x_frame, c)
        wt = _col(w_frame, c)
        rows = x.shape[0]
        res = np.full(rows, np.nan)
        for t in range(rows):
            lo = max(0, t - w + 1)
            xv = x[lo : t + 1]
            wv = wt[lo : t + 1]
            finite = np.isfinite(xv) & np.isfinite(wv)
            xv = xv[finite]
            wv = wv[finite]
            if xv.size < mp:
                continue
            total = float(wv.sum())
            if total <= _EPS:
                continue
            if kind == "semivar":
                # R3-113: true semivariance keeps the square (NO sqrt).
                below = np.maximum(target - xv, 0.0)
                res[t] = float(np.sum(wv * below * below) / total)
            elif kind == "downside":
                # R3-113: sqrt'd variant = weighted downside deviation.
                below = np.maximum(target - xv, 0.0)
                res[t] = float(np.sqrt(np.sum(wv * below * below) / total))
            elif kind == "drawdown":
                pos = xv > 0.0
                if int(pos.sum()) < 2:
                    continue
                p = xv[pos]
                wpos = wv[pos]
                ttl = float(wpos.sum())
                if ttl <= _EPS:
                    continue
                running_max = np.maximum.accumulate(p)
                dd = np.maximum(0.0, 1.0 - p / running_max)
                res[t] = float(np.sum(wpos * dd) / ttl)
        out[c] = res
    return _rebuild(x_frame, out)


_mk(
    "ts_weighted_semivariance",
    "加权下半方差 sum w*max(target-x,0)^2 / sum w（无 sqrt，Polars）。",
    ["x", "weight", "window", "target", "min_periods"],
    lambda x, weight, window=20, target=0.0, min_periods=None: _pair_window_kernel(
        x, weight, window, float(target), min_periods, "semivar"
    ),
)
_mk(
    "ts_weighted_downside_deviation",
    "加权下行偏离 sqrt(sum w*max(target-x,0)^2 / sum w)（Polars）。",
    ["x", "weight", "window", "target", "min_periods"],
    lambda x, weight, window=20, target=0.0, min_periods=None: _pair_window_kernel(
        x, weight, window, float(target), min_periods, "downside"
    ),
)
_mk(
    "ts_weighted_drawdown_area",
    "加权回撤深度面积(运行峰值起) sum(w*DD)/sum(w)（Polars）。",
    ["x", "weight", "window"],
    lambda x, weight, window=60: _pair_window_kernel(
        x, weight, window, 0.0, 2, "drawdown"
    ),
)


def _weighted_es(x_frame, w_frame, window, quantile, side, min_tail_count):
    w = max(2, int(window))
    q = float(quantile)
    if not 0.0 < q <= 0.5:
        raise ValueError("quantile must be in (0, 0.5] for expected-shortfall tail semantics")
    kind = str(side).lower()
    if kind not in {"lower", "upper"}:
        raise ValueError("side must be 'lower' or 'upper'")
    if min_tail_count is not None:
        min_tail = max(2, int(min_tail_count))
    else:
        min_tail = max(2, 3)
    cols = _cols(x_frame)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        x = _col(x_frame, c)
        wt = _col(w_frame, c)
        rows = x.shape[0]
        res = np.full(rows, np.nan)
        for t in range(rows):
            lo = max(0, t - w + 1)
            xv = x[lo : t + 1]
            wv = wt[lo : t + 1]
            finite = np.isfinite(xv) & np.isfinite(wv)
            xv = xv[finite]
            wv = wv[finite]
            if xv.size < max(min_tail, 3):
                continue
            res[t] = _weighted_es_tail(xv, wv, q, kind, min_tail)
        out[c] = res
    return _rebuild(x_frame, out)


_mk(
    "ts_weighted_expected_shortfall",
    "加权期望损失: 加权分位数之外尾部的加权均值（Polars）。",
    ["x", "weight", "window", "quantile", "side", "min_tail_count"],
    lambda x, weight, window=60, quantile=0.05, side="lower", min_tail_count=None: _weighted_es(
        x, weight, window, quantile, side, min_tail_count
    ),
)


# ---------------------------------------------------------------------------
# CPT value (shared kernel).
# ---------------------------------------------------------------------------

def _cpt(returns, window, preset, min_coverage=_MIN_COVERAGE):
    w = max(2, int(window))
    key = str(preset).lower()
    if key not in _PRESETS:
        raise ValueError(f"unknown CPT preset: {preset!r}")
    params = _PRESETS[key]
    min_periods = max(5, w // 10)
    cols = _cols(returns)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        out[c] = _column_cpt(_col(returns, c), w, params, min_periods, min_coverage=float(min_coverage))
    return _rebuild(returns, out)


_mk(
    "ts_cpt_value",
    "累积前景理论价值(固定 bmw2016 参数集)（Polars）。",
    ["returns", "window", "preset", "min_coverage"],
    lambda returns, window=60, preset="bmw2016", min_coverage=_MIN_COVERAGE: _cpt(
        returns, window, preset, min_coverage
    ),
)
