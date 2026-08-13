# -*- coding: utf-8 -*-
"""Realistic A-share shaped panel audit for the 39 market-language ops.

Builds a realistic A-share daily panel (random-walk close, gap-aware open,
high/low envelopes, log-normal volume/amount, suspensions as NaN, industry
groups, forward-filled fundamentals with update-event flags) and runs each of
the 39 new operators with its natural real-field inputs.  Reports per-op:

  * coverage = fraction of output cells that are finite (non-NaN)
  * value range (min/max/mean)
  * any exception

A low coverage on realistic data is the strongest signal that an operator is
hard to use with real A-share data as-wired.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.registry import OperatorRegistry

_N = 160          # stocks (>= 2*grid^2 = 128 for default grid=8 copula)
_D = 260          # trading days
_DATE0 = pd.Timestamp("2023-01-02")

RNG = np.random.default_rng(7)


def _build_panel() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range(_DATE0, periods=_D)
    idx = pd.MultiIndex.from_product([dates, range(_N)]).sort_values()

    # close random walk with regime vol clusters
    rv = RNG.normal(0.0, 1.0, (_D, _N))
    vol = 0.02 * np.exp(0.5 * (np.sin(np.linspace(0, 40, _D))[:, None]))  # (420,1)
    ret = rv * vol
    close = 20.0 * np.exp(np.cumsum(ret, axis=0))

    # open = prev close with small gap; high/low envelopes; occasional limit touch
    prev_close = np.concatenate([close[:1], close[:-1]])
    gap = RNG.normal(0, 0.005, (_D, _N))
    open_ = prev_close * (1 + gap)
    high = np.maximum(open_, close) * (1 + np.abs(RNG.normal(0, 0.006, (_D, _N))))
    low = np.minimum(open_, close) * (1 - np.abs(RNG.normal(0, 0.006, (_D, _N))))
    high = np.maximum(np.maximum(high, open_), close)
    low = np.minimum(np.minimum(low, open_), close)

    volume = RNG.lognormal(mean=14.5, sigma=0.6, size=(_D, _N))
    amount = volume * close
    turnover_ratio = RNG.lognormal(mean=-4.0, sigma=0.5, size=(_D, _N)) * 100
    market_cap = close * RNG.lognormal(mean=20.0, sigma=0.4, size=(_D, _N))
    pe_ratio = np.clip(RNG.lognormal(mean=2.8, sigma=0.5, size=(_D, _N)), 5, 200)
    high_limit = np.round(prev_close * 1.10, 2)
    low_limit = np.round(prev_close * 0.90, 2)

    # suspensions: ~3% random NaN rows (not at the edges)
    sus_mask = RNG.random((_D, _N)) < 0.03
    for arr in (close, open_, high, low, volume, amount, turnover_ratio, market_cap, pe_ratio):
        arr = arr.reshape(-1)
        arr[sus_mask.reshape(-1)] = np.nan

    symbols = [f"{i:06d}.SH" for i in range(1, _N + 1)]
    cols = dict(zip(symbols, symbols))

    def _frame(arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(arr, index=dates, columns=symbols)

    # industry group ids (object panel, ~30 sectors, drifting membership)
    ind = np.array([[f"S{i % 30:02d}" for i in range(_N)] for _ in range(_D)], dtype=object)
    # occasionally a name changes sector (realistic, tests per-day grouping)
    for _ in range(20):
        r, c = RNG.integers(0, _D), RNG.integers(0, _N)
        ind[r, c] = f"S{RNG.integers(0, 30):02d}"
    industry = pd.DataFrame(ind, index=dates, columns=symbols)

    # fundamentals: ffill with real update days (update_event = 1 when a report lands)
    eps_raw = RNG.normal(0.3, 0.15, (_D, _N))
    fund_raw = pd.DataFrame(eps_raw, index=dates, columns=symbols)
    update_event = pd.DataFrame(
        (RNG.random((_D, _N)) < 0.015).astype(float), index=dates, columns=symbols
    )
    update_event.iloc[:5] = np.nan  # cold start
    eps = fund_raw.where(update_event > 0).ffill()

    # event flags (limit-up days) + mark (that day's |ret|)
    event = pd.DataFrame((np.abs(ret) > np.nanpercentile(np.abs(ret), 97, axis=0)).astype(float),
                         index=dates, columns=symbols)
    mark = pd.DataFrame(np.abs(ret), index=dates, columns=symbols)

    # minute returns: 48 bars per trading day (09:30..15:00 style), scaled to the
    # day's own daily return magnitude.
    rngm = np.random.default_rng(3)
    bar_times = []
    for d in range(_D):
        day = dates[d]
        for b in range(48):
            bar_times.append(day + pd.Timedelta(minutes=30 + b))
    minute_ret = pd.DataFrame(
        np.nan, index=pd.DatetimeIndex(bar_times), columns=symbols[:12],
    )
    for c in minute_ret.columns:
        for d in range(_D):
            base = ret[d, int(c.split(".")[0]) - 1]
            if not np.isfinite(base):
                continue
            chunk = np.zeros(48)
            chunk[1:] = base / np.sqrt(47) * rngm.normal(0, 1, 47)
            minute_ret.iloc[d * 48:(d + 1) * 48, minute_ret.columns.get_loc(c)] = chunk

    return {
        "close": _frame(close), "open": _frame(open_), "high": _frame(high), "low": _frame(low),
        "ret": _frame(ret), "volume": _frame(volume), "amount": _frame(amount),
        "turnover_ratio": _frame(turnover_ratio), "market_cap": _frame(market_cap),
        "pe_ratio": _frame(pe_ratio), "high_limit": _frame(high_limit), "low_limit": _frame(low_limit),
        "industry": industry, "update_event": update_event, "eps": eps,
        "event": event, "mark": mark, "minute_ret": minute_ret,
    }


# param-name -> real-field mapping (natural wiring, using the actual catalog names)
_FIELD_BY_PARAM = {
    "x": "ret", "y": "ret", "a": "ret", "b": "ret",
    "target": "ret", "source": "ret", "condition": "pe_ratio",
    "f1": "ret", "f2": "turnover_ratio", "f3": "volume",
    "scale": "ret_scale", "close": "close", "high": "high", "low": "low",
    "returns": "minute_ret",
    "group_id": "industry", "group": "industry",
    "update_event": "update_event", "event": "event", "mark": "mark",
    "eps": "eps",
}
# Directional-change semantics operate on PRICE (not returns): x = close,
# scale = a per-row vol proxy (ATR-like), so theta = threshold * ATR.
_DC_OPS = {
    "ts_dc_overshoot_ratio", "ts_dc_event_rate",
    "ts_dc_duration_asymmetry", "ts_dc_overshoot_asymmetry",
}
# Intraday profile ops take MINUTE panels (not daily ret).
_PROFILE_OPS = {"intraday_profile_surprise_energy", "intraday_profile_phase_shift"}
# Correlation / slope ops need a *different* second field, else y==x self-correlates
# to exactly 1 (or beta to exactly 1 / break to exactly 0).
_OP_FIELD_OVERRIDE = {
    "ts_modwt_band_corr": {"y": "volume"},
    "ts_expectile_beta": {"x": "volume"},
    "ts_beta_break_score": {"x": "volume"},
    "ts_roll_effective_spread": {"price": "close"},
}
_SCALAR_PARAMS = {
    "window", "recent_window", "prior_window", "quantile", "target_q", "source_q",
    "lag", "side", "target_side", "source_side", "max_lag", "tau", "theta",
    "threshold", "smooth_window", "k", "ridge", "grid", "bins", "level", "band",
    "n_updates", "event_lag", "history_window", "alpha", "steps", "sampling",
    "max_interval", "order", "n_slots", "cap", "history_days", "max_shift",
    "lookback", "min_transitions",
}


def _fetch(panel: dict, name: str, *, op: str | None = None) -> pd.DataFrame:
    if op in _DC_OPS and name == "x":
        return panel["close"]
    if op in _DC_OPS and name == "scale":
        return panel["ret"].abs().rolling(20).mean()  # ATR-like vol proxy
    if op in _PROFILE_OPS and name == "x":
        return panel["minute_ret"]
    name = _OP_FIELD_OVERRIDE.get(op, {}).get(name, name)
    target = _FIELD_BY_PARAM.get(name, name)
    if target == "ret_scale":  # ATR-like volatility scale (real users feed a vol proxy)
        return panel["ret"].abs().rolling(20).mean()
    return panel[target]


def main() -> None:
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    panel = _build_panel()
    MY_OPS = [
        "ts_quantilogram", "ts_cross_quantilogram", "ts_quantile_crossing_spectral_concentration",
        "ts_extremogram", "ts_cross_extremogram", "ts_extremal_dependence_decay",
        "ts_expectile", "ts_expectile_beta",
        "ts_dc_overshoot_ratio", "ts_dc_event_rate", "ts_dc_duration_asymmetry", "ts_dc_overshoot_asymmetry",
        "ts_feature_mode_share", "ts_feature_effective_rank", "ts_feature_subspace_rotation", "ts_beta_break_score",
        "ts_conditional_transfer_entropy", "ts_modwt_band_corr",
        "ohlc_corwin_schultz_spread", "ts_roll_effective_spread",
        "intraday_subsampled_rv_dispersion", "intraday_volatility_signature_slope",
        "intraday_realized_power_variation", "intraday_profile_surprise_energy", "intraday_profile_phase_shift",
        "cs_knn_local_linear_residual", "cs_knn_tangent_residual", "cs_knn_local_gradient_norm",
        "cs_rank_copula_mi", "cs_rank_copula_entropy",
        "group_tail_centrality", "group_tail_lead_score", "relation_diffusion_score",
        "event_mark_autocorr", "event_interval_mark_coupling",
        "update_path_efficiency", "update_acceleration", "update_surprise", "update_direction_persistence",
    ]
    rows = []
    for canon in MY_OPS:
        op = OperatorRegistry.get(canon, "pandas_numpy")
        m = op.metadata
        try:
            kwargs = {}
            for p in m.param_names:
                if p in _SCALAR_PARAMS:
                    continue  # use default
                kwargs[p] = _fetch(panel, p, op=canon)
            out = op.calculate(**kwargs)
            out = pd.DataFrame(out)
            tot = out.size
            fin = int(out.notna().to_numpy().sum())
            cov = fin / tot if tot else 0.0
            tail = out.iloc[int(len(out) * 0.8):]
            tfin = int(tail.notna().to_numpy().sum())
            tcov = tfin / tail.size if tail.size else 0.0
            vals = out.to_numpy(dtype=float)
            fv = vals[np.isfinite(vals)]
            rng = (float(fv.min()), float(fv.max()), float(fv.mean())) if fv.size else (np.nan, np.nan, np.nan)
            rows.append((canon, "ok", cov, tcov, rng))
        except Exception as e:  # noqa: BLE001
            rows.append((canon, f"ERR {type(e).__name__}: {e}", 0.0, 0.0, (np.nan, np.nan, np.nan)))

    print(f"{'op':58s} {'status':40s} {'cov%':>6s} {'tail%':>6s}  {'min':>9s} {'max':>9s} {'mean':>9s}")
    for canon, status, cov, tcov, (lo, hi, mu) in rows:
        print(f"{canon:58s} {status:40s} {100*cov:6.1f} {100*tcov:6.1f}  {lo:9.4g} {hi:9.4g} {mu:9.4g}")


if __name__ == "__main__":
    main()
