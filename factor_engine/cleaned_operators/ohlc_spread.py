# -*- coding: utf-8 -*-
"""OHLC-based microstructure spreads (2026-08-08 Gemini V2 round).

* ``ts_edge_effective_spread`` — a *faithful port* of the official EDGE
  estimator (Ardia, Guidotti & Kroencke 2024, JFE 161:103916).  The kernel is
  translated line-for-line from the reference Python implementation
  (``bidask.edge.edge``), so a golden test against the official package passes
  at float precision on any valid OHLC input.  Prices are log-transformed
  inside; rows where ``O, H, L, C <= 0`` or where ``H < max(O,C)`` or
  ``L > min(O,C)`` (impossible OHLC) are masked to NaN and skipped by the
  reference algorithm's pairwise NaN handling.  A value of ``0.01`` = 1%
  spread.
* ``ts_abdi_ranaldo_spread`` — Abdi & Ranaldo (2017, RFS 30:4437) effective
  spread.  PIT-safe: the estimator needs ``(c_s, η_s, η_{s+1})`` (η = log
  mid-range), so ``output[T]`` uses only completed pairs ``s ≤ T-1`` — the
  last pair is ``(c_{T-1}, η_{T-1}, η_T)`` with day-*T*'s high/low (known at
  day-*T* close) — never ``(c_T, η_T, η_{T+1})``.  Output
  ``sqrt(4·mean[...])`` when the squared-spread estimate is non-negative;
  a *negative* squared spread (an unreliable variance estimate, NOT a zero
  spread) fails closed to NaN (P0-M-72 — the old ``max(..., 0)`` silently
  clipped it to 0).
* ``ts_pastor_stambaugh_liquidity_gamma`` — Pastor & Stambaugh (2003, JPE)
  order-flow-reversal gamma.  ``flow_t = sign(r_t^e)·amount_t / flow_scale``
  (P0-M-73: the unit convention is an explicit ``flow_scale`` parameter, default
  1e6 = "per million currency" — never a hidden kernel constant), regressed as
  ``r_{t+1}^e = α + β r_t^e + γ flow_t + ε`` over a trailing window of
  *completed* pairs (``s+1 ≤ T``).  Output ``γ``.
* ``ts_edge_effective_spread`` — valid-pair coverage gates (P0-M-71):
  ``min_valid_pairs`` (default 2) and ``min_valid_ratio`` (default 0 = disabled)
  fail the window closed to NaN when too few OHLC pairs survive the estimator's
  internal valid pattern.

All operators are strict-PIT, deterministic, and NaN fail-closed.  The unit
conventions (return in decimal, price as continuous price) are assumed to be
already normalised by the data-access layer — no operator hardcodes a
market-specific 1/10000 style conversion.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import ParamRole, ParamSpec, RelationalParamSpec
from cleaned_operators.gemini_v2_common import (
    frame_like,
    register_dual,
    trailing_contiguous_multi,
    union_extended,
)

_EDGE_SPECS = {
    "window": ParamSpec(dtype=int, min=3),
    # P0-M-71: valid-pair coverage gates.  Even with every row finite, many OHLC
    # pairs can internally violate the EDGE valid pattern (impossible OHLC
    # geometry, degenerate ``(h==l) and (l==c1)`` pairs) and be masked out.
    # ``min_valid_pairs`` / ``min_valid_ratio`` are opt-in fail-closed output
    # gates — pure estimator-resolution knobs, never search dimensions.
    "min_valid_pairs": ParamSpec(
        dtype=int, min=2, default=2, searchable=False, param_role=ParamRole.NUMERICAL,
    ),
    "min_valid_ratio": ParamSpec(
        dtype=float, min=0.0, max=1.0, default=0.0, searchable=False,
        param_role=ParamRole.NUMERICAL,
    ),
}
_ABDI_SPECS = {
    "window": ParamSpec(dtype=int, min=3),
}
_PS_SPECS = {
    "window": ParamSpec(dtype=int, min=4),
    "min_periods": ParamSpec(dtype=int, min=3),
    # P0-M-73: the order-flow is ``flow_t = sign(r_t^e)·amount_t / flow_scale``.
    # The old kernel hard-coded ``/1e6`` — a hidden RMB-元-vs-千元-vs-百万元 unit
    # convention that silently rescales the gamma.  The unit is now explicit as a
    # parameter: the consumer supplies the scale that turns their ``amount``
    # (declared ``money_local``) into the flow unit, and the declared output unit
    # is ``return_per_{flow_scale}_currency``.  Default 1e6 preserves the
    # reference Pastor-Stambaugh "per million" convention; never a search
    # dimension (a pure unit conversion).
    "flow_scale": ParamSpec(
        dtype=float, min=1e-9, default=1e6, searchable=False,
        param_role=ParamRole.NUMERICAL,
    ),
}
# R6-164: the regression has N-1 completed pairs (response e_{s+1} for
# s = 0..N-2), so min_periods == window can never be satisfied.  Declared as a
# relational constraint: min_periods <= window - 1.
_PS_RELATIONAL_SPECS = [
    RelationalParamSpec(
        "min_periods <= window - 1",
        "min_periods must be <= window - 1 (the regression uses N-1 completed "
        "pairs: min_periods={min_periods}, window={window})",
    )
]


# ---------------------------------------------------------------------------
# EDGE — faithful port of the official estimator
# ---------------------------------------------------------------------------
def _edge_window(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    min_valid_pairs: int = 2,
    min_valid_ratio: float = 0.0,
) -> float:
    """Reference ``edge()`` translated 1:1 from ``bidask.edge.edge``.

    Operates on the aligned trailing window (already finite).  Rows violating
    the OHLC geometry or with non-positive prices are masked to NaN; the
    reference algorithm's ``nanmean`` handling then skips the affected pairs.

    P0-M-71 valid-pair coverage gates: ``tau`` marks each adjacent pair as
    usable (non-degenerate, both rows valid).  ``n_valid_pairs`` counts those
    and ``valid_ratio = n_valid_pairs / (nobs - 1)`` is the coverage of the
    window.  ``min_valid_pairs`` (default 2 = the reference algorithm's own
    floor) and ``min_valid_ratio`` (default 0 = disabled) fail the window closed
    to NaN when coverage is too low — a few surviving pairs must not be read as
    a reliable spread.
    """
    nobs = int(open_.shape[0])
    if nobs < 3:
        return np.nan
    # Positivity is checked on the *original* prices, never on the log prices:
    # a stock trading below 1.0 has a negative log-price and would be wrongly
    # masked out by ``log(o) > 0`` (P0-03 review).
    raw_valid = (np.asarray(open_, dtype=float) > 0.0) & (np.asarray(high, dtype=float) > 0.0) \
        & (np.asarray(low, dtype=float) > 0.0) & (np.asarray(close, dtype=float) > 0.0)
    o = np.log(np.asarray(open_, dtype=float))
    h = np.log(np.asarray(high, dtype=float))
    l = np.log(np.asarray(low, dtype=float))
    c = np.log(np.asarray(close, dtype=float))
    m = (h + l) / 2.0

    # row-validity mask (impossible OHLC -> NaN, per the operator contract)
    valid = raw_valid.copy()
    for i in range(nobs):
        if not raw_valid[i]:
            continue
        if h[i] < max(o[i], c[i]) or l[i] > min(o[i], c[i]):
            valid[i] = False
    if not valid.all():
        o = np.where(valid, o, np.nan)
        h = np.where(valid, h, np.nan)
        l = np.where(valid, l, np.nan)
        c = np.where(valid, c, np.nan)
        m = (h + l) / 2.0

    h1, l1, c1, m1 = h[:-1], l[:-1], c[:-1], m[:-1]
    o, h, l, c, m = o[1:], h[1:], l[1:], c[1:], m[1:]

    r1 = m - o
    r2 = o - m1
    r3 = m - c1
    r4 = c1 - m1
    r5 = o - c1

    tau = np.where(np.isnan(h) | np.isnan(l) | np.isnan(c1), np.nan, (h != l) | (l != c1))
    po1 = tau * np.where(np.isnan(o) | np.isnan(h), np.nan, o != h)
    po2 = tau * np.where(np.isnan(o) | np.isnan(l), np.nan, o != l)
    pc1 = tau * np.where(np.isnan(c1) | np.isnan(h1), np.nan, c1 != h1)
    pc2 = tau * np.where(np.isnan(c1) | np.isnan(l1), np.nan, c1 != l1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pt = np.nanmean(tau)
        po = np.nanmean(po1) + np.nanmean(po2)
        pc = np.nanmean(pc1) + np.nanmean(pc2)
        if np.nansum(tau) < 2 or po == 0 or pc == 0:
            return np.nan
        # P0-M-71: explicit valid-pair coverage gates (defaults preserve the
        # reference behaviour: min_valid_pairs=2, min_valid_ratio=0 disables).
        n_valid_pairs = int(np.nansum(tau))
        n_pairs_total = nobs - 1
        valid_ratio = n_valid_pairs / n_pairs_total if n_pairs_total > 0 else 0.0
        if n_valid_pairs < int(min_valid_pairs) or valid_ratio < float(min_valid_ratio):
            return np.nan
        d1 = r1 - np.nanmean(r1) / pt * tau
        d3 = r3 - np.nanmean(r3) / pt * tau
        d5 = r5 - np.nanmean(r5) / pt * tau
        x1 = -4.0 / po * d1 * r2 + -4.0 / pc * d3 * r4
        x2 = -4.0 / po * d1 * r5 + -4.0 / pc * d5 * r4
        e1 = np.nanmean(x1)
        e2 = np.nanmean(x2)
        v1 = np.nanmean(x1**2) - e1**2
        v2 = np.nanmean(x2**2) - e2**2

    vt = v1 + v2
    s2 = (v2 * e1 + v1 * e2) / vt if vt > 0 else (e1 + e2) / 2.0
    return float(np.sqrt(np.abs(s2)))


def _edge_series(
    open_2d: np.ndarray,
    high_2d: np.ndarray,
    low_2d: np.ndarray,
    close_2d: np.ndarray,
    window: int,
    min_valid_pairs: int = 2,
    min_valid_ratio: float = 0.0,
) -> np.ndarray:
    rows, cols = open_2d.shape
    w = max(3, int(window))
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            lo = max(0, r - w + 1)
            run = trailing_contiguous_multi(
                close_2d[lo : r + 1, c],
                high_2d[lo : r + 1, c],
                low_2d[lo : r + 1, c],
                open_2d[lo : r + 1, c],
            )
            if run is None:
                continue
            if not (np.all(np.isfinite(run[0])) and np.all(np.isfinite(run[1]))
                    and np.all(np.isfinite(run[2])) and np.all(np.isfinite(run[3]))):
                continue
            # R16-153: FULL-WINDOW contiguous — a gap-shrunken short run
            # (N < window) must NOT emit as the same estimator; an adaptive-N
            # variant is a different canonical.
            if run[0].shape[0] != min(w, r + 1):
                continue
            if run[0].shape[0] < 3:
                continue
            if np.any(run[0] <= 0.0) or np.any(run[1] <= 0.0) or np.any(run[2] <= 0.0) or np.any(run[3] <= 0.0):
                continue
            val = _edge_window(run[3], run[1], run[2], run[0], min_valid_pairs, min_valid_ratio)
            if np.isfinite(val):
                out[r, c] = val
    return out


def _ts_edge_effective_spread(
    open: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 20,
    min_valid_pairs: int = 2,
    min_valid_ratio: float = 0.0,
) -> pd.DataFrame:
    if int(window) < 3:
        raise ValueError("ts_edge_effective_spread requires window >= 3")
    out = _edge_series(
        open.to_numpy(dtype=float), high.to_numpy(dtype=float),
        low.to_numpy(dtype=float), close.to_numpy(dtype=float), int(window),
        int(min_valid_pairs), float(min_valid_ratio),
    )
    return frame_like(close, out)


# ---------------------------------------------------------------------------
# Abdi-Ranaldo (PIT-safe rolling form)
# ---------------------------------------------------------------------------
def _abdi_ranaldo_window(c: np.ndarray, eta: np.ndarray, window: int) -> float:
    n = c.shape[0]
    if n < 3:
        return np.nan
    # terms over s = 0..n-2: (c_s - eta_s)*(c_s - eta_{s+1})
    terms = (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    if terms.shape[0] < 2:
        return np.nan
    terms = terms[-window:] if window < terms.shape[0] else terms
    mean_term = float(np.nanmean(terms))
    if not np.isfinite(mean_term):
        return np.nan
    s2 = 4.0 * mean_term
    # P0-M-72: a negative squared-spread estimate (s2 < 0) is an unreliable
    # variance estimate, NOT a "true spread of 0".  The old ``max(s2, 0.0)``
    # silently clipped it to 0 — a read as "no spread" from a broken estimate.
    # Fail closed to NaN instead.  Exact 0 (a genuinely constant price path)
    # remains a legitimate zero spread.
    if s2 < 0.0:
        return np.nan
    return float(np.sqrt(s2))


def _abdi_ranaldo_series(close_2d: np.ndarray, high_2d: np.ndarray, low_2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = close_2d.shape
    w = max(3, int(window))
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            lo = max(0, r - w + 1)
            run = trailing_contiguous_multi(
                close_2d[lo : r + 1, c], high_2d[lo : r + 1, c], low_2d[lo : r + 1, c]
            )
            if run is None:
                continue
            # R16-153: FULL-WINDOW contiguous (a gap-shrunken short run must not
            # emit as the same estimator).
            if run[0].shape[0] != min(w, r + 1):
                continue
            if np.any(run[0] <= 0.0) or np.any(run[1] <= 0.0) or np.any(run[2] <= 0.0):
                continue
            cc = np.log(run[0])
            eta = (np.log(run[1]) + np.log(run[2])) / 2.0
            if not (np.all(np.isfinite(cc)) and np.all(np.isfinite(eta))):
                continue
            val = _abdi_ranaldo_window(cc, eta, w)
            if np.isfinite(val):
                out[r, c] = val
    return out


def _ts_abdi_ranaldo_spread(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    window: int = 20,
    correction: str = "monthly",  # P1-30: fixed — the only legal value; not searchable
) -> pd.DataFrame:
    if int(window) < 3:
        raise ValueError("ts_abdi_ranaldo_spread requires window >= 3")
    if str(correction) != "monthly":
        raise ValueError("ts_abdi_ranaldo_spread correction must be 'monthly'")
    out = _abdi_ranaldo_series(
        close.to_numpy(dtype=float), high.to_numpy(dtype=float), low.to_numpy(dtype=float), int(window)
    )
    return frame_like(close, out)


# ---------------------------------------------------------------------------
# Pastor-Stambaugh order-flow reversal gamma
# ---------------------------------------------------------------------------
def _ps_gamma_window(e: np.ndarray, flow: np.ndarray, min_periods: int) -> float:
    n = e.shape[0]
    if n < 4:
        return np.nan
    # completed pairs: response e_{s+1} for s = 0..n-2
    pred_e = e[:-1]
    pred_f = flow[:-1]
    resp = e[1:]
    n_pairs = pred_e.shape[0]
    if n_pairs < max(3, int(min_periods)):
        return np.nan
    design = np.column_stack((np.ones(n_pairs), pred_e, pred_f))
    if np.linalg.matrix_rank(design) < 3:
        return np.nan
    try:
        beta, *_ = np.linalg.lstsq(design, resp, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    gamma = float(beta[2])
    return gamma if np.isfinite(gamma) else np.nan


def _ps_series(
    ret_2d: np.ndarray,
    bench_2d: np.ndarray,
    amount_2d: np.ndarray,
    window: int,
    min_periods: int,
    flow_scale: float = 1e6,
) -> np.ndarray:
    rows, cols = ret_2d.shape
    w = max(4, int(window))
    mp = max(3, int(min_periods))
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            lo = max(0, r - w + 1)
            run = trailing_contiguous_multi(
                ret_2d[lo : r + 1, c], bench_2d[lo : r + 1, c], amount_2d[lo : r + 1, c]
            )
            if run is None:
                continue
            # R16-153: FULL-WINDOW contiguous (a gap-shrunken short run must not
            # emit as the same estimator).
            if run[0].shape[0] != min(w, r + 1):
                continue
            if run[0].shape[0] < 4:
                continue
            e = run[0] - run[1]
            # P0-M-73: flow unit is explicit via ``flow_scale`` (default 1e6 =
            # the reference "per million" convention).  The kernel never
            # hard-codes a currency-unit convention; the consumer passes the
            # scale that normalises their ``amount`` (money_local) to the flow
            # unit, so RMB 元 vs 千元 vs 百万元 inputs map to the same factor.
            flow = np.sign(e) * run[2] / float(flow_scale)
            val = _ps_gamma_window(e, flow, mp)
            if np.isfinite(val):
                out[r, c] = val
    return out


def _ts_pastor_stambaugh_liquidity_gamma(
    ret: pd.DataFrame,
    benchmark_ret: pd.DataFrame,
    amount: pd.DataFrame,
    window: int = 60,
    min_periods: int = 20,
    flow_scale: float = 1e6,
) -> pd.DataFrame:
    if int(window) < 4:
        raise ValueError("ts_pastor_stambaugh_liquidity_gamma requires window >= 4")
    out = _ps_series(
        ret.to_numpy(dtype=float), benchmark_ret.to_numpy(dtype=float),
        amount.to_numpy(dtype=float), int(window), int(min_periods), float(flow_scale),
    )
    return frame_like(ret, out)


_SPECS: dict[str, dict[str, Any]] = {
    "ts_edge_effective_spread": {
        "fn": _ts_edge_effective_spread,
        "params": ["open", "high", "low", "close", "window", "min_valid_pairs", "min_valid_ratio"],
        "category": "market_microstructure",
        "domain": "liquidity",
        "unit": "ratio",
        "cost": 4,
        "tags_extra": [],
        "input_units": {"open": "continuous_price", "high": "continuous_price",
                        "low": "continuous_price", "close": "continuous_price"},
        "output_unit": "ratio",
        "param_specs": _EDGE_SPECS,
    },
    "ts_abdi_ranaldo_spread": {
        "fn": _ts_abdi_ranaldo_spread,
        # P1-30: ``correction`` is fixed to ``"monthly"`` (the only legal value)
        # and is not exposed as a search parameter — the kernel keeps an internal
        # ``correction`` default but the signature only advertises the real inputs.
        "params": ["close", "high", "low", "window"],
        "category": "market_microstructure",
        "domain": "liquidity",
        "unit": "ratio",
        "cost": 3,
        "tags_extra": [],
        "input_units": {"close": "continuous_price", "high": "continuous_price",
                        "low": "continuous_price"},
        "output_unit": "ratio",
        "param_specs": _ABDI_SPECS,
    },
    "ts_pastor_stambaugh_liquidity_gamma": {
        "fn": _ts_pastor_stambaugh_liquidity_gamma,
        "params": ["ret", "benchmark_ret", "amount", "window", "min_periods", "flow_scale"],
        "category": "market_microstructure",
        "domain": "liquidity",
        # R6-163 + P0-M-73: flow = sign(e)·amount/flow_scale, so
        # [gamma] = return / (flow_scale currency units) — NOT a dimensionless
        # ratio.  Declared honestly; ``flow_scale`` makes the unit convention
        # explicit (default 1e6 = "per million currency") instead of a hidden
        # hard-coded 1e6 in the kernel.
        "unit": "return_per_million_currency",
        "cost": 4,
        "tags_extra": [],
        "input_units": {"ret": "return_decimal", "benchmark_ret": "return_decimal",
                        "amount": "money_local"},
        "output_unit": "return_per_million_currency",
        "param_specs": _PS_SPECS,
        "relational_specs": _PS_RELATIONAL_SPECS,
    },
}


def _register() -> None:
    for canonical, spec in _SPECS.items():
        register_dual(
            canonical,
            spec["fn"],
            spec["params"],
            category=spec["category"],
            domain=spec["domain"],
            unit=spec["unit"],
            cost=spec["cost"],
            source="ohlc_spread",
            tags_extra=spec["tags_extra"],
            input_units=spec.get("input_units"),
            output_unit=spec.get("output_unit"),
            param_specs=spec.get("param_specs"),
            relational_specs=spec.get("relational_specs"),
        )
    union_extended(*_SPECS.keys())


_register()
