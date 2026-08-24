# -*- coding: utf-8 -*-
"""Bounded production contracts for pivot-derived support/resistance operators.

A structural line may not search arbitrarily far into pre-run history. Every
operator below uses an explicit ``history_window`` measured in confirmation bars,
which makes warmup, replay and incremental materialization deterministic.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator


def _positive(value: int, name: str, *, minimum: int = 1) -> int:
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return parsed


def _events(frame: pd.DataFrame, left_window: int, right_window: int, *, high: bool):
    left = _positive(left_window, "left_window")
    right = _positive(right_window, "right_window")
    values = frame.to_numpy(dtype=float)
    rows, cols = values.shape
    prices = np.full((rows, cols), np.nan, dtype=float)
    pivot_positions = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        series = values[:, col]
        for pivot in range(left, rows - right):
            window = series[pivot-left:pivot+right+1]
            if not np.isfinite(window).all():
                continue
            center = float(series[pivot])
            others = np.delete(window, left)
            ok = center > float(np.max(others)) if high else center < float(np.min(others))
            if ok:
                confirmed_at = pivot + right
                prices[confirmed_at, col] = center
                pivot_positions[confirmed_at, col] = float(pivot)
    return prices, pivot_positions


def _bounded_last(frame, left_window, right_window, history_window, *, high, output):
    history = _positive(history_window, "history_window")
    prices, positions = _events(frame, left_window, right_window, high=high)
    rows, cols = prices.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        active: deque[tuple[int, float, float]] = deque()
        for t in range(rows):
            cutoff = t - history + 1
            while active and active[0][0] < cutoff:
                active.popleft()
            if np.isfinite(prices[t, col]):
                active.append((t, positions[t, col], prices[t, col]))
            if not active:
                continue
            confirm, pivot, price = active[-1]
            if output == "price":
                out[t, col] = price
            elif output == "confirmation_age":
                out[t, col] = float(t - confirm)
            elif output == "pivot_age":
                out[t, col] = float(t - pivot)
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def _bounded_line(frame, left_window, right_window, history_window, points, *, high, output, log=False):
    history = _positive(history_window, "history_window")
    k = _positive(points, "points", minimum=2)
    prices, positions = _events(frame, left_window, right_window, high=high)
    rows, cols = prices.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        active: deque[tuple[int, float, float]] = deque()
        for t in range(rows):
            cutoff = t - history + 1
            while active and active[0][0] < cutoff:
                active.popleft()
            if np.isfinite(prices[t, col]) and np.isfinite(positions[t, col]):
                active.append((t, positions[t, col], prices[t, col]))
            if len(active) < 2:
                continue
            selected = list(active)[-k:]
            xs = np.asarray([row[1] for row in selected], dtype=float)
            ys = np.asarray([row[2] for row in selected], dtype=float)
            if log:
                ys = np.log(ys)
            xbar, ybar = xs.mean(), ys.mean()
            denom = float(np.dot(xs-xbar, xs-xbar))
            if denom <= 0:
                continue
            slope = float(np.dot(xs-xbar, ys-ybar) / denom)
            out[t, col] = slope if output == "slope" else ybar + slope*(float(t)-xbar)
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def _register(name, params, fn, description):
    metadata = OperatorMetadata(
        name=name,
        category="price_structure",
        description=description,
        param_names=list(params),
        return_type="series",
        tags=["pit_safe","causal","bounded_history","production_repair"],
    )
    def _calculate_series(self, *args, **kwargs):
        return fn(*args, **kwargs)
    cls = type(
        f"Bounded_{name}",
        (SeriesOperator,),
        {"metadata":metadata,"_calculate_series":_calculate_series,"__module__":__name__},
    )
    register_operator(
        name=name,
        category="price_structure",
        business_category="technical_extension",
        canonical=name,
        source="bounded_structure_repairs",
        backend="pandas_numpy",
        status="production",
    )(cls)


def _last_high(high,left_window,right_window,history_window):
    return _bounded_last(high,left_window,right_window,history_window,high=True,output="price")
def _last_low(low,left_window,right_window,history_window):
    return _bounded_last(low,left_window,right_window,history_window,high=False,output="price")
def _high_age(high,left_window,right_window,history_window):
    return _bounded_last(high,left_window,right_window,history_window,high=True,output="pivot_age")
def _low_age(low,left_window,right_window,history_window):
    return _bounded_last(low,left_window,right_window,history_window,high=False,output="pivot_age")
def _res_level(high,left_window,right_window,history_window,points):
    return _bounded_line(high,left_window,right_window,history_window,points,high=True,output="level")
def _sup_level(low,left_window,right_window,history_window,points):
    return _bounded_line(low,left_window,right_window,history_window,points,high=False,output="level")
def _res_slope(high,left_window,right_window,history_window,points):
    return _bounded_line(high,left_window,right_window,history_window,points,high=True,output="slope")
def _sup_slope(low,left_window,right_window,history_window,points):
    return _bounded_line(low,left_window,right_window,history_window,points,high=False,output="slope")
def _dist_res(close,high,left_window,right_window,history_window,points):
    level=_res_level(high,left_window,right_window,history_window,points)
    return close/level.replace(0,np.nan)-1.0
def _dist_sup(close,low,left_window,right_window,history_window,points):
    level=_sup_level(low,left_window,right_window,history_window,points)
    return close/level.replace(0,np.nan)-1.0
def _break_res(close,high,left_window,right_window,history_window,points):
    return _dist_res(close,high,left_window,right_window,history_window,points).clip(lower=0.0)
def _break_sup(close,low,left_window,right_window,history_window,points):
    return (-_dist_sup(close,low,left_window,right_window,history_window,points)).clip(lower=0.0)
def _res_log_slope(high,left_window,right_window,history_window,points):
    # Log-price slope: fit the line on ln(pivot price) vs pivot bar, making the
    # slope a per-bar log return comparable across price levels (audit 13).
    return _bounded_line(high,left_window,right_window,history_window,points,high=True,output="slope",log=True)
def _sup_log_slope(low,left_window,right_window,history_window,points):
    return _bounded_line(low,left_window,right_window,history_window,points,high=False,output="slope",log=True)

_register("ts_last_pivot_high",["high","left_window","right_window","history_window"],_last_high,"Latest confirmed pivot high inside bounded confirmation history.")
_register("ts_last_pivot_low",["low","left_window","right_window","history_window"],_last_low,"Latest confirmed pivot low inside bounded confirmation history.")
_register("ts_pivot_high_age",["high","left_window","right_window","history_window"],_high_age,"Bars since original pivot-high bar, after confirmation, bounded by history_window.")
_register("ts_pivot_low_age",["low","left_window","right_window","history_window"],_low_age,"Bars since original pivot-low bar, after confirmation, bounded by history_window.")
_register("ts_resistance_level",["high","left_window","right_window","history_window","points"],_res_level,"Projected resistance line from recent confirmed pivot highs.")
_register("ts_support_level",["low","left_window","right_window","history_window","points"],_sup_level,"Projected support line from recent confirmed pivot lows.")
_register("ts_resistance_slope",["high","left_window","right_window","history_window","points"],_res_slope,"Slope of bounded resistance line.")
_register("ts_support_slope",["low","left_window","right_window","history_window","points"],_sup_slope,"Slope of bounded support line.")
_register("ts_distance_to_resistance",["close","high","left_window","right_window","history_window","points"],_dist_res,"Signed close-to-resistance distance.")
_register("ts_distance_to_support",["close","low","left_window","right_window","history_window","points"],_dist_sup,"Signed close-to-support distance.")
_register("ts_resistance_break",["close","high","left_window","right_window","history_window","points"],_break_res,"Positive breakout magnitude above bounded resistance line.")
_register("ts_support_break",["close","low","left_window","right_window","history_window","points"],_break_sup,"Positive breakdown magnitude below bounded support line.")
_register("ts_resistance_log_slope",["high","left_window","right_window","history_window","points"],_res_log_slope,"Log-price slope of the bounded resistance line fitted to confirmed pivot highs.")
_register("ts_support_log_slope",["low","left_window","right_window","history_window","points"],_sup_log_slope,"Log-price slope of the bounded support line fitted to confirmed pivot lows.")

# New split canonicals live on the same extended surface as the rest of the
# structure pack.  The raw ts_resistance_slope / ts_support_slope remain as the
# price-level raw variants (extended/research); these log-price variants are the
# preferred canonical form.
import cleaned_operators.operator_surface as _surface
_surface.extend_extended_only({"ts_resistance_log_slope", "ts_support_log_slope"})
