# -*- coding: utf-8 -*-
"""Backward-compatible explicit top/bottom-K rolling operators."""
from __future__ import annotations

import numpy as np

from factor_engine.cleaned_operators.overhaul.base import (
    Spec,
    frame_pd,
    pl,
    pl_base_with,
    pl_cols,
    positive_int,
    register_specs,
)


def _array(values, window, k, min_periods, top, stat):
    window = positive_int(window, "window")
    k = window if k is None else positive_int(k, "k")
    if k > window:
        raise ValueError("k must not exceed window")
    if stat == "std" and k < 2:
        raise ValueError("sample Top/Bottom-K standard deviation requires k >= 2")
    min_periods = k if min_periods is None else positive_int(min_periods, "min_periods")
    if min_periods > window:
        raise ValueError("min_periods must not exceed window")
    required = max(k, min_periods, 2 if stat == "std" else 1)
    out = np.full(values.shape, np.nan, dtype=float)
    for col in range(values.shape[1]):
        for row in range(values.shape[0]):
            sample = values[max(0, row - window + 1): row + 1, col]
            sample = np.sort(sample[np.isfinite(sample)])
            if sample.size < required:
                continue
            selected = sample[-k:] if top else sample[:k]
            if stat == "mean":
                out[row, col] = selected.mean()
            elif stat == "sum":
                out[row, col] = selected.sum()
            else:
                out[row, col] = selected.std(ddof=1)
    return out


def _pd(x, window, k=None, min_periods=None, *, top, stat, **_):
    return frame_pd(x, _array(x.to_numpy(dtype=float), window, k, min_periods, top, stat))


def _pl(x, window, k=None, min_periods=None, *, top, stat, **_):
    cols = pl_cols(x)
    values = np.column_stack([x[c].cast(pl.Float64, strict=False).to_numpy() for c in cols])
    out = _array(values, window, k, min_periods, top, stat)
    return pl_base_with(x, {c: pl.Series(c, out[:, i]) for i, c in enumerate(cols)})


def _spec(top, stat):
    return (
        lambda x, window, k=None, min_periods=None, **kw: _pd(
            x, window, k, min_periods, top=top, stat=stat, **kw
        ),
        lambda x, window, k=None, min_periods=None, **kw: _pl(
            x, window, k, min_periods, top=top, stat=stat, **kw
        ),
    )


def register():
    specs = {}
    for name, top, stat, description in (
        ("ts_topk_mean", True, "mean", "mean of largest K values"),
        ("ts_topk_sum", True, "sum", "sum of largest K values"),
        ("ts_topk_std", True, "std", "sample std (ddof=1) of largest K values; k>=2"),
        ("ts_bottomk_mean", False, "mean", "mean of smallest K values"),
        ("ts_bottomk_sum", False, "sum", "sum of smallest K values"),
        ("ts_bottomk_std", False, "std", "sample std (ddof=1) of smallest K values; k>=2"),
    ):
        pandas_fn, polars_fn = _spec(top, stat)
        # ts_topk_sum canonical params are ``["x","d","k"]`` (R19-050 aliases);
        # inherited ParamSpec keys must sit inside param_names (R6-157).
        params = ["x", "d", "k", "min_periods"] if name == "ts_topk_sum" else ["x", "window", "k", "min_periods"]
        specs[name] = Spec(
            "time_series_order",
            params,
            description,
            pandas_fn,
            polars_fn,
        )
    register_specs(specs)


register()
