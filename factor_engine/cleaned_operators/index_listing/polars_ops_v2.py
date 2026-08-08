# -*- coding: utf-8 -*-
"""Polars backends for next-stage index / listing operators (genuine expressions)."""
from __future__ import annotations

from typing import Any

import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator


def _cols(df, *others):
    cols = [c for c in df.columns if c != "date"]
    for o in others:
        cols = [c for c in cols if c in o.columns]
    return cols


def _meta(name: str, description: str, params: list[str]) -> OperatorMetadata:
    return OperatorMetadata(
        name=name, category="index", description=description, param_names=params,
        return_type="series", tags=["index", "polars", "native", "typed_v2"],
    )


def _register(name: str, description: str, params: list[str], fn):
    @register_operator(
        name=name, category="index", business_category="index",
        canonical=name, source="index_listing.polars_ops_v2",
    )
    class _IndexListingPolars(SeriesOperator):
        metadata = _meta(name, description, params)

        def _calculate_series(self, *args, **kwargs):
            return fn(*args, **kwargs)

    return _IndexListingPolars


def _binary(a, b, expr_fn):
    cols = _cols(a, b)
    return a.with_columns([expr_fn(a[c], b[c]).alias(c) for c in cols])


_register(
    "index_weight_gap_to_free_float", "指数权重 vs 自由流通权重偏离（Polars）。", ["index_weight", "free_float_weight"],
    lambda iw, fw: _binary(iw, fw, lambda a, b: (a - b) / b.fill_null(0.0).replace(0, None).abs()),
)
_register(
    "suspension_frequency", "停牌频率（Polars）。", ["is_suspend", "window"],
    lambda is_s, window=60: is_s.with_columns(
        [
            pl.when(
                is_s[c].is_not_null().cast(pl.Float64).rolling_sum(int(window)) == 0
            )
            .then(None)
            .otherwise(
                (is_s[c].is_not_null() & (is_s[c] != 0))
                .cast(pl.Float64)
                .rolling_sum(int(window))
                / is_s[c].is_not_null().cast(pl.Float64).rolling_sum(int(window))
            )
            .alias(c)
            for c in _cols(is_s)
        ]
    ),
)


def _reconstitution_churn(member, window):
    w = int(window)
    m = member
    cols = _cols(m)
    out = []
    for c in cols:
        flag = m[c].fill_null(0.0)
        entries = ((flag == 1) & (flag.shift(1).fill_null(0.0) != 1)).cast(pl.Float64)
        exits = ((flag != 1) & (flag.shift(1).fill_null(0.0) == 1)).cast(pl.Float64)
        out.append((entries + exits).rolling_sum(w).alias(c))
    return m.with_columns(out)


_register("index_reconstitution_churn", "窗口内纳入+剔除次数（Polars）。", ["member", "window"],
          lambda m, window=60: _reconstitution_churn(m, int(window)))


def _multi_index_entry(entry_a, entry_b, entry_c, window):
    # Unknown-state contract (S9): NaN must not be treated as "not entered".
    # Sum only the known index flags per row; rows with all three unknown are NaN.
    w = int(window)
    cols = _cols(entry_a, entry_b, entry_c)
    out = []
    for c in cols:
        a = entry_a[c].fill_null(0.0).cast(pl.Float64)
        b = entry_b[c].fill_null(0.0).cast(pl.Float64)
        cc = entry_c[c].fill_null(0.0).cast(pl.Float64)
        known = entry_a[c].is_not_null() | entry_b[c].is_not_null() | entry_c[c].is_not_null()
        total = pl.DataFrame(
            {"_v": a + b + cc, "_k": known}
        ).with_columns(
            pl.when(pl.col("_k")).then(pl.col("_v")).otherwise(None).alias("_o")
        )["_o"]
        out.append(total.rolling_sum(w).alias(c))
    return entry_a.with_columns(out)


_register("multi_index_entry_intensity", "多指数同时纳入强度（Polars）。", ["entry_index_a", "entry_index_b", "entry_index_c", "window"],
          lambda a, b, c, window=20: _multi_index_entry(a, b, c, int(window)))


def _event_decay(entry_event, window, decay, missing_policy="break"):
    w = int(window)
    d = float(decay)
    policy = str(missing_policy or "break").lower()
    import numpy as np

    weights = np.array([d ** k for k in range(w)], dtype=float)

    def _apply(vals):
        arr = np.asarray(vals, dtype=float)
        if len(arr) < w:
            return float("nan")
        # R4 (unknown != no-event): a NaN event is UNKNOWN, never a 0 event.
        # Feeding a gap through as zeros manufactures a fake "no event" window.
        # Both policies leave the output undefined while a gap is inside the
        # window; "break" also resets the observed state, "carry" keeps the
        # weight state (rolling_map recomputes per window, so the observable
        # difference is limited to the reset behavior documented on pandas).
        if not np.all(np.isfinite(arr)):
            return float("nan")
        return float(np.dot(arr[::-1], weights))

    cols = _cols(entry_event)
    return entry_event.with_columns(
        [entry_event[c].rolling_map(_apply, window_size=w).alias(c) for c in cols]
    )


_register("index_event_decay", "指数事件衰减近因信号（Polars）。",
          ["entry_event", "window", "decay", "missing_policy"],
          lambda e, window=20, decay=0.9, missing_policy="break": _event_decay(
              e, int(window), float(decay), missing_policy,
          ))
