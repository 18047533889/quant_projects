# -*- coding: utf-8 -*-
"""Targeted native-Polars fixes discovered by backend parity CI."""
from __future__ import annotations

import numpy as np

from cleaned_operators.overhaul.base import (
    PolarsFunctionOperator,
    pl,
    pl_unary_rolling_map,
    positive_int,
    window_params,
)
from cleaned_operators.registry import OperatorRegistry


def topbottom_native(x, window, k, min_periods, *, top: bool, stat: str):
    w, mp = window_params(window, min_periods, default_mp=1)
    k_i = positive_int(k, "k")
    if k_i > w:
        raise ValueError("k must not exceed window")

    def fn(values):
        # Polars rolling_map supplies a Series-like object. Convert once before
        # applying NumPy finite-value semantics; never convert the whole panel
        # to pandas or route through a pandas operator.
        arr = np.asarray(values, dtype=float)
        valid = np.sort(arr[np.isfinite(arr)])
        if valid.size < max(mp, k_i):
            return np.nan
        selected = valid[-k_i:] if top else valid[:k_i]
        if stat == "mean":
            return float(selected.mean())
        if stat == "sum":
            return float(selected.sum())
        return float(selected.std(ddof=1)) if selected.size >= 2 else np.nan

    return pl_unary_rolling_map(x, w, 1, fn)


def register() -> None:
    if pl is None:
        return
    for name, top, stat, description in (
        ("ts_topk_mean", True, "mean", "窗口内最大 K 个值的均值"),
        ("ts_topk_sum", True, "sum", "窗口内最大 K 个值之和"),
        ("ts_topk_std", True, "std", "窗口内最大 K 个值的标准差"),
        ("ts_bottomk_mean", False, "mean", "窗口内最小 K 个值的均值"),
        ("ts_bottomk_sum", False, "sum", "窗口内最小 K 个值之和"),
        ("ts_bottomk_std", False, "std", "窗口内最小 K 个值的标准差"),
    ):
        def calculate(x, window, k, min_periods=None, _top=top, _stat=stat, **_):
            return topbottom_native(
                x,
                window,
                k,
                min_periods,
                top=_top,
                stat=_stat,
            )

        OperatorRegistry.register(
            PolarsFunctionOperator(
                name,
                "time_series_order",
                ["x", "window", "k", "min_periods"],
                description,
                calculate,
            ),
            canonical=name,
            backend="polars",
            source="operator_overhaul_native_polars",
            status="production",
            backend_explicit=True,
            # Round-7 P0 override-chain pinning: pin the exact prior polars source.
            replace=True,
            replacement_reason="overhaul polars-fix layer replaces prior bootstrap source",
            expected_old_source=str(
                (OperatorRegistry._catalog.get(name, {}).get("backend_meta") or {})
                .get("polars", {}).get("source", "") or ""
            ),
        )
