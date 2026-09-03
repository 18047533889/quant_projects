# -*- coding: utf-8 -*-
"""PERF-2: vectorized daily_agg kernel whitelist binder.

Attaches the ``__vec__`` attribute to the scalar kernel functions in the
consumer modules so that the dispatch points in ``_core.daily_agg{,_two,_three}``
route to the (day, bar, inst) 3-D vectorized implementations.

WHITELIST RULE: an operator kernel is added here ONLY after the PERF-2
equivalence harness (``factor_engine/tests/perf_intra_vec_equiv.py``) proves
byte-equivalence (rtol/atol 1e-12) over random fixtures incl. NaN gaps and
all-NaN days.  An unproven kernel is NOT whitelisted: ``daily_agg*`` falls
back to the scalar path unchanged.
"""
from __future__ import annotations

from factor_engine.cleaned_operators.intraday import _core as _c


def _bind_one(fn, vec) -> None:
    if vec is None:
        raise LookupError(
            f"PERF-2 binder: no vector kernel implementation for {fn.__name__} — "
            "refusing to silently bind a dead path"
        )
    fn.__vec__ = vec  # type: ignore[attr-defined]


def bind_whitelist(force: bool = True) -> list[str]:
    """Bind the current PERF-2 kernel whitelist; returns whitelisted ids.

    Fail-closed (100k GO §6.3): only harness-proven kernels are bound.  A
    bind whose vector implementation is missing raises instead of silently
    falling back to the scalar path (which would silently time scalar under a
    "vec" benchmark).

    The bound whitelist is surfaced to the ``intraday_vector_kernels_bound``
    counter and the harness ``run_all()`` reads ``count_bound()`` so a bind
    whose target kernel is no longer a registered vector implementation (or a
    removed scalar kernel) raises loudly instead of silently narrowing the
    fast path.
    """
    from factor_engine.cleaned_operators.intraday.smart_money import (
        _stock_graph_features,
        _common_trading_intensity,
    )
    from factor_engine.cleaned_operators.intraday.true_gap_batch3 import (
        _session_mean_reversion_kernel,
        _price_delay_kernel,
        _volume_imbalance_kernel,
    )
    from factor_engine.cleaned_operators.intraday.vwap_path import (
        _time_above_vwap,
    )

    # (scalar kernel, vector impl name) — a missing vector impl raises loudly
    # (fail-closed) rather than silently taking the scalar path.
    pairs = [
        (_session_mean_reversion_kernel, "_vec_session_mean_reversion"),
        (_price_delay_kernel, "_vec_price_delay"),
        (_volume_imbalance_kernel, "_vec_volume_imbalance"),
        (_stock_graph_features, "_vec_stock_graph_features"),
        (_common_trading_intensity, "_vec_common_trading_intensity"),
        (_time_above_vwap, "_vec_time_above_vwap"),
    ]
    bound: list[str] = []
    _bound_fns = []
    for fn, vec_name in pairs:
        _bind_one(fn, getattr(_c, vec_name, None))
        bound.append(fn.__name__)
        _bound_fns.append(fn)
    globals()["_BOUND_FNS"] = list(_bound_fns)
    return [
        "true_gap_batch3:intra_session_mean_reversion[min_finite]",
        "true_gap_batch3:intra_price_delay[min_finite]",
        "true_gap_batch3:intra_volume_imbalance[min_finite]",
        "smart_money:intra_dynamic_stock_graph_features[min_finite]",
        "smart_money:intra_common_trading_intensity[min_finite]",
        "vwap_path:intra_time_above_vwap[min_finite]",
    ]


def count_bound() -> int:
    """Number of distinct kernels with a live ``__vec__`` attribute.

    Used by the equivalence harness to fail loudly if a bind silently dropped a
    kernel (e.g. a whitelisted scalar kernel removed/renamed), per 100k GO §6.3
    telemetry (``intraday_vector_kernels_bound``).
    """
    fns = globals().get("_BOUND_FNS", [])
    return sum(1 for fn in fns if getattr(fn, "__vec__", None) is not None)


def is_whitelisted(fn) -> bool:
    return getattr(fn, "__vec__", None) is not None