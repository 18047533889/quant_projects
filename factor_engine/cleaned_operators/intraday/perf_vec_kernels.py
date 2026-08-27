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
    fn.__vec__ = vec  # type: ignore[attr-defined]


def bind_whitelist(force: bool = True) -> list[str]:
    """Bind the current PERF-2 kernel whitelist; returns whitelisted ids."""
    from factor_engine.cleaned_operators.intraday.true_gap_batch3 import (  # noqa: F401
        _session_mean_reversion_kernel,
        _price_delay_kernel,
        _volume_imbalance_kernel,
    )
    from factor_engine.cleaned_operators.intraday.smart_money import (  # noqa: F401
        _common_trading_intensity,
        _stock_graph_features,
    )
    from factor_engine.cleaned_operators.intraday.vwap_path import (  # noqa: F401
        _vwap_path_common,
        _vwap_path_pct_common,
        _vwap_excursion,
        _time_above_vwap,
    )

    # ---- registry of vector kernels -----------------------------------------
    _bind_one(_session_mean_reversion_kernel, _c._vec_session_mean_reversion)
    _bind_one(_price_delay_kernel, _c._vec_price_delay)
    _bind_one(_volume_imbalance_kernel, _c._vec_volume_imbalance)

    ids = [
        "true_gap_batch3:intra_session_mean_reversion[min_finite]",
        "true_gap_batch3:intra_price_delay[min_finite]",
        "true_gap_batch3:intra_volume_imbalance[min_finite]",
    ]
    for rid, (n, _fn) in list(_c._DAILY_AGG_VEC_REGISTRY.items()):
        _ = n, _fn
    return ids


def is_whitelisted(fn) -> bool:
    return getattr(fn, "__vec__", None) is not None