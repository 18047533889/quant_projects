# -*- coding: utf-8 -*-
"""R26-088..092: event_interval window clock + EventBool semantics.

* the window semantics is BAR (trailing rows), declared MIN_SUPPORT_WINDOW;
* min-interval support matches the kernel (>=6);
* EventBool is strict {0,1,NaN} — non-binary finite values raise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_event_interval_window_is_bar_not_event_count():
    import factor_engine.cleaned_operators.event_interval  # noqa: F401  (declares at import)
    from factor_engine.cleaned_operators.closure.window_semantics import window_semantics_for
    s = window_semantics_for("event_interval_memory")
    assert s is not None and s.value == "min_support_window"


def test_event_bool_strict():
    from factor_engine.cleaned_operators.event_interval import _event_mask

    assert np.array_equal(_event_mask(np.array([1.0, 0.0, np.nan, 1.0])),
                          np.array([True, False, False, True]))
    with pytest.raises(ValueError, match="EventBool"):
        _event_mask(np.array([1.0, 0.2, 0.0]))


def test_event_interval_memory_min_support_six():
    from factor_engine.cleaned_operators.event_interval import _interval_memory_series

    # Irregular spacing so consecutive-interval correlation is defined.
    rng = np.random.default_rng(0)
    ev = np.zeros(80)
    pos = np.sort(rng.choice(80, size=12, replace=False))
    ev[pos] = 1.0
    out = _interval_memory_series(ev[:, None], 80, 80)
    assert np.isfinite(out[-1, 0])
    # 4 events -> 3 intervals -> below the 6-interval floor -> NaN.
    ev4 = np.zeros(80)
    ev4[np.sort(rng.choice(80, size=4, replace=False))] = 1.0
    out4 = _interval_memory_series(ev4[:, None], 80, 80)
    assert np.isnan(out4[-1, 0])
