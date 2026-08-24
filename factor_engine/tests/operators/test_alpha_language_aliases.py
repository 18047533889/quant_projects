# -*- coding: utf-8 -*-
"""Alpha-language DSL aliases resolve to their existing canonical targets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

ALIASES = {
    "event_age": "ts_days_since",
    "event_decay": "event_decay_asof",
    "event_interval_mean": "ts_event_spacing_mean",
    "event_interval_cv": "ts_event_spacing_cv",
    "report_lag": "fin_lag",
    "report_age": "fin_staleness",
    "report_single_quarter": "fin_quarter_from_cumulative",
    "report_rolling_std": "fin_std",
    "report_rank": "fin_percentile_history",
    "report_surprise_to_trend": "fin_surprise_zscore",
}


def test_alpha_aliases_resolve_to_expected_canonical():
    for alias, target in ALIASES.items():
        assert OperatorRegistry.resolve_canonical(alias) == target, f"{alias} -> {target}"


def test_alpha_aliases_run_on_pandas():
    """Aliases resolve to an executable pandas runtime with a working DSL name."""
    n, cols = 60, 2
    dates = pd.bdate_range("2024-01-02", periods=n)
    assets = [f"S{i}" for i in range(cols)]
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(0, 1, (n, cols)), index=dates, columns=assets)
    cond = pd.DataFrame(x > 0, index=dates, columns=assets)
    for alias in ("event_age", "event_decay", "event_interval_mean", "event_interval_cv"):
        inst = OperatorRegistry.get(alias, "pandas_numpy")
        out = inst.calculate(cond, 20)
        assert out.shape == cond.shape
        assert out.dtypes.apply(lambda d: d == float).all()


def test_event_decay_matches_event_decay_asof_output():
    dates = pd.bdate_range("2024-01-02", periods=40)
    x = pd.DataFrame(
        np.where(np.arange(40).reshape(-1, 1) % 7 == 0, 1.0, 0.0) * np.ones((1, 2)),
        index=dates, columns=["S0", "S1"],
    )
    via_alias = OperatorRegistry.get("event_decay", "pandas_numpy").calculate(x, 10)
    direct = OperatorRegistry.get("event_decay_asof", "pandas_numpy").calculate(x, 10)
    pd.testing.assert_frame_equal(via_alias, direct)
