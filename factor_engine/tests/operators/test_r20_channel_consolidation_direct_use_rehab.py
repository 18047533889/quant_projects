# -*- coding: utf-8 -*-
"""R20 channel/consolidation DirectUse rehabilitation oracles.

Slice: R20-CHANNEL-CONSOLIDATION-DIRECTUSE.

Duplicate-audit outcome (documented skip):
* ``channel_width_pct``  == ``donchian_width_pct``        (same inclusive-t
  rolling_max(high)/rolling_min(low) band math) -> NOT registered.
* ``channel_position``   == ``donchian_channel_position`` (same
  (close-lo)/(hi-lo) ratio) -> NOT registered.
* ``ts_channel_width_pct`` (structure_patterns_v2) is regression-line fitted
  (pivot lines, abs()-close) — mathematically distinct, unchanged.
* ``ts_consolidation_width`` is PRICE-SCALE (no /close normalization) — the
  dimensionless close-only tightness canonicals below are genuinely new.

Landed canonicals (both rolling-only, trailing window ENDING at t,
min_periods=window, NOT in ``_RECURSIVE_EWM``):
* ``consolidation_pct``       — rolling_std(close,w)/rolling_mean(close,w)
  (coefficient of variation; low = consolidation)
* ``consolidation_range_pct`` — (rolling_max(close,w)-rolling_min(close,w))/close
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ensure_technical_chain() -> None:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    if OperatorRegistry.lifecycle() == "frozen":
        return
    if OperatorRegistry.get("consolidation_pct", "pandas_numpy") is not None:
        return
    from factor_engine.cleaned_operators.technical import signal  # noqa: F401
    from factor_engine.cleaned_operators.technical import polars_signal  # noqa: F401
    from factor_engine.cleaned_operators import composite_fastpath  # noqa: F401
    from factor_engine.cleaned_operators.technical import indicators_v2  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def _bootstrap():
    _ensure_technical_chain()


def _op(name: str):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)
    assert op is not None, f"{name} not registered"
    return op


def _trending_panel(n=40, seed=7):
    rng = np.random.default_rng(seed)
    close = pd.DataFrame({"A": 100.0 + np.cumsum(rng.normal(0.05, 0.6, n))})
    return close


# ---------------------------------------------------------------------------
# consolidation_pct — coefficient of variation oracle
# ---------------------------------------------------------------------------
def test_consolidation_pct_matches_manual_oracle():
    close = _trending_panel()
    w = 5
    mean = close.rolling(w, min_periods=w).mean()
    std = close.rolling(w, min_periods=w).std(ddof=1)
    expected = std / mean.where(mean > 0.0)
    out = _op("consolidation_pct").calculate(close, window=w)
    pd.testing.assert_frame_equal(out, expected)
    # Warmup: first w-1 bars NaN, every post-warmup bar finite and positive.
    assert out.iloc[: w - 1].isna().all().all()
    assert out.iloc[w - 1:].notna().all().all()
    assert (out.dropna() > 0.0).all().all()


def test_consolidation_pct_is_scale_invariant():
    # CV is dimensionless: a 100x-scaled panel gives identical values.
    close = _trending_panel(n=50, seed=3)
    a = _op("consolidation_pct").calculate(close, window=6)
    b = _op("consolidation_pct").calculate(close * 100.0, window=6)
    np.testing.assert_allclose(
        a.to_numpy(), b.to_numpy(), rtol=1e-10, atol=1e-12, equal_nan=True
    )


def test_consolidation_pct_masks_non_positive_close():
    # R5-38: close<=0 inside the window contaminates the level series — the
    # masked series yields NaN-driven stats, and a bad close at t is NaN.
    close = _trending_panel(n=30, seed=5)
    close.iloc[10, 0] = -1.0
    out = _op("consolidation_pct").calculate(close, window=5)
    # bad close at t -> NaN at t (masked input has NaN there)
    assert np.isnan(out.iloc[10, 0])
    # any window containing index 10 has <w valid obs -> NaN (min_periods=w)
    assert out.iloc[10:15].isna().all().all()
    # windows fully after the bad bar recover
    assert out.iloc[15:].notna().all().all()


def test_consolidation_pct_constant_series_is_zero():
    # Perfectly flat close -> std == 0 -> CV == 0 (max consolidation), not NaN.
    close = pd.DataFrame({"A": [10.0] * 12})
    out = _op("consolidation_pct").calculate(close, window=4)
    assert out.iloc[:3].isna().all().all()
    assert np.allclose(out.iloc[3:].to_numpy(), 0.0)


# ---------------------------------------------------------------------------
# consolidation_range_pct — close-only range tightness oracle
# ---------------------------------------------------------------------------
def test_consolidation_range_pct_matches_manual_oracle():
    close = _trending_panel()
    w = 5
    pos = close.where(close > 0.0)
    hi = pos.rolling(w, min_periods=w).max()
    lo = pos.rolling(w, min_periods=w).min()
    expected = (hi - lo) / pos
    out = _op("consolidation_range_pct").calculate(close, window=w)
    pd.testing.assert_frame_equal(out, expected)
    assert out.iloc[: w - 1].isna().all().all()
    assert out.iloc[w - 1:].notna().all().all()
    assert (out.dropna() >= 0.0).all().all()


def test_consolidation_range_pct_is_scale_invariant():
    close = _trending_panel(n=50, seed=11)
    a = _op("consolidation_range_pct").calculate(close, window=6)
    b = _op("consolidation_range_pct").calculate(close * 0.01, window=6)
    np.testing.assert_allclose(
        a.to_numpy(), b.to_numpy(), rtol=1e-10, atol=1e-12, equal_nan=True
    )


def test_consolidation_range_pct_masks_non_positive_close():
    close = _trending_panel(n=30, seed=5)
    close.iloc[10, 0] = -1.0
    out = _op("consolidation_range_pct").calculate(close, window=5)
    assert np.isnan(out.iloc[10, 0])
    # any window containing index 10 (t in 10..14, window 5) is NaN — the
    # masked rolling max/min sees < w valid obs; earlier/later windows recover.
    assert out.iloc[10:15].isna().all().all()
    assert out.iloc[15:].notna().all().all()
    # windows entirely BEFORE the bad bar (t in 5..9: indices t-4..t) stay valid
    assert out.iloc[5:10].notna().all().all()


def test_consolidation_range_pct_constant_series_is_zero():
    close = pd.DataFrame({"A": [10.0] * 12})
    out = _op("consolidation_range_pct").calculate(close, window=4)
    assert out.iloc[:3].isna().all().all()
    assert np.allclose(out.iloc[3:].to_numpy(), 0.0)


def test_consolidation_measures_agree_on_rank():
    # On a synthetic regime switch (tight range -> wide drift) both tightness
    # measures must rank the consolidated segment strictly tighter.
    tight = 100.0 + np.sin(np.arange(20) * 0.05) * 0.05
    wide = 100.0 + np.linspace(0.0, 8.0, 20)
    close = pd.DataFrame({"A": np.concatenate([tight, wide])})
    out = _op("consolidation_pct").calculate(close, window=10)
    rng = _op("consolidation_range_pct").calculate(close, window=10)
    for res in (out, rng):
        assert res.iloc[-1, 0] > res.iloc[len(tight) - 1, 0]


# ---------------------------------------------------------------------------
# causality / prefix invariance
# ---------------------------------------------------------------------------
def test_consolidation_family_is_prefix_invariant_under_truncation():
    # Rolling windows end at t with bounded state: truncated == full prefix.
    close = _trending_panel(n=50, seed=13)
    for name in ("consolidation_pct", "consolidation_range_pct"):
        full = _op(name).calculate(close, window=6)
        head = _op(name).calculate(close.iloc[:35], window=6)
        pd.testing.assert_frame_equal(full.iloc[:35], head)


# ---------------------------------------------------------------------------
# governance / promotion
# ---------------------------------------------------------------------------
def test_consolidation_family_param_specs_and_governance():
    for name in ("consolidation_pct", "consolidation_range_pct"):
        specs = _op(name).metadata.param_specs
        assert set(specs) == {"window"}, f"{name} ParamSpec keys: {sorted(specs)}"
        assert specs["window"].dtype is int
        assert specs["window"].min == 2
        assert specs["window"].param_role is not None
        # single-window family: no relational specs
        assert not _op(name).metadata.relational_specs

    from factor_engine.cleaned_operators.technical.indicators_v2 import _RECURSIVE_EWM

    for name in ("consolidation_pct", "consolidation_range_pct"):
        assert name not in _RECURSIVE_EWM
        tags = _op(name).metadata.tags
        assert "stateful" not in tags
        assert "causal" in tags and "pit_safe" in tags


def test_consolidation_family_promotion_and_duplicate_skip():
    from factor_engine.mining.direct_use import _RELATIVE_ALPHA_OPS

    promoted = {"consolidation_pct", "consolidation_range_pct"}
    assert promoted <= _RELATIVE_ALPHA_OPS

    from factor_engine.cleaned_operators.operator_surface import (
        _DAILY_CHANNEL_CONSOLIDATION_PACK_2026_08,
        classify_canonical,
    )

    assert promoted <= _DAILY_CHANNEL_CONSOLIDATION_PACK_2026_08
    assert _DAILY_CHANNEL_CONSOLIDATION_PACK_2026_08 <= set(
        __import__("factor_engine.cleaned_operators.operator_surface", fromlist=["DAILY_FACTOR_MIGRATED"]).DAILY_FACTOR_MIGRATED
    )
    for name in promoted:
        assert classify_canonical(name) == "daily"

    # Duplicate-skip contract: the channel_* math is already registered under
    # the donchian_* names and must NOT be re-registered.
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    assert OperatorRegistry.get("channel_width_pct", "pandas_numpy") is None
    assert OperatorRegistry.get("channel_position", "pandas_numpy") is None
    for dup in ("channel_width_pct", "channel_position"):
        assert dup not in _RELATIVE_ALPHA_OPS
    # donchian twins remain promoted (unchanged by this slice)
    assert {"donchian_width_pct", "donchian_channel_position"} <= _RELATIVE_ALPHA_OPS


def test_consolidation_names_not_price_level_intermediate():
    # The new names are dimensionless alphas, not raw price-scale levels.
    from factor_engine.mining.direct_use import _PRICE_LEVEL_INTERMEDIATE_OPS

    for name in ("consolidation_pct", "consolidation_range_pct"):
        assert name not in _PRICE_LEVEL_INTERMEDIATE_OPS
