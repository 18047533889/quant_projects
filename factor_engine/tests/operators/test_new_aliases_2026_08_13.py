# -*- coding: utf-8 -*-
"""Test for 29 newly added DSL aliases (2026-08-13).

Validates that common quant platform DSL names correctly map to canonical operators.
"""
import pytest


def test_ema_ewm_aliases(_loaded):
    """EMA/EWM family aliases map to ts_ema and ewm_* canonicals."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    # ts_ema aliases
    assert OperatorRegistry.get("ts_ewm_mean") is OperatorRegistry.get("ts_ema")
    assert OperatorRegistry.get("ts_ewma") is OperatorRegistry.get("ts_ema")

    # ewm_* family (ts_ewm_std/ts_ewm_var alias their ewm_* names; see below)
    assert OperatorRegistry.get("ts_ewm_std") is OperatorRegistry.get("ewm_std")
    assert OperatorRegistry.get("ts_ewm_var") is OperatorRegistry.get("ewm_var")
    # NOTE: ts_ewm_cov / ts_ewm_corr are NOT aliases.  They are genuine
    # polars-only pairwise canonicals (polars_native/ts_batch1.py), distinct from
    # the pandas EWMCov/EWMCorr registered under ewm_cov/ewm_corr.  These two are
    # tested in test_pairwise_ewm_are_distinct_canonicals below.


def test_sma_aliases(_loaded):
    """SMA aliases map to ts_mean canonical."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    assert OperatorRegistry.get("ts_sma") is OperatorRegistry.get("ts_mean")


def test_lag_delay_aliases(_loaded):
    """Lag/delay aliases map to ts_delay canonical."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    assert OperatorRegistry.get("ts_lag") is OperatorRegistry.get("ts_delay")


def test_decay_aliases(_loaded):
    """Decay aliases map to ts_decay_linear canonical."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    assert OperatorRegistry.get("wdecay") is OperatorRegistry.get("ts_decay_linear")


def test_safe_division_aliases(_loaded):
    """Safe division aliases map to safe_div_null canonical."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    assert OperatorRegistry.get("safe_divide") is OperatorRegistry.get("safe_div_null")
    assert OperatorRegistry.get("div_safe") is OperatorRegistry.get("safe_div_null")


def test_returns_aliases(_loaded):
    """Returns aliases map to ts_pct canonical."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    # NOTE: ``ret`` is intentionally NOT registered as an operator alias.  In the
    # LQTP surface ``ret`` is the conventional return FIELD, and an operator alias
    # collides (``rolling_beta_to_market(ret, 20)`` would DSLParseError).  ``ret()``
    # must still resolve through the field path.  ``ts_return``/``returns`` carry the
    # operator→ts_pct mapping instead (see _aliases / _dedupe).
    assert OperatorRegistry.get("returns") is OperatorRegistry.get("ts_pct")


def test_clip_aliases(_loaded):
    """Clipping aliases map to clip canonical."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    assert OperatorRegistry.get("bound") is OperatorRegistry.get("clip")


def test_pairwise_ewm_are_distinct_canonicals(_loaded):
    """ts_ewm_cov / ts_ewm_corr are polars-only pairwise canonicals, not aliases.

    They must NOT share identity with the pandas EWMCov/EWMCorr (ewm_cov/ewm_corr),
    and each must be independently callable on its own backend.
    """
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    cov_pair = OperatorRegistry.get("ts_ewm_cov", backend="polars")
    corr_pair = OperatorRegistry.get("ts_ewm_corr", backend="polars")
    cov_1d = OperatorRegistry.get("ewm_cov", backend="pandas_numpy")
    corr_1d = OperatorRegistry.get("ewm_corr", backend="pandas_numpy")
    assert cov_pair is not None
    assert corr_pair is not None
    assert cov_1d is not None
    assert corr_1d is not None
    assert cov_pair is not cov_1d  # distinct pairwise vs 1d estimators
    assert corr_pair is not corr_1d


def test_normalization_aliases(_loaded):
    """Normalization aliases map to zscore canonical."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    assert OperatorRegistry.get("cs_standardize") is OperatorRegistry.get("zscore")


def test_all_14_new_aliases_bidirectional(_loaded):
    """All 14 new aliases resolve correctly in both directions."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    ALIAS_MAPPINGS = {
        "ts_ewm_mean": "ts_ema",
        "ts_ewma": "ts_ema",
        "ts_sma": "ts_mean",
        "ts_ewm_std": "ewm_std",
        "ts_ewm_var": "ewm_var",
        "ts_lag": "ts_delay",
        "wdecay": "ts_decay_linear",
        "safe_divide": "safe_div_null",
        "div_safe": "safe_div_null",
        "bound": "clip",
        "cs_standardize": "zscore",
    }

    for alias, canonical in ALIAS_MAPPINGS.items():
        alias_op = OperatorRegistry.get(alias)
        canonical_op = OperatorRegistry.get(canonical)
        assert alias_op is canonical_op, f"{alias} should resolve to {canonical}"


def test_alias_count(_loaded):
    """Verify the registry has sufficient aliases registered."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    # Should have at least 141 + 14 = 155 aliases now
    alias_count = len(OperatorRegistry._aliases)
    assert alias_count >= 155, f"Expected ≥155 aliases, got {alias_count}"


@pytest.fixture(scope="module")
def _loaded():
    """Load all operators once per module."""
    from factor_engine.cleaned_operators import load_all
    load_all()
    return True
