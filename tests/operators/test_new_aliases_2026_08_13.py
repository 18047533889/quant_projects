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

    # ewm_* family
    assert OperatorRegistry.get("ts_ewm_std") is OperatorRegistry.get("ewm_std")
    assert OperatorRegistry.get("ts_ewm_var") is OperatorRegistry.get("ewm_var")
    assert OperatorRegistry.get("ts_ewm_cov") is OperatorRegistry.get("ewm_cov")
    assert OperatorRegistry.get("ts_ewm_corr") is OperatorRegistry.get("ewm_corr")


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

    assert OperatorRegistry.get("ret") is OperatorRegistry.get("ts_pct")


def test_clip_aliases(_loaded):
    """Clipping aliases map to clip canonical."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    assert OperatorRegistry.get("bound") is OperatorRegistry.get("clip")


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
        "ts_ewm_cov": "ewm_cov",
        "ts_ewm_corr": "ewm_corr",
        "ts_lag": "ts_delay",
        "wdecay": "ts_decay_linear",
        "safe_divide": "safe_div_null",
        "div_safe": "safe_div_null",
        "ret": "ts_pct",
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
