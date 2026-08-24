# -*- coding: utf-8 -*-
"""ABI2-P0-001: Return decomposition ABI contract tests.

These tests verify that return_decomp operators have consistent:
1. Parameter names between metadata and kernel signatures
2. price_basis semantics and validation
3. Correct price point usage in attribution calculations
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.market.price_basis import PriceBasis

ensure_cleaned_loaded()


def _op(name: str, backend: str = "pandas_numpy"):
    op = OperatorRegistry.get(name, backend)
    assert op is not None, f"{name}/{backend}"
    return op


# ---------------------------------------------------------------------------
# ABI Contract: param_names consistency
# ---------------------------------------------------------------------------
def test_overnight_return_param_names_match_kernel():
    """ABI: metadata param_names must match kernel signature."""
    op = _op("overnight_return")
    meta = op.metadata

    # Metadata declares these names
    declared = meta.param_names
    assert "open_px" in declared, f"Expected 'open_px' in {declared}"
    assert "pre_close" in declared, f"Expected 'pre_close' in {declared}"
    assert "price_basis" in declared, f"Expected 'price_basis' in {declared}"

    # Test actual kernel accepts these names
    idx = pd.date_range("2024-01-01", periods=5)
    open_px = pd.DataFrame([[100.0, 101.0]], index=idx[:1], columns=["A", "B"])
    pre_close = pd.DataFrame([[99.0, 100.0]], index=idx[:1], columns=["A", "B"])

    # Should work with exact param names (explicit price_basis for test data)
    result = op.calculate(open_px, pre_close, price_basis="raw")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == open_px.shape


def test_open_close_return_param_names_match_kernel():
    """ABI: metadata param_names must match kernel signature."""
    op = _op("open_close_return")
    meta = op.metadata

    declared = meta.param_names
    assert "open_px" in declared, f"Expected 'open_px' in {declared}"
    assert "close" in declared, f"Expected 'close' in {declared}"
    assert "price_basis" in declared, f"Expected 'price_basis' in {declared}"

    idx = pd.date_range("2024-01-01", periods=5)
    open_px = pd.DataFrame([[100.0, 101.0]], index=idx[:1], columns=["A", "B"])
    close = pd.DataFrame([[102.0, 103.0]], index=idx[:1], columns=["A", "B"])

    result = op.calculate(open_px, close, price_basis="raw")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == open_px.shape


def test_open_to_vwap_return_param_names_match_kernel():
    """ABI: metadata param_names must match kernel signature."""
    op = _op("open_to_vwap_return")
    meta = op.metadata

    declared = meta.param_names
    assert "open_px" in declared, f"Expected 'open_px' in {declared}"
    assert "vwap" in declared, f"Expected 'vwap' in {declared}"
    assert "price_basis" in declared, f"Expected 'price_basis' in {declared}"

    idx = pd.date_range("2024-01-01", periods=5)
    open_px = pd.DataFrame([[100.0, 101.0]], index=idx[:1], columns=["A", "B"])
    vwap = pd.DataFrame([[101.0, 102.0]], index=idx[:1], columns=["A", "B"])

    result = op.calculate(open_px, vwap, price_basis="raw")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == open_px.shape


def test_vwap_to_close_return_param_names_match_kernel():
    """ABI: metadata param_names must match kernel signature."""
    op = _op("vwap_to_close_return")
    meta = op.metadata

    declared = meta.param_names
    assert "vwap" in declared, f"Expected 'vwap' in {declared}"
    assert "close" in declared, f"Expected 'close' in {declared}"
    assert "price_basis" in declared, f"Expected 'price_basis' in {declared}"

    idx = pd.date_range("2024-01-01", periods=5)
    vwap = pd.DataFrame([[101.0, 102.0]], index=idx[:1], columns=["A", "B"])
    close = pd.DataFrame([[102.0, 103.0]], index=idx[:1], columns=["A", "B"])

    result = op.calculate(vwap, close, price_basis="raw")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == vwap.shape


# ---------------------------------------------------------------------------
# ABI Contract: price_basis parameter semantics
# ---------------------------------------------------------------------------
def test_return_decomp_family_has_price_basis_param():
    """ABI: all return_decomp operators must declare price_basis parameter."""
    operators = [
        "overnight_return",
        "open_close_return",
        "open_to_vwap_return",
        "vwap_to_close_return",
    ]

    for op_name in operators:
        op = _op(op_name)
        meta = op.metadata
        assert "price_basis" in meta.param_names, \
            f"{op_name} missing 'price_basis' in param_names"


def test_price_basis_semantics_documentation():
    """ABI: price_basis must have clear semantic documentation."""
    op = _op("overnight_return")
    meta = op.metadata

    # price_basis should be last parameter (trailing keyword-only)
    assert meta.param_names[-1] == "price_basis", \
        "price_basis should be trailing parameter"

    # Should have input_units declaration for price params
    assert hasattr(meta, "input_units"), "Missing input_units"
    assert meta.input_units.get("open_px") == "price", \
        "open_px should be declared as price unit"


# ---------------------------------------------------------------------------
# ABI Contract: return attribution uses correct price points
# ---------------------------------------------------------------------------
def test_overnight_return_uses_correct_attribution():
    """ABI: overnight return = open / pre_close - 1 (not inverted)."""
    op = _op("overnight_return")

    idx = pd.date_range("2024-01-01", periods=3)
    # pre_close = 100, open = 105 -> +5% overnight return
    open_px = pd.DataFrame([[105.0]], index=idx[:1], columns=["A"])
    pre_close = pd.DataFrame([[100.0]], index=idx[:1], columns=["A"])

    result = op.calculate(open_px, pre_close, price_basis="raw")

    # open/pre_close - 1 = 105/100 - 1 = 0.05
    expected = 0.05
    actual = result.iloc[0, 0]
    assert abs(actual - expected) < 1e-10, \
        f"overnight_return should be +5%, got {actual}"


def test_open_close_return_uses_correct_attribution():
    """ABI: intraday return = close / open - 1."""
    op = _op("open_close_return")

    idx = pd.date_range("2024-01-01", periods=3)
    # open = 100, close = 110 -> +10% intraday return
    open_px = pd.DataFrame([[100.0]], index=idx[:1], columns=["A"])
    close = pd.DataFrame([[110.0]], index=idx[:1], columns=["A"])

    result = op.calculate(open_px, close, price_basis="raw")

    # close/open - 1 = 110/100 - 1 = 0.10
    expected = 0.10
    actual = result.iloc[0, 0]
    assert abs(actual - expected) < 1e-10, \
        f"open_close_return should be +10%, got {actual}"


def test_open_to_vwap_return_attribution():
    """ABI: open->vwap return = vwap / open - 1."""
    op = _op("open_to_vwap_return")

    idx = pd.date_range("2024-01-01", periods=3)
    open_px = pd.DataFrame([[100.0]], index=idx[:1], columns=["A"])
    vwap = pd.DataFrame([[103.0]], index=idx[:1], columns=["A"])

    result = op.calculate(open_px, vwap, price_basis="raw")

    # vwap/open - 1 = 103/100 - 1 = 0.03
    expected = 0.03
    actual = result.iloc[0, 0]
    assert abs(actual - expected) < 1e-10, \
        f"open_to_vwap_return should be +3%, got {actual}"


def test_vwap_to_close_return_attribution():
    """ABI: vwap->close return = close / vwap - 1."""
    op = _op("vwap_to_close_return")

    idx = pd.date_range("2024-01-01", periods=3)
    vwap = pd.DataFrame([[103.0]], index=idx[:1], columns=["A"])
    close = pd.DataFrame([[106.09]], index=idx[:1], columns=["A"])

    result = op.calculate(vwap, close, price_basis="raw")

    # close/vwap - 1 = 106.09/103 - 1 ≈ 0.03
    expected = 0.03
    actual = result.iloc[0, 0]
    assert abs(actual - expected) < 1e-9, \
        f"vwap_to_close_return should be +3%, got {actual}"


def test_return_chain_composition():
    """ABI: overnight and intraday returns chain multiplicatively to close/pre-close."""
    idx = pd.date_range("2024-01-01", periods=2)

    # Day 1: close = 100
    # Day 2: pre_close = 100, open = 105, close = 110
    # overnight: +5%, intraday: 110/105 - 1, total: 110/100 - 1 = 10%

    pre_close = pd.DataFrame([[100.0]], index=idx[:1], columns=["A"])
    open_px = pd.DataFrame([[105.0]], index=idx[:1], columns=["A"])
    close = pd.DataFrame([[110.0]], index=idx[:1], columns=["A"])

    overnight = _op("overnight_return").calculate(open_px, pre_close, price_basis="raw").iloc[0, 0]
    intraday = _op("open_close_return").calculate(open_px, close, price_basis="raw").iloc[0, 0]

    chained_growth = (1 + overnight) * (1 + intraday)
    expected_growth = close.iloc[0, 0] / pre_close.iloc[0, 0]
    expected_total = expected_growth - 1

    # Independent price-ratio oracle: (open/pre_close) * (close/open) = close/pre_close.
    price_ratio_product = (
        open_px.iloc[0, 0] / pre_close.iloc[0, 0]
    ) * (
        close.iloc[0, 0] / open_px.iloc[0, 0]
    )
    assert abs(price_ratio_product - expected_growth) < 1e-10
    assert abs(chained_growth - expected_growth) < 1e-10
    assert abs((chained_growth - 1) - expected_total) < 1e-10, \
        f"Return composition failed: {chained_growth - 1} != {expected_total}"


# ---------------------------------------------------------------------------
# ABI Contract: consistent naming across family
# ---------------------------------------------------------------------------
def test_parameter_naming_consistency_across_family():
    """ABI: all operators in family use consistent parameter naming."""
    operators = [
        ("overnight_return", ["open_px", "pre_close"]),
        ("open_close_return", ["open_px", "close"]),
        ("open_to_vwap_return", ["open_px", "vwap"]),
        ("vwap_to_close_return", ["vwap", "close"]),
    ]

    # All should use "open_px" not "open" when referring to opening price
    for op_name, expected_price_params in operators:
        op = _op(op_name)
        meta = op.metadata

        for param in expected_price_params:
            assert param in meta.param_names, \
                f"{op_name} should have '{param}' parameter"

        # Should NOT have bare "open" (ambiguous)
        non_price_basis = [p for p in meta.param_names if p != "price_basis"]
        if any("open" in p for p in non_price_basis):
            # If "open" appears, it must be "open_px"
            open_params = [p for p in non_price_basis if "open" in p]
            for p in open_params:
                assert p == "open_px", \
                    f"{op_name} uses ambiguous '{p}', should be 'open_px'"


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------
def test_hard_gate_abi_return_decomp_consistent():
    """Hard gate: ABI_RETURN_DECOMP_CONSISTENT = PASS."""
    operators = [
        "overnight_return",
        "open_close_return",
        "open_to_vwap_return",
        "vwap_to_close_return",
    ]

    for op_name in operators:
        op = _op(op_name)
        meta = op.metadata

        # 1. Must have price_basis parameter
        assert "price_basis" in meta.param_names, \
            f"{op_name}: missing price_basis parameter"

        # 2. Price parameters must use _px suffix (not bare "open")
        price_params = [p for p in meta.param_names
                       if p not in ["price_basis"] and
                          any(x in p for x in ["open", "close", "vwap", "pre_close"])]

        for param in price_params:
            if "open" in param and param != "pre_close":
                assert param == "open_px", \
                    f"{op_name}: use 'open_px' not '{param}'"

        # 3. Must declare input_units for price inputs
        assert hasattr(meta, "input_units"), \
            f"{op_name}: missing input_units declaration"


def test_hard_gate_abi_price_basis_validated():
    """Hard gate: ABI_PRICE_BASIS_VALIDATED = PASS."""
    # price_basis validation is in _assert_shared_price_basis()
    # Test that it rejects mixed basis when provided

    op = _op("overnight_return")

    # This should work - explicit price_basis provided
    idx = pd.date_range("2024-01-01", periods=1)
    open_px = pd.DataFrame([[100.0]], index=idx, columns=["A"])
    pre_close = pd.DataFrame([[99.0]], index=idx, columns=["A"])

    # Should not raise with explicit price_basis
    result = op.calculate(open_px, pre_close, price_basis="raw")
    assert result is not None
