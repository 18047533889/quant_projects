"""ADX replacement kernel: independent semantics and physical admission."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from factor_engine.backend.contracts import CapabilityLevel, ExecutionKind
from factor_engine.backend.operator_capability import (
    capability_for,
    production_eligible_backends,
    supports_polars,
)
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _independent_adx(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int):
    high = high.replace([np.inf, -np.inf], np.nan).astype(float)
    low = low.replace([np.inf, -np.inf], np.nan).astype(float)
    close = close.replace([np.inf, -np.inf], np.nan).astype(float)
    previous_close = close.shift(1)
    true_range = pd.DataFrame(
        np.maximum.reduce([
            (high - low).to_numpy(),
            (high - previous_close).abs().to_numpy(),
            (low - previous_close).abs().to_numpy(),
        ]),
        index=high.index,
        columns=high.columns,
    )
    raw_plus = high - high.shift(1)
    raw_minus = low.shift(1) - low
    plus = raw_plus.where((raw_plus > raw_minus) & (raw_plus > 0), 0.0)
    minus = raw_minus.where((raw_minus > raw_plus) & (raw_minus > 0), 0.0)
    atr = true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    pdm = plus.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    mdm = minus.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    pdi = 100 * pdm / atr.mask(atr.abs() <= 1e-12)
    mdi = 100 * mdm / atr.mask(atr.abs() <= 1e-12)
    denominator = pdi + mdi
    dx = 100 * (pdi - mdi).abs() / denominator.mask(denominator.abs() <= 1e-12)
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _panels(values: dict[str, np.ndarray]):
    index = pd.date_range("2024-01-01", periods=len(next(iter(values.values()))), name="timestamp")
    close = pd.DataFrame(values, index=index)
    high = close + 1.25
    low = close - 1.0
    return high, low, close


def _run_backend(backend: str, high, low, close, window=3):
    load_all()
    op = OperatorRegistry.get("ADX", backend, mode="any")
    if backend == "pandas_numpy":
        return op.calculate(high, low, close, window)
    timestamp = pl.Series("timestamp", high.index)
    to_polars = lambda frame: pl.DataFrame({
        "timestamp": timestamp,
        **{column: frame[column].to_numpy() for column in frame.columns},
    })
    out = op.calculate(to_polars(high), to_polars(low), to_polars(close), window)
    assert out["timestamp"].to_list() == list(high.index.to_pydatetime())
    return pd.DataFrame(
        {column: out[column].to_numpy() for column in high.columns},
        index=high.index,
    )


def test_adx_warmup_zero_denominator_and_multistock_reference():
    high, low, close = _panels({
        "trend": np.arange(1.0, 25.0),
        "flat": np.full(24, 7.0),
        "reverse": np.arange(25.0, 1.0, -1.0),
    })
    expected = _independent_adx(high, low, close, 3)
    actual = _run_backend("polars", high, low, close, 3)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)
    assert expected["trend"].iloc[:5].isna().all()
    assert expected["trend"].iloc[5] == 100.0
    assert expected["flat"].isna().all()


def test_adx_nan_inf_recovery_and_public_backend_parity():
    base = np.arange(1.0, 29.0)
    a, b = base.copy(), base[::-1].copy()
    a[4], a[12] = np.nan, np.inf
    b[7], b[16] = -np.inf, np.nan
    high, low, close = _panels({"A": a, "B": b})
    expected = _independent_adx(high, low, close, 3)
    polars_result = _run_backend("polars", high, low, close, 3)
    pandas_result = _run_backend("pandas_numpy", high, low, close, 3)
    pd.testing.assert_frame_equal(polars_result, expected, check_exact=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(polars_result, pandas_result, check_exact=False, rtol=1e-12, atol=1e-12)
    assert polars_result["A"].iloc[-1] == 100.0
    assert polars_result["B"].iloc[-1] == 100.0


def test_adx_prefix_causality_and_stock_isolation():
    rng = np.random.default_rng(56)
    high, low, close = _panels({
        "A": np.cumsum(rng.normal(size=40)) + 100,
        "B": np.cumsum(rng.normal(size=40)) + 200,
    })
    full = _run_backend("polars", high, low, close, 5)
    prefix = _run_backend("polars", high.iloc[:27], low.iloc[:27], close.iloc[:27], 5)
    pd.testing.assert_frame_equal(full.iloc[:27], prefix, check_exact=False, rtol=1e-12, atol=1e-12)

    changed = close.copy()
    changed["B"] += np.linspace(0, 1000, len(changed))
    isolated = _run_backend("polars", high.assign(B=changed["B"] + 1.25), low.assign(B=changed["B"] - 1), changed, 5)
    pd.testing.assert_series_equal(full["A"], isolated["A"])


def test_adx_physical_spec_and_public_auto_gate_are_honest():
    load_all()
    op = OperatorRegistry.get("ADX", "polars", mode="any")
    spec = op._physical_spec
    assert spec.execution_kind is ExecutionKind.POLARS_NATIVE_EXPR
    assert spec.stateful and spec.requires_sorted and spec.materializes_full_panel
    assert not spec.supports_lazy and not spec.supports_streaming
    assert spec.supports_nulls and spec.supports_nan and spec.supports_inf
    assert not spec.validation_errors()
    assert all(
        len(getattr(spec, name)) == 64
        for name in (
            "implementation_source_hash",
            "parameter_domain_hash",
            "semantic_contract_hash",
            "implementation_closure_hash",
        )
    )
    capability = capability_for("ADX", "polars")
    # The implementation is now honestly classified, but public production
    # admission remains closed until the separate primitive evidence artifact
    # certifies parity, edges and no-fallback for this exact implementation.
    assert capability.level is CapabilityLevel.IMPLEMENTED
    assert capability.execution_kind is ExecutionKind.POLARS_NATIVE_EXPR
    assert capability.materializes_full_panel
    assert not capability.supports_lazy and not capability.supports_streaming
    assert not supports_polars("ADX", mode="production")
    assert "polars" not in production_eligible_backends("ADX")
