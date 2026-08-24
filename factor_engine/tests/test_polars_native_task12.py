from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import polars as pl


_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_technical_final_imports_with_authoritative_metadata_abi(monkeypatch) -> None:
    import factor_engine.cleaned_operators.base_polars as base_polars

    monkeypatch.setattr(base_polars, "register_operator", lambda **_kwargs: lambda cls: cls)
    module = _load_module(
        "_task12_technical_final",
        "cleaned_operators/polars_native/technical_final.py",
    )

    assert module.ElderRay.metadata.name == "ElderRay"
    assert set(vars(module.ElderRay.metadata)) <= set(base_polars.OperatorMetadata.__dataclass_fields__)


def test_ts_leverage_effect_is_causal_and_matches_historical_definition(monkeypatch) -> None:
    import factor_engine.cleaned_operators.base as base

    monkeypatch.setattr(base, "register_operator", lambda **_kwargs: lambda cls: cls)
    module = _load_module(
        "_task12_ts_advanced_batch5",
        "cleaned_operators/polars_native/ts_advanced_batch5.py",
    )
    op = module.TSLeverageEffectPolarsNative()

    values = np.array(
        [100.0, 101.0, 99.0, 103.0, 102.0, 106.0, 104.0, 108.0,
         107.0, 111.0, 109.0, 114.0, 112.0, 117.0, 115.0, 120.0,
         118.0, 123.0, 121.0, 126.0, 124.0, 129.0, 127.0, 132.0],
        dtype=float,
    )
    window = 10
    result = op._calculate_series(pl.Series("x", values), window).to_numpy()
    ret = pl.Series("x", values).diff()
    lagged_ret = ret.shift(1)
    trailing_vol = ret.rolling_std(window)
    covariance = (lagged_ret * trailing_vol).rolling_mean(window) - (
        lagged_ret.rolling_mean(window) * trailing_vol.rolling_mean(window)
    )
    expected = (
        covariance
        / (
            lagged_ret.rolling_std(window, ddof=0)
            * trailing_vol.rolling_std(window, ddof=0)
        )
    ).to_numpy()
    np.testing.assert_allclose(result, expected, equal_nan=True)

    prefix_end = 19
    poisoned = values.copy()
    poisoned[prefix_end:] = np.linspace(-1.0e9, 1.0e9, len(values) - prefix_end)
    poisoned_result = op._calculate_series(pl.Series("x", poisoned), window).to_numpy()
    np.testing.assert_allclose(
        result[:prefix_end], poisoned_result[:prefix_end], equal_nan=True
    )
    np.testing.assert_array_equal(
        np.isnan(result[:prefix_end]), np.isnan(poisoned_result[:prefix_end])
    )

    nan_poisoned = values.copy()
    nan_poisoned[prefix_end:] = np.nan
    nan_result = op._calculate_series(pl.Series("x", nan_poisoned), window).to_numpy()
    np.testing.assert_allclose(result[:prefix_end], nan_result[:prefix_end], equal_nan=True)
    np.testing.assert_array_equal(
        np.isnan(result[:prefix_end]), np.isnan(nan_result[:prefix_end])
    )
