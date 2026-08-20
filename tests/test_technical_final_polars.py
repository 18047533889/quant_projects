from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import polars as pl


_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str):
    import cleaned_operators.base_polars as base_polars

    base_polars.register_operator = lambda **_kwargs: lambda cls: cls
    spec = importlib.util.spec_from_file_location(
        name, _ROOT / "cleaned_operators/polars_native/technical_final.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_elder_ray_outputs_respect_warmup_and_component(monkeypatch) -> None:
    module = _load_module("technical_final_semantics_elder")
    high = pl.DataFrame({"x": [11.0, 12.0, 13.0, 14.0, 15.0]})
    low = pl.DataFrame({"x": [9.0, 10.0, 11.0, 12.0, 13.0]})
    close = pl.DataFrame({"x": [10.0, 11.0, 12.0, 13.0, 14.0]})

    bull = module.ElderRay()._calculate_series(high, low, close, ema=3, output="bull")["x"].to_numpy()
    bear = module.ElderRay()._calculate_series(high, low, close, ema=3, output="bear")["x"].to_numpy()
    spread = module.ElderRay()._calculate_series(high, low, close, ema=3, output="spread")["x"].to_numpy()

    assert np.isnan(bull[:2]).all()
    assert np.isnan(bear[:2]).all()
    np.testing.assert_allclose(spread, np.full(5, 2.0))
    np.testing.assert_allclose(bull[2:], [1.75, 1.875, 1.9375])
    np.testing.assert_allclose(bear[2:], [-0.25, -0.125, -0.0625])


def test_tsi_is_causal_and_warms_after_both_emas() -> None:
    module = _load_module("technical_final_semantics_tsi")
    values = np.array([10.0, 11.0, 13.0, 12.0, 15.0, 16.0, 18.0, 17.0, 19.0, 21.0])
    result = module.TSI()._calculate_series(
        pl.DataFrame({"x": values}), long_window=3, short_window=2
    )["x"].to_numpy()
    assert np.isnan(result[:3]).all()
    poisoned = values.copy()
    poisoned[6:] = np.linspace(-100.0, 100.0, len(values) - 6)
    poisoned_result = module.TSI()._calculate_series(
        pl.DataFrame({"x": poisoned}), 3, 2
    )["x"].to_numpy()
    np.testing.assert_allclose(result[:6], poisoned_result[:6], equal_nan=True)


def test_vortex_masks_zero_true_range() -> None:
    module = _load_module("technical_final_semantics_vortex")
    flat = pl.DataFrame({"x": [5.0] * 5})
    result = module.VortexPlus()._calculate_series(flat, flat, flat, period=2)["x"].to_numpy()
    assert np.isnan(result).all()
