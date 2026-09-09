"""Independent regression oracles for V9-M04/M05/M06."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.contracts import ExecutionKind
from factor_engine.backend.operator_errors import OperatorParameterError
from factor_engine.backend.polars_backend_kind import (
    PolarsImplementationKind,
    polars_backend_kind,
)
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.ts_model import wavelet_spectral as ws


def _manual_haar_pairs(values: np.ndarray) -> list[tuple[int, float]]:
    current = np.asarray(values, dtype=float).copy()
    pairs: list[tuple[int, float]] = []
    scale = 2
    while current.size >= 2:
        detail = (current[::2] - current[1::2]) / np.sqrt(2.0)
        pairs.append((scale, float(detail @ detail)))
        current = (current[::2] + current[1::2]) / np.sqrt(2.0)
        scale *= 2
    return list(reversed(pairs))


def _full_fft_low_ratio(values: np.ndarray) -> float:
    centered = values - np.mean(values)
    full = np.abs(np.fft.fft(centered)) ** 2
    n = values.size
    half = max(1, (n // 2 + 1) // 2)
    selected = {0}
    for k in range(1, half):
        selected.update((k, n - k))
    return float(np.sum(full[list(selected)]) / np.sum(full))


def test_haar_scale_energy_pairs_and_linear_slope():
    values = np.arange(128.0)
    pairs = ws._haar_energy(values, 128)
    expected = _manual_haar_pairs(values)
    assert [scale for scale, _ in pairs] == [128, 64, 32, 16, 8, 4, 2]
    assert [scale for scale, _ in pairs] == [scale for scale, _ in expected]
    assert [energy for _, energy in pairs] == pytest.approx(
        [energy for _, energy in expected]
    )
    assert ws._wavelet_stats(values, 128, "slope") == pytest.approx(2.0)


def test_haar_parseval_includes_final_approximation():
    values = np.linspace(-3.0, 7.0, 64)
    pairs = ws._haar_energy(values, 64)
    final_approx_energy = float(values.sum() ** 2 / values.size)
    assert sum(energy for _, energy in pairs) + final_approx_energy == pytest.approx(float(values @ values))


def test_spectral_one_sided_weights_match_full_fft_parseval():
    n = 128
    t = np.arange(n)
    values = np.cos(2.0 * np.pi * t / n) + (-1.0) ** t
    assert ws._spectral_low_ratio(values, n) == pytest.approx(1.0 / 3.0)
    assert ws._spectral_low_ratio(values, n) == pytest.approx(_full_fft_low_ratio(values))


@pytest.mark.parametrize("n", [63, 64, 127, 128])
def test_spectral_odd_even_and_band_boundary(n: int):
    t = np.arange(n)
    values = 0.7 * np.cos(2.0 * np.pi * 3.0 * t / n)
    values += 1.3 * np.cos(2.0 * np.pi * (n // 2) * t / n)
    assert ws._spectral_low_ratio(values, n) == pytest.approx(_full_fft_low_ratio(values), abs=1e-12)


def test_wavelet_public_window_domain_has_no_hidden_expansion():
    for window in (32, 64, 128, 256):
        assert ws._fixed_window_anchor(window) == window
    for window in (31, 60, 100, 127, 257):
        with pytest.raises(ValueError, match="window must be one of"):
            ws._fixed_window_anchor(window)
    frame = pd.DataFrame({"A": np.arange(128.0)})
    with pytest.raises(ValueError, match="got 100"):
        ws._apply(frame, lambda values: ws._wavelet_stats(values, 100, "slope"))
    assert ws._fixed_window_anchor(np.int64(32)) == 32
    for invalid in (32.0, 32.5, np.float64(32), True, np.bool_(True)):
        with pytest.raises(OperatorParameterError):
            ws._fixed_window_anchor(invalid)


def test_actual_polars_backend_kernels_match_reference():
    import factor_engine.cleaned_operators.polars_native.ts_advanced_batch4  # noqa: F401

    n = 128
    t = np.arange(n)
    values = np.cos(2.0 * np.pi * t / n) + (-1.0) ** t
    dates = pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 5, 7), interval="1d", eager=True)[:n]
    second = np.arange(n, dtype=float)
    second[40] = np.nan
    frame = pl.DataFrame({"date": dates, "A": values, "B": second})
    spectral_op = OperatorRegistry.get("ts_spectral_low_frequency_ratio", "polars", mode="any")
    wavelet_op = OperatorRegistry.get("ts_wavelet_energy_slope", "polars", mode="any")
    spectral = spectral_op.calculate(x=frame)
    wavelet = wavelet_op.calculate(x=frame)
    assert spectral_op.metadata.param_names == ["x", "window"]
    assert wavelet_op.metadata.param_names == ["x", "window"]
    assert spectral_op.metadata.param_specs["window"].default == 128
    assert wavelet_op.metadata.param_specs["window"].default == 128
    assert spectral.columns == frame.columns
    assert wavelet.columns == frame.columns
    assert spectral["date"].equals(frame["date"])
    assert spectral["A"][-1] == pytest.approx(1.0 / 3.0)
    assert wavelet["A"][-1] == pytest.approx(ws._wavelet_stats(values, n, "slope"))
    assert spectral["A"][: n - 1].is_nan().sum() == n - 1
    assert spectral["B"][-1] != spectral["B"][-1]

    for op in (spectral_op, wavelet_op):
        # Normal bootstrap can retain rolling_pack's authoritative CPU bridge;
        # explicit batch4 registration is separately checked below. Neither
        # path is native, and neither is production eligible.
        if type(op).__module__.endswith("rolling_pack"):
            assert op._physical_spec.execution_kind == ExecutionKind.POLARS_PANDAS_DELEGATE
            expected_kind = PolarsImplementationKind.POLARS_UDF_PANDAS_DELEGATE
        else:
            assert type(op).__module__.endswith("polars_native.ts_advanced_batch4")
            assert op._physical_spec.execution_kind == ExecutionKind.DELEGATE_PYTHON
            assert op._physical_spec.validation_errors() == ()
            expected_kind = PolarsImplementationKind.UNSUPPORTED
        assert op._physical_spec.is_production_eligible() is False
        assert polars_backend_kind(op, production_mode=True) == expected_kind
        assert polars_backend_kind(op, production_mode=False) == expected_kind

    long = pl.DataFrame({
        "date": [dates[0], dates[0]],
        "stock_code": ["A", "B"],
        "value": [1.0, 2.0],
    })
    expected_rejection = ("duplicate values" if type(spectral_op).__module__.endswith("rolling_pack")
                          else "single-stock")
    with pytest.raises(ValueError, match=expected_rejection):
        spectral_op.calculate(long, window=2)
