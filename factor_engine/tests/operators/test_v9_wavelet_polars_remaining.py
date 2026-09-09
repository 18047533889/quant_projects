"""Registry-level regression tests for the three remaining Haar bridges."""
from __future__ import annotations

import math
import json
import os
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.backend.contracts import ExecutionKind
from factor_engine.backend.polars_backend_kind import PolarsImplementationKind, polars_backend_kind
from factor_engine.cleaned_operators.registry import OperatorRegistry


NAMES = (
    "ts_wavelet_entropy",
    "ts_wavelet_low_frequency_ratio",
    "ts_wavelet_high_frequency_ratio",
)


def _manual_energies(values: np.ndarray) -> list[float]:
    current = np.asarray(values, dtype=float).copy()
    energies: list[float] = []
    while current.size >= 2:
        detail = (current[::2] - current[1::2]) / math.sqrt(2.0)
        energies.append(float(detail @ detail))
        current = (current[::2] + current[1::2]) / math.sqrt(2.0)
    return list(reversed(energies))


def _manual_stats(values: np.ndarray) -> tuple[float, float, float]:
    energies = _manual_energies(values)
    total = sum(energies)
    weights = np.asarray([energy / total for energy in energies if energy > 0.0])
    entropy = 0.0 if weights.size == 1 else float(-(weights * np.log(weights)).sum() / np.log(weights.size))
    return entropy, energies[0] / total, energies[-1] / total


def _operators():
    ensure_cleaned_loaded()
    result = [OperatorRegistry.get(name, "polars", mode="any") for name in NAMES]
    assert all(op is not None for op in result)
    return result


def test_registry_defaults_specs_and_honest_classification():
    for name, op in zip(NAMES, _operators(), strict=True):
        assert op.metadata.param_names == ["x", "window"]
        spec = op.metadata.param_specs["window"]
        assert spec.choices == (32, 64, 128, 256)
        assert spec.default == 128
        assert op._physical_spec.execution_kind in {
            ExecutionKind.DELEGATE_PYTHON,
            ExecutionKind.POLARS_PANDAS_DELEGATE,
        }
        assert op._physical_spec.is_production_eligible() is False
        assert polars_backend_kind(op, production_mode=True) in {
            PolarsImplementationKind.UNSUPPORTED,
            PolarsImplementationKind.POLARS_UDF_PANDAS_DELEGATE,
        }
        assert polars_backend_kind(op, production_mode=False) != PolarsImplementationKind.POLARS_NATIVE


@pytest.mark.parametrize(
    "values, expected",
    [
        (np.r_[np.ones(64), np.full(64, 2.0)], (0.0, 1.0, 0.0)),
        ((-1.0) ** np.arange(128), (0.0, 0.0, 1.0)),
        (np.arange(128.0) + 0.25 * ((-1.0) ** np.arange(128)), None),
    ],
)
def test_distinct_analytic_inputs_against_independent_haar(values, expected):
    entropy, low, high = _operators()
    dates = pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 5, 7), interval="1d", eager=True)[:128]
    frame = pl.DataFrame({"date": dates, "A": values})
    actual = tuple(op.calculate(x=frame)["A"][-1] for op in (entropy, low, high))
    oracle = expected if expected is not None else _manual_stats(values)
    assert actual == pytest.approx(oracle, abs=1e-12)


def test_prefix_missing_and_append_future_parity():
    entropy, low, high = _operators()
    rng = np.random.default_rng(91)
    prefix = rng.normal(size=128)
    full = np.r_[prefix, rng.normal(size=20)]
    frame = pl.DataFrame({"A": full, "B": np.r_[prefix[:70], np.nan, prefix[71:], rng.normal(size=20)]})
    for op in (entropy, low, high):
        result = op.calculate(x=frame)
        prefix_result = op.calculate(x=pl.DataFrame({"A": prefix}))
        assert result["A"][127] == pytest.approx(prefix_result["A"][-1])
        assert result["A"][:127].is_nan().sum() == 127
        assert result["B"][127] != result["B"][127]


def test_strict_window_and_long_panel_rejection():
    op = _operators()[0]
    frame = pl.DataFrame({"A": np.arange(128.0)})
    # The public binder canonicalizes an integral float before the strict kernel
    # gate; fractional floats, bools and values outside the discrete domain fail.
    assert op.calculate(x=frame, window=32.0).height == 128
    for invalid in (32.5, True, 100):
        with pytest.raises(Exception):
            op.calculate(x=frame, window=invalid)
    long = pl.DataFrame({"date": [1, 1], "stock_code": ["A", "B"], "value": [1.0, 2.0]})
    with pytest.raises(ValueError):
        op.calculate(x=long)


def test_explicit_legacy_batch4_startup_selects_correct_cpu_classes():
    code = r'''\
import hashlib, inspect, json, numpy as np, polars as pl
import factor_engine.cleaned_operators.ts_model.wavelet_spectral
import factor_engine.cleaned_operators.polars_native.ts_advanced_batch4 as batch4
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.backend.polars_backend_kind import polars_backend_kind
names = ("ts_wavelet_entropy", "ts_wavelet_low_frequency_ratio", "ts_wavelet_high_frequency_ratio")
values = np.r_[np.ones(64), np.full(64, 2.0)]
frame = pl.DataFrame({"date": list(range(128)), "A": values})
rows = []
for name in names:
    op = OperatorRegistry.get(name, "polars", mode="any")
    out = op.calculate(x=frame)
    rows.append({
        "name": name,
        "module": type(op).__module__,
        "value": out["A"][-1],
        "columns": out.columns,
        "kind": op._physical_spec.execution_kind.value,
        "backend_kind": polars_backend_kind(op, production_mode=False).value,
        "eligible": op._physical_spec.is_production_eligible(),
        "source_match": op._physical_spec.implementation_source_hash == hashlib.sha256(
            (inspect.getsource(batch4._wavelet_spectral_cpu_wide) + inspect.getsource(type(op)._calculate_series)).encode()
        ).hexdigest(),
    })
print(json.dumps(rows))
'''
    env = dict(os.environ)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    rows = json.loads(completed.stdout.strip().splitlines()[-1])
    assert [row["value"] for row in rows] == pytest.approx([0.0, 1.0, 0.0])
    for row in rows:
        assert row["module"].endswith("polars_native.ts_advanced_batch4")
        assert row["columns"] == ["date", "A"]
        assert row["kind"] == ExecutionKind.DELEGATE_PYTHON.value
        assert row["backend_kind"] == PolarsImplementationKind.UNSUPPORTED.value
        assert row["eligible"] is False
        assert row["source_match"] is True
