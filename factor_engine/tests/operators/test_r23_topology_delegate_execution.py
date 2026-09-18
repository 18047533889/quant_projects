"""Real execution coverage for advanced-topology Polars delegates."""

from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.contracts import ExecutionKind
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry
@dataclass(frozen=True)
class Case:
    name: str
    params: dict[str, Any]
    invalid: dict[str, Any]
    module: str
    class_name: str


CASES = (
    Case(
        "ts_persistence_diagram_shift",
        {"window": 24, "tau": 1, "embedding_dim": 3},
        {"window": 6, "tau": 2, "embedding_dim": 3},
        "factor_engine.cleaned_operators.polars_native.ts_advanced_batch3",
        "TSPersistenceDiagramShiftPolarsNative",
    ),
    Case(
        "ts_fisher_information_shift",
        {"recent_window": 12, "prior_window": 12},
        {"recent_window": 7},
        "factor_engine.cleaned_operators.polars_native.ts_advanced_batch5",
        "TSFisherInformationShiftPolarsNative",
    ),
)


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(2303)
    t = np.arange(90, dtype=float)
    return pd.DataFrame({"A": np.sin(t / 3.0) + 0.2*np.sin(t / 7.0) + 0.06*rng.normal(size=90)})


def _direct_class(case: Case):
    return getattr(importlib.import_module(case.module), case.class_name)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_direct_batch_and_final_registry_match_canonical(case: Case) -> None:
    pl = pytest.importorskip("polars")
    load_all()
    frame = _frame()
    reference = OperatorRegistry.get(case.name, "pandas_numpy", mode="any")
    final = OperatorRegistry.get(case.name, "polars", mode="any")
    expected = reference.calculate(frame, **case.params).to_numpy()
    assert np.isfinite(expected).sum() > 0

    direct_class = _direct_class(case)
    for operator in (direct_class(), final):
        actual = operator.calculate(pl.from_pandas(frame), **case.params).to_numpy()
        assert np.isfinite(actual).sum() == np.isfinite(expected).sum()
        np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12)

        cut = 80
        changed = frame.copy()
        changed.iloc[cut:, 0] += 1000.0
        future_changed = operator.calculate(pl.from_pandas(changed), **case.params).to_numpy()
        np.testing.assert_allclose(future_changed[:cut], actual[:cut], equal_nan=True, rtol=1e-12, atol=1e-12)

    assert direct_class().physical_spec().execution_kind is ExecutionKind.POLARS_PANDAS_DELEGATE


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_direct_batch_nan_inf_and_invalid_match_canonical(case: Case) -> None:
    pl = pytest.importorskip("polars")
    load_all()
    frame = _frame()
    frame.iloc[9, 0] = np.nan
    frame.iloc[28, 0] = np.inf
    frame.iloc[51, 0] = -np.inf
    reference = OperatorRegistry.get(case.name, "pandas_numpy", mode="any")
    direct = _direct_class(case)()
    expected = reference.calculate(frame, **case.params).to_numpy()
    actual = direct.calculate(pl.from_pandas(frame), **case.params).to_numpy()
    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12)

    bad = dict(case.params)
    bad.update(case.invalid)
    errors = []
    for operator, data in ((reference, frame), (direct, pl.from_pandas(frame))):
        with pytest.raises((TypeError, ValueError)) as caught:
            operator.calculate(data, **bad)
        errors.append((type(caught.value), str(caught.value)))
    assert errors[1] == errors[0]
