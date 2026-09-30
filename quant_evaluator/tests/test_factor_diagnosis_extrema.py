"""Constant diagnosis must not depend on floating-point variance range."""
import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.diagnosis.factor import diagnose_factor


@pytest.mark.parametrize("values,validity,expected", [
    ([1e-200, 2e-200], None, False),
    ([1e308, 1e308], None, True),
    ([3.0, 3.0], None, True),
    ([3.0, 4.0], None, False),
    ([3.0, 4.0], [True, False], True),
    ([np.nan, np.inf], None, True),
    ([3.0, np.nan], None, True),
])
def test_constant_diagnosis_uses_finite_valid_extrema(values, validity, expected):
    batch = FactorBatch(
        ("factor",), AxisRef("time", "int64", 1, np.array([0], dtype=np.int64)),
        AxisRef("asset", "int64", 2, np.arange(2, dtype=np.int64)),
        np.asarray(values, dtype=np.float64).reshape(1, 2, 1),
        validity=None if validity is None else np.asarray(validity).reshape(1, 2, 1),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        diagnosis = diagnose_factor(batch)
    assert bool(diagnosis.is_constant) is expected
    assert ("Factor is constant" in diagnosis.warnings) is expected
    finite = np.asarray(values)[np.isfinite(values)]
    if validity is not None:
        finite = np.asarray(values)[np.isfinite(values) & np.asarray(validity)]
    assert diagnosis.num_valid_observations == len(finite)
    assert diagnosis.num_missing == 2 - len(finite)
