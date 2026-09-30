"""Constant diagnosis must not depend on floating-point variance range."""
import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.diagnosis.factor import diagnose_all_factors, diagnose_factor


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


@pytest.mark.parametrize("values,expected", [
    ([1e308, 1e308], 1e308),
    ([-1e308, -1e308], -1e308),
    ([1e308, -1e308], 0.0),
])
def test_diagnosis_mean_of_finite_values_stays_finite(values, expected):
    batch = FactorBatch(
        ("factor",), AxisRef("time", "int64", 1, np.array([0], dtype=np.int64)),
        AxisRef("asset", "int64", 2, np.arange(2, dtype=np.int64)),
        np.asarray(values, dtype=np.float64).reshape(1, 2, 1),
    )
    assert diagnose_factor(batch).mean_value == expected


def _legacy_diagnosis(batch, factor_idx):
    """Reference the former flatten-based calculation for exact parity."""
    values = batch.values[:, :, factor_idx].flatten()
    finite_mask = np.isfinite(values)
    has_nans = np.any(np.isnan(values))
    has_infs = np.any(np.isinf(values))
    if batch.validity is not None:
        valid_mask = finite_mask & batch.validity[:, :, factor_idx].flatten()
    else:
        valid_mask = finite_mask
    num_valid = int(np.sum(valid_mask))
    num_total = len(values)
    valid_values = values[valid_mask]
    if num_valid:
        min_value = float(np.min(valid_values))
        max_value = float(np.max(valid_values))
        is_constant = min_value == max_value
        with np.errstate(over="ignore", invalid="ignore"):
            mean_value = float(np.mean(valid_values))
        if not np.isfinite(mean_value):
            scale = max(abs(min_value), abs(max_value))
            mean_value = float(np.mean(valid_values / scale) * scale)
    else:
        min_value = max_value = mean_value = None
        is_constant = True
    warnings = []
    coverage = num_valid / num_total if num_total else 0.0
    if coverage < 0.5:
        warnings.append(f"Low coverage: {coverage:.2%}")
    if is_constant:
        warnings.append("Factor is constant")
    if has_nans:
        warnings.append("Contains NaN values")
    if has_infs:
        warnings.append("Contains Inf values")
    return (num_valid, num_total - num_valid, coverage, is_constant, has_nans,
            has_infs, min_value, max_value, mean_value, tuple(warnings))


def _diagnosis_tuple(diagnosis):
    return (diagnosis.num_valid_observations, diagnosis.num_missing,
            diagnosis.coverage, diagnosis.is_constant, diagnosis.has_nans,
            diagnosis.has_infs, diagnosis.min_value, diagnosis.max_value,
            diagnosis.mean_value, diagnosis.warnings)


def test_multifactor_diagnosis_matches_legacy_flatten_at_boundaries():
    values = np.array([
        [[1.0, np.nan, np.nan, 1e308],
         [2.0, np.inf, np.inf, 1e308],
         [3.0, 4.0, -np.inf, np.nan]],
        [[4.0, 4.0, np.nan, np.nan],
         [5.0, 4.0, np.nan, np.nan],
         [6.0, 4.0, np.nan, np.nan]],
    ])
    validity = np.ones(values.shape, dtype=bool)
    validity[:, :, 1] = [[False, True, True], [True, False, True]]
    validity[:, :, 2] = False
    batch = FactorBatch(
        ("ordinary", "masked_nonfinite", "no_valid", "overflow_mean"),
        AxisRef("time", "int64", 2, np.arange(2, dtype=np.int64)),
        AxisRef("asset", "int64", 3, np.arange(3, dtype=np.int64)),
        values, validity=validity,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        actual = diagnose_all_factors(batch)
        for index, factor_id in enumerate(batch.factor_ids):
            assert _diagnosis_tuple(actual[factor_id]) == _legacy_diagnosis(batch, index)
            assert _diagnosis_tuple(diagnose_factor(batch, index)) == _legacy_diagnosis(batch, index)



@pytest.mark.parametrize("shape", [(0, 3, 1), (4, 0, 2)])
def test_empty_axis_diagnosis_matches_legacy_flatten(shape):
    time_count, asset_count, factor_count = shape
    values = np.empty(shape, dtype=np.float64)
    batch = FactorBatch(
        tuple(f"f{i}" for i in range(factor_count)),
        AxisRef("time", "int64", time_count, np.arange(time_count, dtype=np.int64)),
        AxisRef("asset", "int64", asset_count, np.arange(asset_count, dtype=np.int64)),
        values,
    )
    assert all(actual.flags.c_contiguous for actual in (batch.values,))
    with np.errstate(over="ignore", invalid="ignore"):
        diagnostics = diagnose_all_factors(batch)
        for index, factor_id in enumerate(batch.factor_ids):
            assert _diagnosis_tuple(diagnostics[factor_id]) == _legacy_diagnosis(batch, index)


@pytest.mark.parametrize("factor_count", [1, 4])
def test_random_mask_diagnosis_matches_legacy_for_frozen_noncontiguous_input(factor_count):
    rng = np.random.default_rng(20261001 + factor_count)
    time_count, source_assets = 9, 18
    source_values = rng.standard_normal((time_count, source_assets, factor_count))[:, ::2, :]
    source_validity = (rng.random((time_count, source_assets, factor_count)) > 0.23)[:, ::2, :]
    assert not source_values.flags.c_contiguous
    assert not source_validity.flags.c_contiguous
    source_values[1, 2, :] = np.nan
    source_values[3, 1, :] = np.inf
    batch = FactorBatch(
        tuple(f"f{i}" for i in range(factor_count)),
        AxisRef("time", "int64", time_count, np.arange(time_count, dtype=np.int64)),
        AxisRef("asset", "int64", source_values.shape[1], np.arange(source_values.shape[1], dtype=np.int64)),
        source_values, validity=source_validity,
    )
    assert batch.values.flags.c_contiguous
    assert batch.validity.flags.c_contiguous
    diagnostics = diagnose_all_factors(batch)
    for index, factor_id in enumerate(batch.factor_ids):
        expected = _legacy_diagnosis(batch, index)
        assert _diagnosis_tuple(diagnostics[factor_id]) == expected
        assert _diagnosis_tuple(diagnose_factor(batch, index)) == expected
