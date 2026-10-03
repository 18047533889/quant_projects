import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.quantile_assignments import QuantileAssignmentBatch
from quant_evaluator.contracts.quantile_policy import QuantileTiePolicy
from quant_evaluator.metrics.quantile import (
    assign_quantiles_batch,
    compute_quantile_returns,
    compute_quantile_returns_from_assignments,
)


def _axes(t, n, f):
    time = AxisRef("time", "int", t, np.arange(t))
    assets = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    return time, assets, tuple(f"f{i}" for i in range(f))


def _labels(time, assets, values, validity=None):
    t = len(time.values)
    return LabelBundle(
        "preassigned-test", values, 1, validity=validity,
        decision_time=tuple(time.values.tolist()),
        label_start_time=tuple(range(1, t + 1)),
        label_end_time=tuple(range(2, t + 2)),
        asset_axis=assets,
    )


def _pre_refactor_oracle(values, factor_validity, label_values, label_validity,
                        n_quantiles, min_assets):
    """Literal pre-reuse assignment + aggregation, independent of QE helpers."""
    t_count, n_assets, n_factors = values.shape
    out = np.full((t_count, n_quantiles, n_factors), np.nan, dtype=np.float64)
    counts_out = np.zeros((t_count, n_quantiles, n_factors), dtype=np.int32)
    for f in range(n_factors):
        for t in range(t_count):
            row = values[t, :, f].astype(float, copy=True)
            row[~factor_validity[t, :, f]] = np.nan
            finite = np.isfinite(row)
            finite_values = row[finite]
            ids = np.full(n_assets, -1, dtype=np.int32)
            if finite_values.size >= n_quantiles:
                sv = np.sort(finite_values)
                bounds = []
                for b in range(n_quantiles - 1):
                    pos = (b + 1) / n_quantiles * (len(sv) - 1)
                    lo = int(pos)
                    frac = pos - lo
                    if lo >= len(sv) - 1:
                        bound = sv[-1]
                    elif frac < 1e-9:
                        bound = sv[lo]
                    elif frac > 1.0 - 1e-9:
                        bound = sv[lo + 1]
                    else:
                        left, right = float(sv[lo]), float(sv[lo + 1])
                        delta = right - left
                        bound = (left + frac * delta if np.isfinite(delta)
                                 else (1.0 - frac) * left + frac * right)
                    bounds.append(bound)
                ids[finite] = np.clip(np.searchsorted(bounds, finite_values,
                                                        side="right"),
                                      0, n_quantiles - 1)
            labels = label_values[t].astype(float, copy=True)
            if label_validity is not None:
                labels[~label_validity[t]] = np.nan
            good = (ids >= 0) & np.isfinite(labels)
            counts = np.bincount(ids[good], minlength=n_quantiles)
            sums = np.bincount(ids[good], weights=labels[good],
                               minlength=n_quantiles)
            enough = counts >= min_assets
            counts_out[t, :, f] = counts
            out[t, enough, f] = sums[enough] / counts[enough]
    return out, counts_out


def _pre_refactor_ids(values, factor_validity, n_quantiles):
    ids = np.full(values.shape, -1, dtype=np.int32)
    for t in range(values.shape[0]):
        for f in range(values.shape[2]):
            row = values[t, :, f].astype(float, copy=True)
            row[~factor_validity[t, :, f]] = np.nan
            valid = np.isfinite(row)
            sorted_values = np.sort(row[valid])
            if len(sorted_values) < n_quantiles:
                continue
            boundaries = []
            for b in range(n_quantiles - 1):
                pos = (b + 1) / n_quantiles * (len(sorted_values) - 1)
                lo, frac = int(pos), pos - int(pos)
                if lo >= len(sorted_values) - 1 or frac < 1e-9:
                    boundary = sorted_values[min(lo, len(sorted_values)-1)]
                elif frac > 1.0 - 1e-9:
                    boundary = sorted_values[lo + 1]
                else:
                    left, right = float(sorted_values[lo]), float(sorted_values[lo+1])
                    delta = right - left
                    boundary = (left + frac * delta if np.isfinite(delta)
                                else (1-frac)*left + frac*right)
                boundaries.append(boundary)
            ids[t, valid, f] = np.clip(np.searchsorted(
                boundaries, row[valid], side="right"), 0, n_quantiles - 1)
    return ids


def test_preassigned_aggregation_is_exactly_the_legacy_value_path_with_masks():
    rng = np.random.default_rng(8103)
    t, n, f, q = 23, 41, 3, 7
    time, assets, factor_ids = _axes(t, n, f)
    values = rng.normal(size=(t, n, f))
    values[:, ::9, 1] = np.nan
    values[3, :, 2] = 1.0  # ties and a constant cross-section
    values[0, :, 0] = np.nan  # all missing
    values[1, :, 0] = np.nan
    values[1, 0, 0] = 2.0  # singleton finite panel cannot form Q bins
    values[2, :, 0] = 3.0  # ties at every boundary
    factor_validity = np.ones(values.shape, dtype=bool)
    factor_validity[5, 2, 0] = False
    label_values = rng.normal(size=(t, n))
    label_values[::4, ::5] = np.nan
    label_validity = np.ones((t, n), dtype=bool)
    label_validity[2::6, 1::7] = False
    batch = FactorBatch(factor_ids, time, assets, values,
                        validity=factor_validity)
    labels = _labels(time, assets, label_values, label_validity)

    effective = np.where(factor_validity, values, np.nan)
    expected, expected_counts = _pre_refactor_oracle(
        values, factor_validity, label_values, label_validity, q, 2)
    standard, standard_counts = compute_quantile_returns(
        batch, labels, n_quantiles=q, min_assets=2)
    np.testing.assert_array_equal(standard, expected)
    np.testing.assert_array_equal(standard_counts, expected_counts)
    ids = _pre_refactor_ids(values, factor_validity, q)
    np.testing.assert_array_equal(
        assign_quantiles_batch(effective, n_quantiles=q, method="max"), ids)
    memberships = QuantileAssignmentBatch(
        ids, time, assets, factor_ids, q, QuantileTiePolicy.MAX)
    actual, actual_counts = compute_quantile_returns_from_assignments(
        memberships, labels, min_assets=2)
    np.testing.assert_array_equal(memberships.assignments, ids)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_counts, expected_counts)


def test_standard_compute_aggregates_directly_into_final_public_buffers(monkeypatch):
    import quant_evaluator.metrics.quantile as qe_quantile

    rng = np.random.default_rng(8105)
    t, n, f, q = 9, 29, 3, 5
    time, assets, factor_ids = _axes(t, n, f)
    values = rng.normal(size=(t, n, f))
    values[2, :, 1] = 0.0
    factor_validity = np.ones(values.shape, dtype=bool)
    factor_validity[4, 3, 2] = False
    label_values = rng.normal(size=(t, n))
    label_values[1::3, ::6] = np.nan
    label_validity = np.ones((t, n), dtype=bool)
    label_validity[5, 2::4] = False
    batch = FactorBatch(factor_ids, time, assets, values,
                        validity=factor_validity)
    labels = _labels(time, assets, label_values, label_validity)
    expected, expected_counts = _pre_refactor_oracle(
        values, factor_validity, label_values, label_validity, q, 2)

    captured = []
    original = qe_quantile._aggregate_quantile_assignments

    def capture_outputs(*args, **kwargs):
        buffers = kwargs.get("out")
        captured.append(buffers)
        return original(*args, **kwargs)

    monkeypatch.setattr(qe_quantile, "_aggregate_quantile_assignments", capture_outputs)
    returns, counts = compute_quantile_returns(
        batch, labels, n_quantiles=q, min_assets=2)
    np.testing.assert_array_equal(returns, expected)
    np.testing.assert_array_equal(counts, expected_counts)
    assert len(captured) == f
    for return_view, counts_view in captured:
        assert return_view.shape == (t, q, 1)
        assert counts_view.shape == (t, q, 1)
        assert np.shares_memory(return_view, returns)
        assert np.shares_memory(counts_view, counts)


def test_assignment_contract_owns_an_immutable_copy_without_changing_input_flags():
    time, assets, factors = _axes(4, 5, 1)
    caller = np.zeros((4, 5, 1), dtype=np.int32)
    original_flags = caller.flags.writeable
    contract = QuantileAssignmentBatch(caller, time, assets, factors, 5)
    assert caller.flags.writeable is original_flags
    caller[0, 0, 0] = 4
    assert contract.assignments[0, 0, 0] == 0
    assert not contract.assignments.flags.writeable
    with pytest.raises(ValueError):
        contract.assignments.flags.writeable = True


@pytest.mark.parametrize("bad, error", [
    (np.zeros((3, 4, 1), dtype=np.int32), ValueError),
    (np.zeros((4, 5), dtype=np.int32), ValueError),
    (np.zeros((4, 5, 1), dtype=np.int64), TypeError),
    (np.zeros((4, 5, 1), dtype=np.float64), TypeError),
])
def test_assignment_contract_rejects_wrong_shape_or_non_int32(bad, error):
    time, assets, factors = _axes(4, 5, 1)
    with pytest.raises(error):
        QuantileAssignmentBatch(bad, time, assets, factors, 5)


@pytest.mark.parametrize("bad_id", [-2, 5])
def test_assignment_contract_rejects_ids_outside_missing_or_quantile_range(bad_id):
    time, assets, factors = _axes(4, 5, 1)
    ids = np.zeros((4, 5, 1), dtype=np.int32)
    ids[1, 2, 0] = bad_id
    with pytest.raises(ValueError, match="assignment IDs"):
        QuantileAssignmentBatch(ids, time, assets, factors, 5)


def test_assignment_contract_rejects_boolean_quantile_count():
    time, assets, factors = _axes(4, 5, 1)
    with pytest.raises(ValueError):
        QuantileAssignmentBatch(np.zeros((4, 5, 1), dtype=np.int32),
                                time, assets, factors, True)


@pytest.mark.parametrize("shape", [(0, 5, 1), (3, 0, 1)])
def test_empty_assignment_axes_return_well_typed_empty_ids(shape):
    from quant_evaluator.metrics.quantile import assign_quantiles_batch

    ids = assign_quantiles_batch(np.empty(shape, dtype=np.float64), n_quantiles=5)
    assert ids.shape == shape
    assert ids.dtype == np.int32


def test_aggregation_rejects_misaligned_scoring_time_and_asset_axes():
    t, n = 4, 5
    time, assets, factors = _axes(t, n, 1)
    ids = QuantileAssignmentBatch(
        np.zeros((t, n, 1), dtype=np.int32), time, assets, factors, 3)
    labels = _labels(time, assets, np.ones((t, n)))
    wrong_time = AxisRef("time", "int", t, np.arange(t) + 1)
    wrong_labels = _labels(wrong_time, assets, np.ones((t, n)))
    with pytest.raises(ValueError, match="decision_time"):
        compute_quantile_returns_from_assignments(ids, wrong_labels)

    other_assets = AxisRef("asset", "str", n, np.array([f"x{i}" for i in range(n)]))
    wrong_assets = _labels(time, other_assets, np.ones((t, n)))
    with pytest.raises(ValueError, match="asset_axis"):
        compute_quantile_returns_from_assignments(ids, wrong_assets)


def _tiny_aggregation_case():
    ids = np.array([
        [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
        [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]],
    ], dtype=np.int32)
    labels = np.array([[1., 2., 3., 4.], [10., 20., 30., 40.]])
    return ids, labels


@pytest.mark.parametrize("bad_out", [
    [np.empty((2, 2, 3)), np.empty((2, 2, 3), dtype=np.int32)],
    (np.empty((2, 2, 3)),),
])
def test_aggregation_output_requires_a_two_array_tuple(bad_out):
    from quant_evaluator.metrics.quantile import _aggregate_quantile_assignments

    ids, labels = _tiny_aggregation_case()
    with pytest.raises(TypeError, match="out must be a"):
        _aggregate_quantile_assignments(ids, labels, n_quantiles=2,
                                        min_assets=2, out=bad_out)


@pytest.mark.parametrize("bad_index", [0, 1])
def test_aggregation_output_rejects_wrong_buffer_shape(bad_index):
    from quant_evaluator.metrics.quantile import _aggregate_quantile_assignments

    ids, labels = _tiny_aggregation_case()
    outputs = [np.empty((2, 2, 3), dtype=np.float64),
               np.empty((2, 2, 3), dtype=np.int32)]
    outputs[bad_index] = np.empty((2, 2, 2), dtype=outputs[bad_index].dtype)
    with pytest.raises(ValueError, match="out buffers must both have shape"):
        _aggregate_quantile_assignments(ids, labels, n_quantiles=2,
                                        min_assets=2, out=tuple(outputs))


@pytest.mark.parametrize("bad_index", [0, 1])
def test_aggregation_output_rejects_wrong_buffer_dtype(bad_index):
    from quant_evaluator.metrics.quantile import _aggregate_quantile_assignments

    ids, labels = _tiny_aggregation_case()
    outputs = [np.empty((2, 2, 3), dtype=np.float64),
               np.empty((2, 2, 3), dtype=np.int32)]
    outputs[bad_index] = np.empty((2, 2, 3), dtype=np.float32 if bad_index == 0 else np.int64)
    with pytest.raises(TypeError, match="float64 returns and int32 counts"):
        _aggregate_quantile_assignments(ids, labels, n_quantiles=2,
                                        min_assets=2, out=tuple(outputs))


@pytest.mark.parametrize("readonly_index", [0, 1])
def test_aggregation_output_rejects_readonly_buffers(readonly_index):
    from quant_evaluator.metrics.quantile import _aggregate_quantile_assignments

    ids, labels = _tiny_aggregation_case()
    outputs = [np.empty((2, 2, 3), dtype=np.float64),
               np.empty((2, 2, 3), dtype=np.int32)]
    outputs[readonly_index].flags.writeable = False
    with pytest.raises(ValueError, match="out buffers must be writable"):
        _aggregate_quantile_assignments(ids, labels, n_quantiles=2,
                                        min_assets=2, out=tuple(outputs))


def test_reused_aggregation_output_buffers_clear_stale_invalid_rows():
    from quant_evaluator.metrics.quantile import _aggregate_quantile_assignments

    ids, labels = _tiny_aggregation_case()
    returns_buffer = np.full((2, 2, 3), 123.0, dtype=np.float64)
    counts_buffer = np.full((2, 2, 3), 77, dtype=np.int32)
    returns, counts = _aggregate_quantile_assignments(
        ids, labels, n_quantiles=2, min_assets=2,
        out=(returns_buffer, counts_buffer),
    )
    assert returns is returns_buffer
    assert counts is counts_buffer
    assert np.isnan(returns[0]).all()
    np.testing.assert_array_equal(counts[0], 0)
    np.testing.assert_array_equal(counts[1], np.full((2, 3), 2, dtype=np.int32))
    np.testing.assert_array_equal(returns[1],
                                  np.array([[15., 20., 15.], [35., 30., 35.]]))
