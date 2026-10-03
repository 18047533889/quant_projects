from __future__ import annotations

from fractions import Fraction
import math
import weakref

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.factor_tile_source import FactorTile
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.scripts.source_linear_shape_oracle import (
    METRIC_IDS,
    _six_profiles,
    reference_source_linear_shape,
)


class Source:
    def __init__(self, values, *, validity=None, tile_size=4):
        self.values = np.asarray(values, dtype=np.float64)
        t, n, f = self.values.shape
        self.factor_ids = tuple(f"f{i}" for i in range(f))
        self.time_values = np.datetime64("2020-01-01") + np.arange(t).astype("timedelta64[D]")
        self.asset_values = np.asarray([f"A{i}" for i in range(n)])
        self.time_axis = AxisRef("time", "datetime64[ns]", t,
                                 self.time_values.astype("datetime64[ns]"))
        self.asset_axis = AxisRef("asset", "str", n, self.asset_values)
        self.dtype = "float64"
        self.snapshot_id = "test-snapshot"
        self.max_tile_size = tile_size
        self.validity = validity
        self.reads = []
        self.closed = False

    def read_tile(self, start, end):
        self.reads.append((start, end))
        batch = FactorBatch(
            self.factor_ids[start:end], self.time_axis, self.asset_axis,
            self.values[:, :, start:end],
            None if self.validity is None else self.validity[:, :, start:end],
        )
        return FactorTile(start, end, batch, self.snapshot_id)

    def close(self):
        self.closed = True


def labels_for(source, values, validity=None):
    dates = tuple(source.time_axis.values)
    starts = tuple(value + np.timedelta64(1, "D") for value in dates)
    ends = tuple(value + np.timedelta64(2, "D") for value in dates)
    return LabelBundle(
        "forward", np.asarray(values, dtype=np.float64), 1,
        decision_time=dates, execution_time=dates,
        signal_available_time=dates, label_start_time=starts,
        label_end_time=ends, validity=validity,
        asset_axis=source.asset_axis,
    )


def test_factor_only_edges_and_label_mask_only_changes_bucket_means():
    values = np.tile(np.array([1.0, 2.0, 3.0, 4.0])[None, :, None], (20, 1, 1))
    source = Source(values)
    forward = np.tile(np.array([np.nan, 20.0, 30.0, 40.0])[None, :], (20, 1))
    labels = labels_for(source, forward, np.isfinite(forward))

    result = reference_source_linear_shape(
        source, labels, n_quantiles=2, min_assets=1, min_periods=20,
        max_tile_size=1,
    )

    # Factor-only linear edge is 2.5: assets 1,2 are low; 3,4 high.
    # Missing return for asset 1 removes it only from the low-bucket mean.
    assert result.scalar_metrics["quantile_adjacent_spread"][0] == 15.0
    assert result.scalar_metrics["top_quantile_cliff"][0] == 15.0
    assert result.scalar_metrics["bottom_quantile_cliff"][0] == 15.0
    assert np.isnan(result.scalar_metrics["quantile_curvature"][0])
    assert np.isnan(result.scalar_metrics["quantile_tail_asymmetry"][0])
    assert all(result.observation_counts[name][0] == int(np.isfinite(result.scalar_metrics[name][0]))
               for name in METRIC_IDS)


def test_max_tie_policy_sends_values_equal_to_edge_to_higher_bucket():
    values = np.tile(np.array([1.0, 2.0, 2.0, 3.0])[None, :, None], (20, 1, 1))
    source = Source(values)
    forward = np.tile(np.array([10.0, 20.0, 30.0, 40.0])[None, :], (20, 1))
    result = reference_source_linear_shape(
        source, labels_for(source, forward), n_quantiles=2, min_assets=1,
        min_periods=20,
    )
    # Edge is exactly 2; side=right puts both tied 2s in the upper bucket.
    assert result.scalar_metrics["quantile_adjacent_spread"][0] == 20.0


def test_temporal_min_periods_boundary_and_n_less_than_q():
    for t, expected_count in ((19, 0), (20, 1)):
        values = np.tile(np.array([1.0, 2.0, 3.0, 4.0])[None, :, None], (t, 1, 1))
        source = Source(values)
        forward = np.tile(np.array([10.0, 20.0, 30.0, 40.0])[None, :], (t, 1))
        result = reference_source_linear_shape(
            source, labels_for(source, forward), n_quantiles=2,
            min_assets=1, min_periods=20,
        )
        assert result.observation_counts["top_quantile_cliff"][0] == expected_count

    too_small = Source(np.ones((20, 1, 1)))
    y = np.ones((20, 1))
    result = reference_source_linear_shape(
        too_small, labels_for(too_small, y), n_quantiles=2,
        min_assets=1, min_periods=20,
    )
    assert all(np.isnan(result.scalar_metrics[name][0]) for name in METRIC_IDS)


def test_q3_formula_values_and_partial_finite_adjacent_pairs():
    assert _six_profiles(np.array([1.0, 3.0, 6.0])) == (
        1.0, 1.0, 2.5, 2.5, 3.0, 2.0,
    )
    partial = _six_profiles(np.array([1.0, 3.0, np.nan, 9.0]))
    assert partial[2] == 2.0  # only the first adjacent pair is finite
    assert partial[5] == 2.0
    assert np.isnan(partial[0])
    assert np.isnan(partial[1])
    assert np.isnan(partial[3])
    assert np.isnan(partial[4])


def test_fraction_math_handles_subnormal_and_extreme_linear_values():
    least = np.nextafter(np.float64(0.0), np.float64(1.0))
    values = np.array([0.0, least])
    result = _six_profiles(values)
    assert result[2] == float(Fraction.from_float(least))
    assert result[4] == least
    assert result[5] == least
    # Exact dyadic arithmetic computes the difference before its final rounding.
    huge = np.finfo(np.float64).max
    assert _six_profiles(np.array([-huge, huge]))[4] == math.inf


def test_curvature_stencils_are_averaged_before_final_rounding():
    huge = np.finfo(np.float64).max
    # The two exact stencil terms are +H and -2H. Rounding each first
    # overflows, while their exact average is the finite value -H/2.
    profile = np.array([0.0, 0.0, huge, 0.0])
    assert _six_profiles(profile)[0] == -huge / 2


def test_overflow_formula_is_retained_but_not_counted_as_finite_observation():
    huge = np.finfo(np.float64).max
    values = np.tile(np.array([1.0, 2.0, 3.0, 4.0])[None, :, None], (20, 1, 1))
    source = Source(values)
    forward = np.tile(np.array([-huge, -huge, huge, huge])[None, :], (20, 1))
    result = reference_source_linear_shape(
        source, labels_for(source, forward), n_quantiles=2,
        min_assets=1, min_periods=20,
    )
    assert result.scalar_metrics["top_quantile_cliff"][0] == math.inf
    assert result.observation_counts["top_quantile_cliff"][0] == 0


def test_n_quantiles_requires_at_least_two_before_first_read():
    source = Source(np.ones((20, 4, 1)))
    with pytest.raises(ValueError, match="n_quantiles"):
        reference_source_linear_shape(
            source, labels_for(source, np.ones((20, 4))), n_quantiles=1,
        )
    assert source.reads == []


def test_previous_tile_backing_array_is_released_before_next_read():
    class EphemeralTileSource(Source):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.previous_tile_ref = None

        def read_tile(self, start, end):
            if self.previous_tile_ref is not None:
                assert self.previous_tile_ref() is None
            self.reads.append((start, end))
            cube = self.values[:, :, start:end].copy()
            self.previous_tile_ref = weakref.ref(cube)
            batch = FactorBatch(
                self.factor_ids[start:end], self.time_axis, self.asset_axis,
                cube,
                None if self.validity is None else self.validity[:, :, start:end].copy(),
            )
            return FactorTile(start, end, batch, self.snapshot_id)

    values = np.tile(np.arange(12.0)[None, :, None], (20, 1, 4))
    source = EphemeralTileSource(values, tile_size=2)
    forward = np.tile(np.arange(12.0)[None, :], (20, 1))
    reference_source_linear_shape(
        source, labels_for(source, forward), n_quantiles=2,
        min_assets=1, min_periods=20, max_tile_size=2,
    )
    assert source.reads == [(0, 2), (2, 4)]
    assert source.previous_tile_ref() is None



def test_result_budget_rejects_before_first_read_and_source_lifecycle_is_caller_owned():
    source = Source(np.ones((20, 4, 1)))
    y = np.ones((20, 4))
    with pytest.raises(MemoryError):
        reference_source_linear_shape(
            source, labels_for(source, y), max_result_bytes=1,
            n_quantiles=2, min_assets=1,
        )
    assert source.reads == []
    assert not source.closed


def test_tiled_coverage_and_inputs_are_not_mutated():
    values = np.tile(np.arange(12.0)[None, :, None], (20, 1, 4))
    original = values.copy()
    source = Source(values, tile_size=2)
    forward = np.tile(np.arange(12.0)[None, :], (20, 1))
    result = reference_source_linear_shape(
        source, labels_for(source, forward), n_quantiles=2,
        min_assets=1, min_periods=20, max_tile_size=2,
    )
    assert source.reads == [(0, 2), (2, 4)]
    assert not source.closed
    assert np.array_equal(source.values, original)
    assert result.factor_ids == source.factor_ids
    assert result.metadata["coverage_scope"] == "every_factor; bounded_factor_tiles"
    assert result.metadata["source_request_fingerprint"]


def test_typed_asset_axis_mismatch_is_rejected_before_read():
    source = Source(np.ones((20, 4, 1)))
    y = np.ones((20, 4))
    labels = labels_for(source, y)
    wrong_axis = AxisRef("asset", "int64", 4, np.arange(4))
    wrong = LabelBundle(
        labels.target_id, labels.values, labels.horizon,
        decision_time=labels.decision_time, execution_time=labels.execution_time,
        signal_available_time=labels.signal_available_time,
        label_start_time=labels.label_start_time, label_end_time=labels.label_end_time,
        validity=labels.validity, asset_axis=wrong_axis,
    )
    with pytest.raises(ValueError, match="asset axis"):
        reference_source_linear_shape(source, wrong, n_quantiles=2, min_assets=1)
    assert source.reads == []
