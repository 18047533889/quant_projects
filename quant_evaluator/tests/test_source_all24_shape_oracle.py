from __future__ import annotations

import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.scripts.source_all24_shape_oracle import METRIC_IDS, reference_shape_tile


def _inputs(*, times=20, masked=False, constant=False, sparse_assets=None):
    assets = np.arange(100)
    time_values = np.datetime64("2020-01-01") + np.arange(times).astype("timedelta64[D]")
    time_axis = AxisRef("time", "datetime64[D]", times, time_values)
    asset_axis = AxisRef("asset", "int64", len(assets), assets)
    values = np.tile(assets[None, :, None].astype(float), (times, 1, 1))
    validity = np.ones(values.shape, dtype=bool)
    if constant:
        values[:, :, 0] = 7.0
    if sparse_assets is not None:
        values[:, sparse_assets:, 0] = np.nan
        validity[:, sparse_assets:, 0] = False
    batch = FactorBatch(("f0",), time_axis, asset_axis, values, validity)
    labels = np.tile((assets // 20 + 1).astype(float), (times, 1))
    label_validity = np.ones(labels.shape, dtype=bool)
    if masked:
        labels[:, 2] = np.nan
        label_validity[:, 3] = False
    starts = tuple(value + np.timedelta64(1, "D") for value in time_values)
    ends = tuple(value + np.timedelta64(2, "D") for value in time_values)
    label = LabelBundle(
        "forward", labels, 1, decision_time=tuple(time_values),
        execution_time=tuple(time_values), signal_available_time=tuple(time_values),
        label_start_time=starts, label_end_time=ends,
        validity=label_validity, asset_axis=asset_axis,
    )
    return batch, label


def test_all12_metrics_hand_checked_on_five_flat_quantile_profiles():
    batch, label = _inputs()
    result = reference_shape_tile(batch, label)

    assert tuple(result.scalar_metrics) == METRIC_IDS
    np.testing.assert_array_equal(result.scalar_metrics["coverage"], [1.0])
    np.testing.assert_array_equal(result.scalar_metrics["quantile_spread"], [4.0])
    np.testing.assert_array_equal(result.scalar_metrics["quantile_monotonicity"], [1.0])
    np.testing.assert_array_equal(result.scalar_metrics["daily_quantile_monotonicity_rate"], [1.0])
    np.testing.assert_array_equal(result.scalar_metrics["turnover"], [0.0])
    assert np.isnan(result.scalar_metrics["factor_turnover_rate"][0])
    expected_shape = {
        "quantile_curvature": 0.0,
        "quantile_tail_asymmetry": 0.0,
        "quantile_adjacent_spread": 1.0,
        "quantile_extreme_cliff": 1.0,
        "top_quantile_cliff": 1.0,
        "bottom_quantile_cliff": 1.0,
    }
    for metric, expected in expected_shape.items():
        np.testing.assert_array_equal(result.scalar_metrics[metric], [expected])

    expected_counts = {
        "coverage": 2000, "quantile_spread": 20,
        "quantile_monotonicity": 4, "daily_quantile_monotonicity_rate": 20,
        "turnover": 19, "factor_turnover_rate": 19,
    }
    for metric, expected in expected_counts.items():
        assert result.observation_counts[metric][0] == expected
    for metric in expected_shape:
        assert result.observation_counts[metric][0] == 1


def test_pairwise_label_masks_do_not_change_factor_only_bucket_edges():
    batch, label = _inputs(times=20, masked=True)
    result = reference_shape_tile(batch, label)

    assert result.observation_counts["coverage"][0] == 20 * 98
    assert result.scalar_metrics["coverage"][0] == 98 / 100
    assert result.scalar_metrics["quantile_spread"][0] == 4.0
    assert result.scalar_metrics["quantile_monotonicity"][0] == 1.0
    assert result.scalar_metrics["daily_quantile_monotonicity_rate"][0] == 1.0


def test_constant_tied_factor_and_short_history_follow_nan_and_count_rules():
    batch, label = _inputs(times=19, constant=True)
    result = reference_shape_tile(batch, label)

    assert result.observation_counts["coverage"][0] == 19 * 100
    assert result.scalar_metrics["coverage"][0] == 1.0
    for metric in METRIC_IDS[1:4] + METRIC_IDS[6:]:
        assert np.isnan(result.scalar_metrics[metric][0])
        assert result.observation_counts[metric][0] == 0
    assert np.isnan(result.scalar_metrics["factor_turnover_rate"][0])
    assert result.observation_counts["factor_turnover_rate"][0] == 18
    assert result.scalar_metrics["turnover"][0] == 0.0
    assert result.observation_counts["turnover"][0] == 18


def test_rank_turnover_observations_require_ten_finite_assets_per_date():
    batch, label = _inputs(times=20, sparse_assets=9)
    result = reference_shape_tile(batch, label)

    assert result.observation_counts["coverage"][0] == 20 * 9
    assert np.isnan(result.scalar_metrics["turnover"][0])
    assert result.observation_counts["turnover"][0] == 0


def test_empty_asset_axis_has_zero_coverage_and_no_other_observations():
    times = np.datetime64("2020-01-01") + np.arange(20).astype("timedelta64[D]")
    time_axis = AxisRef("time", "datetime64[D]", len(times), times)
    asset_axis = AxisRef("asset", "int64", 0, np.empty(0, dtype=np.int64))
    batch = FactorBatch(("f0",), time_axis, asset_axis,
                        np.empty((20, 0, 1), dtype=np.float64))
    labels = LabelBundle(
        "forward", np.empty((20, 0), dtype=np.float64), 1,
        decision_time=tuple(times),
        label_start_time=tuple(value + np.timedelta64(1, "D") for value in times),
        label_end_time=tuple(value + np.timedelta64(2, "D") for value in times),
        asset_axis=asset_axis,
    )
    result = reference_shape_tile(batch, labels)

    assert result.scalar_metrics["coverage"][0] == 0.0
    assert result.observation_counts["coverage"][0] == 0
    for metric in METRIC_IDS[1:]:
        assert np.isnan(result.scalar_metrics[metric][0])
        assert result.observation_counts[metric][0] == 0
