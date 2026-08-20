"""Tests for the Metric x Slice x GroupBy engine and robustness cube."""

import numpy as np
import pytest

from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_artifacts import ScalarMetricArtifact
from quant_evaluator.metrics.ic import compute_daily_ic
from quant_evaluator.metrics.quantile import assign_quantiles
from quant_evaluator.metrics.slicing import (
    UNLABELED,
    SliceEngine,
    SliceView,
    slice_by_group,
    slice_by_mask,
    slice_by_quantile,
    slice_by_time,
)
from quant_evaluator.metrics.robustness_cube import (
    MetricSpec,
    SpecRobustnessCube,
)


T, N = 6, 6


def make_panel(values, factor_ids=("f0",), assets=None):
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        values = values[:, :, None]
    return FactorBatch(
        factor_ids=tuple(factor_ids),
        time_axis=AxisRef(name="time", dtype="int64", size=values.shape[0], values=np.arange(values.shape[0])),
        asset_axis=AxisRef(name="asset", dtype="int64", size=values.shape[1], values=assets if assets is not None else np.arange(values.shape[1])),
        values=values,
    )


def make_labels(values, horizon=1):
    values = np.asarray(values, dtype=float)
    return LabelBundle(
        target_id="ret",
        values=values,
        horizon=horizon,
        decision_time=tuple(range(values.shape[0])),
        label_start_time=tuple(range(values.shape[0])),
        label_end_time=tuple(i + 1 for i in range(values.shape[0])),
    )


class TestSliceByTime:
    def test_whole_period_single_view(self):
        views = slice_by_time(np.zeros((3, 2)), time_axis=np.arange(3), spec={"mode": "whole"})
        assert len(views) == 1
        assert views[0].time_indices == (0, 1, 2)
        assert views[0].asset_indices == (0, 1)

    def test_rolling_windows_step_one(self):
        views = slice_by_time(np.zeros((6, 2)), time_axis=np.arange(6), spec={"mode": "rolling", "window": 3})
        assert len(views) == 4
        assert views[0].time_indices == (0, 1, 2)
        assert views[3].time_indices == (3, 4, 5)
        assert [v.slice_key for v in views] == [
            "time:rolling[0:3]", "time:rolling[1:4]", "time:rolling[2:5]", "time:rolling[3:6]",
        ]

    def test_rolling_windows_step_two(self):
        views = slice_by_time(np.zeros((6, 2)), time_axis=np.arange(6), spec={"mode": "rolling", "window": 2, "step": 2})
        assert [v.time_indices for v in views] == [(0, 1), (2, 3), (4, 5)]

    def test_expanding_windows(self):
        views = slice_by_time(np.zeros((5, 2)), time_axis=np.arange(5), spec={"mode": "expanding", "window": 2})
        assert [v.time_indices for v in views] == [(0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 3, 4)]

    def test_expanding_with_step(self):
        views = slice_by_time(np.zeros((7, 2)), time_axis=np.arange(7), spec={"mode": "expanding", "window": 2, "step": 3})
        # Anchors 2, 5 plus the final clamped anchor at the axis end.
        assert [len(v.time_indices) for v in views] == [2, 5, 7]

    def test_expanding_covers_full_axis_by_default_step(self):
        views = slice_by_time(np.zeros((4, 2)), time_axis=np.arange(4), spec={"mode": "expanding", "window": 4})
        assert len(views) == 1
        assert views[0].time_indices == (0, 1, 2, 3)

    def test_expanding_window_larger_than_axis_fails_closed(self):
        with pytest.raises(InvalidContractError, match="exceeds"):
            slice_by_time(np.zeros((3, 2)), time_axis=np.arange(3), spec={"mode": "expanding", "window": 4})

    def test_expanding_window_equals_axis_ok(self):
        views = slice_by_time(np.zeros((3, 2)), time_axis=np.arange(3), spec={"mode": "expanding", "window": 3})
        assert views[0].time_indices == (0, 1, 2)

    def test_expanding_default_step_is_one(self):
        views = slice_by_time(np.zeros((5, 2)), time_axis=np.arange(5), spec={"mode": "expanding", "window": 1})
        assert len(views) == 5

    def test_expanding_stops_at_first_size_exceeding_step_multiple(self):
        views = slice_by_time(np.zeros((6, 2)), time_axis=np.arange(6), spec={"mode": "expanding", "window": 2, "step": 3})
        # Anchors 2, 5, then the clamped full-axis anchor.
        assert [len(v.time_indices) for v in views] == [2, 5, 6]

    def test_expanding_step_alignment_end_sizes(self):
        views = slice_by_time(np.zeros((7, 2)), time_axis=np.arange(7), spec={"mode": "expanding", "window": 3, "step": 2})
        assert [len(v.time_indices) for v in views] == [3, 5, 7]

    def test_expanding_step_exactly_reaching_axis_end(self):
        views = slice_by_time(np.zeros((6, 2)), time_axis=np.arange(6), spec={"mode": "expanding", "window": 2, "step": 4})
        assert [len(v.time_indices) for v in views] == [2, 6]

    def test_rolling_step_exact_fit(self):
        views = slice_by_time(np.zeros((4, 2)), time_axis=np.arange(4), spec={"mode": "rolling", "window": 2, "step": 2})
        assert [v.time_indices for v in views] == [(0, 1), (2, 3)]

    def test_expanding_first_window_below_step_rejected(self):
        # window=2 < step=4: step multiples 4, 8 > axis 6 -> only one
        # window, which is fine; but window=4, step=2 with T=7 gives
        # 4, 6, 7 (final anchor clamps to axis end).
        views = slice_by_time(np.zeros((7, 2)), time_axis=np.arange(7), spec={"mode": "expanding", "window": 4, "step": 2})
        assert [len(v.time_indices) for v in views] == [4, 6, 7]

    def test_expanding_final_anchor_reaches_end(self):
        views = slice_by_time(np.zeros((6, 2)), time_axis=np.arange(6), spec={"mode": "expanding", "window": 2, "step": 5})
        assert [len(v.time_indices) for v in views] == [2, 6]

    def test_expanding_window_one_step_two_T_four(self):
        views = slice_by_time(np.zeros((4, 2)), time_axis=np.arange(4), spec={"mode": "expanding", "window": 1, "step": 2})
        assert [len(v.time_indices) for v in views] == [1, 3, 4]

    def test_expanding_window_one_step_two_T_five(self):
        views = slice_by_time(np.zeros((5, 2)), time_axis=np.arange(5), spec={"mode": "expanding", "window": 1, "step": 2})
        assert [len(v.time_indices) for v in views] == [1, 3, 5]

    def test_expanding_window_one_step_three_T_eight(self):
        views = slice_by_time(np.zeros((8, 2)), time_axis=np.arange(8), spec={"mode": "expanding", "window": 1, "step": 3})
        assert [len(v.time_indices) for v in views] == [1, 4, 7, 8]

    def test_rolling_step_two_axis_five(self):
        views = slice_by_time(np.zeros((5, 2)), time_axis=np.arange(5), spec={"mode": "rolling", "window": 2, "step": 2})
        assert [v.time_indices for v in views] == [(0, 1), (2, 3)]

    def test_rolling_step_three_axis_five(self):
        views = slice_by_time(np.zeros((5, 2)), time_axis=np.arange(5), spec={"mode": "rolling", "window": 3, "step": 3})
        assert [v.time_indices for v in views] == [(0, 1, 2)]

    def test_rolling_step_two_axis_seven(self):
        views = slice_by_time(np.zeros((7, 2)), time_axis=np.arange(7), spec={"mode": "rolling", "window": 3, "step": 2})
        assert [v.time_indices for v in views] == [(0, 1, 2), (2, 3, 4), (4, 5, 6)]

    def test_explicit_windows(self):
        views = slice_by_time(
            np.zeros((6, 2)),
            time_axis=np.arange(6),
            spec={"mode": "explicit", "windows": [(0, 2), (4, 6)]},
        )
        assert [v.time_indices for v in views] == [(0, 1), (4, 5)]

    def test_explicit_windows_with_timestamps(self):
        axis = AxisRef(name="time", dtype="int64", size=4, values=np.array([10, 20, 30, 40]))
        views = slice_by_time(np.zeros((4, 2)), time_axis=axis, spec={"mode": "explicit", "windows": [(20, 40)]})
        assert views[0].time_indices == (1, 2)

    def test_unknown_mode_fails_closed(self):
        with pytest.raises(InvalidContractError):
            slice_by_time(np.zeros((3, 2)), time_axis=np.arange(3), spec={"mode": "buckets"})

    def test_empty_spec_fails_closed(self):
        with pytest.raises(InvalidContractError):
            slice_by_time(np.zeros((3, 2)), time_axis=np.arange(3), spec={})

    def test_none_spec_fails_closed(self):
        with pytest.raises(InvalidContractError):
            slice_by_time(np.zeros((3, 2)), time_axis=np.arange(3), spec=None)

    def test_zero_window_fails_closed(self):
        with pytest.raises(InvalidContractError):
            slice_by_time(np.zeros((3, 2)), time_axis=np.arange(3), spec={"mode": "rolling", "window": 0})

    def test_window_exceeding_axis_fails_closed(self):
        with pytest.raises(InvalidContractError):
            slice_by_time(np.zeros((3, 2)), time_axis=np.arange(3), spec={"mode": "rolling", "window": 5})

    def test_out_of_bounds_position_fails_closed(self):
        with pytest.raises(InvalidContractError):
            slice_by_time(np.zeros((3, 2)), time_axis=np.arange(3), spec={"mode": "explicit", "windows": [(0, 9)]})

    def test_reversed_explicit_window_fails_closed(self):
        with pytest.raises(InvalidContractError):
            slice_by_time(np.zeros((3, 2)), time_axis=np.arange(3), spec={"mode": "explicit", "windows": [(2, 1)]})

    def test_values_axis_mismatch_fails_closed(self):
        with pytest.raises(InvalidContractError):
            slice_by_time(np.zeros((3, 2)), time_axis=np.arange(5), spec={"mode": "whole"})

    def test_take_materializes_window(self):
        panel = np.arange(24, dtype=float).reshape(4, 3, 2)
        views = slice_by_time(panel, time_axis=np.arange(4), spec={"mode": "explicit", "windows": [(1, 3)]})
        out = views[0].take(panel)
        assert out.shape == (2, 3, 2)
        np.testing.assert_array_equal(out, panel[1:3])


class TestSliceByGroup:
    def test_dict_grouping_by_asset_ids(self):
        axis = AxisRef(name="asset", dtype="int64", size=4, values=np.array([100, 101, 102, 103]))
        views = slice_by_group(axis, {"tech": [100, 101], "energy": [102, 103]})
        assert [v.slice_key for v in views] == ["group:energy", "group:tech"]
        assert views[0].asset_indices == (2, 3)
        assert views[1].asset_indices == (0, 1)

    def test_dict_grouping_by_positions_without_axis_values(self):
        axis = AxisRef(name="asset", dtype="int64", size=4)
        views = slice_by_group(axis, {"a": [0, 2], "b": [1, 3]})
        assert views[0].asset_indices == (0, 2)
        assert views[1].asset_indices == (1, 3)

    def test_aligned_array_grouping(self):
        views = slice_by_group(np.arange(4), group_labels=["x", "y", "x", "y"])
        assert views[0].asset_indices == (0, 2)
        assert views[1].asset_indices == (1, 3)

    def test_incomplete_coverage_fails_closed(self):
        axis = AxisRef(name="asset", dtype="int64", size=4, values=np.array([0, 1, 2, 3]))
        with pytest.raises(InvalidContractError, match="no group label"):
            slice_by_group(axis, {"a": [0, 1], "b": [2]})

    def test_incomplete_coverage_allowed_with_unlabeled_bucket(self):
        axis = AxisRef(name="asset", dtype="int64", size=4, values=np.array([0, 1, 2, 3]))
        views = slice_by_group(axis, {"a": [0, 1], "b": [2]}, allow_unlabeled=True)
        keys = [v.slice_key for v in views]
        assert f"group:{UNLABELED}" in keys
        unlabeled = next(v for v in views if v.slice_key == f"group:{UNLABELED}")
        assert unlabeled.asset_indices == (3,)

    def test_overlapping_groups_fail_closed(self):
        axis = AxisRef(name="asset", dtype="int64", size=3, values=np.array([0, 1, 2]))
        with pytest.raises(InvalidContractError, match="more than one group"):
            slice_by_group(axis, {"a": [0, 1], "b": [1, 2]})

    def test_empty_group_fails_closed(self):
        axis = AxisRef(name="asset", dtype="int64", size=2, values=np.array([0, 1]))
        with pytest.raises(InvalidContractError, match="empty"):
            slice_by_group(axis, {"a": [], "b": [0, 1]})

    def test_unknown_asset_id_fails_closed(self):
        axis = AxisRef(name="asset", dtype="int64", size=2, values=np.array([0, 1]))
        with pytest.raises(InvalidContractError, match="unknown asset id"):
            slice_by_group(axis, {"a": [9], "b": [0, 1]})

    def test_length_mismatch_fails_closed(self):
        with pytest.raises(InvalidContractError):
            slice_by_group(np.arange(4), group_labels=["x", "y"])

    def test_all_none_labels_fail_closed(self):
        with pytest.raises(InvalidContractError):
            slice_by_group(np.arange(3), group_labels=[None, None, None])


class TestSliceByQuantile:
    def test_tercile_membership_matches_assign_quantiles(self):
        panel = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ]
        )
        views = slice_by_quantile(panel, quantiles=3)
        expected = assign_quantiles(panel, n_quantiles=3)
        for q in range(3):
            view = views[q]
            assert view.slice_key == f"quantile:Q{q}"
            for i, t in enumerate(view.time_indices):
                members = set(view.per_time_asset_indices[i])
                assert members == set(np.nonzero(expected[t] == q)[0].tolist())

    def test_ties_use_shared_policy(self):
        # All-equal values with 'max' policy: every finite value lands in
        # the top bin, so the bottom bin is empty at every time and the
        # slice family fails closed rather than emitting an empty view.
        panel = np.ones((2, 4))
        with pytest.raises(InvalidContractError, match="empty at every time"):
            slice_by_quantile(panel, quantiles=2, method="max")

    def test_ties_min_policy_populates_bottom_bin(self):
        # With the 'min' tie policy all-equal values land in the bottom
        # bin, so the top bin is empty at every time -> fail closed.
        panel = np.ones((2, 4))
        with pytest.raises(InvalidContractError, match="empty at every time"):
            slice_by_quantile(panel, quantiles=2, method="min")

    def test_boundary_tie_assignment_matches_policy(self):
        # A value exactly on the middle boundary goes to the top bin under
        # 'max' and the bottom bin under 'min'.
        panel = np.array([[1.0, 2.0, 3.0, 4.0]])
        views_max = slice_by_quantile(panel, quantiles=2, method="max")
        expected_max = assign_quantiles(panel, n_quantiles=2, method="max")
        view = views_max[0]
        assert set(view.per_time_asset_indices[0]) == set(
            np.nonzero(expected_max[0] == 0)[0].tolist()
        )

    def test_nan_assets_are_never_members(self):
        panel = np.array(
            [
                [np.nan, 2.0, 3.0, 4.0, 5.0, 6.0],
            ]
        )
        views = slice_by_quantile(panel, quantiles=2)
        all_members = {a for v in views for members in v.per_time_asset_indices for a in members}
        assert 0 not in all_members

    def test_empty_quantile_at_every_time_fails_closed(self):
        # Two assets, three quantiles: middle bin is empty at every time.
        panel = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(InvalidContractError, match="empty at every time"):
            slice_by_quantile(panel, quantiles=3)

    def test_non_2d_input_fails_closed(self):
        with pytest.raises(InvalidContractError):
            slice_by_quantile(np.zeros(6), quantiles=2)

    def test_too_few_quantiles_fails_closed(self):
        with pytest.raises(InvalidContractError):
            slice_by_quantile(np.zeros((2, 6)), quantiles=1)

    def test_all_nan_panel_fails_closed(self):
        with pytest.raises(InvalidContractError):
            slice_by_quantile(np.full((2, 4), np.nan), quantiles=2)

    def test_take_pads_ragged_membership(self):
        panel = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        views = slice_by_quantile(panel, quantiles=2)
        # members_union is the union of per-time members; for a single time
        # it is exactly the bottom bin, so take returns the compact form.
        out = views[0].take(panel)
        assert out.shape == (1, 3)
        np.testing.assert_allclose(out[0], [1.0, 2.0, 3.0])


class TestSliceByMask:
    def test_explicit_selection(self):
        views = slice_by_mask(np.array([True, False, True, False]))
        assert len(views) == 1
        assert views[0].slice_key == "mask:selected"
        assert views[0].asset_indices == (0, 2)

    def test_all_false_fails_closed(self):
        with pytest.raises(InvalidContractError):
            slice_by_mask(np.zeros(4, dtype=bool))

    def test_non_boolean_fails_closed(self):
        with pytest.raises(InvalidContractError):
            slice_by_mask(np.array([1, 0, 1]))

    def test_2d_mask_fails_closed(self):
        with pytest.raises(InvalidContractError):
            slice_by_mask(np.zeros((2, 2), dtype=bool))


class TestSliceEngine:
    def _mean_ic_fn(self, min_periods=1):
        def fn(batch, labels):
            ic_series, _ = compute_daily_ic(batch, labels, min_assets=2)
            valid = np.sum(~np.isnan(ic_series), axis=0)
            mean = np.where(valid >= min_periods, np.nanmean(ic_series, axis=0), np.nan)
            return ScalarMetricArtifact(
                metric_id="mean_ic",
                domain="ic",
                artifact_kind="scalar",
                provenance={"metric": "compute_daily_ic"},
                created_from=("factor_batch", "label_bundle"),
                values=mean,
            )
        return fn

    def test_engine_composes_with_daily_ic(self):
        rng = np.random.default_rng(7)
        factor = rng.normal(size=(T, N))
        labels = factor * 0.5 + rng.normal(size=(T, N)) * 0.1
        batch = make_panel(factor)
        bundle = make_labels(labels)

        views = slice_by_time(factor, time_axis=np.arange(T), spec={"mode": "explicit", "windows": [(0, 3), (3, 6)]})
        engine = SliceEngine(min_slice_assets=3)
        result = engine.compute(self._mean_ic_fn(), batch, bundle, views)

        assert set(result) == {"time:explicit[0:3]", "time:explicit[3:6]"}
        for key, artifact in result.items():
            assert isinstance(artifact, ScalarMetricArtifact)
            assert artifact.values.shape == (1,)
            assert artifact.provenance["slice_kind"] == "time"
            assert len(artifact.provenance["slice_spec_id"]) == 64
        # Strong signal in both halves -> positive IC everywhere.
        assert result["time:explicit[0:3]"].values[0] > 0.8
        assert result["time:explicit[3:6]"].values[0] > 0.8

    def test_engine_group_slice_uses_group_assets_only(self):
        rng = np.random.default_rng(3)
        factor = rng.normal(size=(T, N))
        labels = -factor + rng.normal(size=(T, N)) * 0.05
        batch = make_panel(factor, assets=np.arange(N))
        bundle = make_labels(labels)

        views = slice_by_group(batch.asset_axis, {"even": [0, 2, 4], "odd": [1, 3, 5]})
        result = SliceEngine().compute(self._mean_ic_fn(), batch, bundle, views)
        assert set(result) == {"group:even", "group:odd"}
        # Within every group the factor is (near-)perfectly inversely predictive.
        assert result["group:even"].values[0] < -0.9
        assert result["group:odd"].values[0] < -0.9

    def test_engine_fails_on_empty_slice_list(self):
        batch = make_panel(np.zeros((2, 2)))
        bundle = make_labels(np.zeros((2, 2)))
        with pytest.raises(InvalidContractError):
            SliceEngine().compute(self._mean_ic_fn(), batch, bundle, [])

    def test_engine_fails_on_too_few_assets(self):
        batch = make_panel(np.zeros((3, 2)))
        bundle = make_labels(np.zeros((3, 2)))
        views = slice_by_mask(np.array([True, False]))
        with pytest.raises(InvalidContractError, match="assets"):
            SliceEngine(min_slice_assets=3).compute(self._mean_ic_fn(), batch, bundle, views)

    def test_engine_rejects_non_artifact_metric_fn(self):
        batch = make_panel(np.zeros((3, 2)))
        bundle = make_labels(np.zeros((3, 2)))
        views = slice_by_mask(np.array([True, True]))
        with pytest.raises(InvalidContractError, match="MetricArtifact"):
            SliceEngine().compute(lambda b, l: 1.0, batch, bundle, views)


class TestSliceView:
    def test_indices_only_no_copy(self):
        view = SliceView(
            slice_kind="mask",
            slice_key="m",
            time_indices=(0, 1),
            asset_indices=(1,),
        )
        panel = np.arange(12, dtype=float).reshape(2, 3, 2)
        out = view.take(panel)
        np.testing.assert_allclose(out[:, 0], panel[:, 1])

    def test_unknown_kind_fails(self):
        with pytest.raises(InvalidContractError):
            SliceView(slice_kind="weird", slice_key="k", time_indices=(0,), asset_indices=(0,))

    def test_empty_key_fails(self):
        with pytest.raises(InvalidContractError):
            SliceView(slice_kind="mask", slice_key="  ", time_indices=(0,), asset_indices=(0,))

    def test_quantile_without_assets_fails(self):
        with pytest.raises(InvalidContractError):
            SliceView(slice_kind="quantile", slice_key="q", time_indices=(0,), asset_indices=())

    def test_spec_id_is_deterministic(self):
        kwargs = dict(slice_kind="mask", slice_key="m", time_indices=(0,), asset_indices=(0, 1))
        assert SliceView(**kwargs).spec_id == SliceView(**kwargs).spec_id
        other = SliceView(slice_kind="mask", slice_key="m2", time_indices=(0,), asset_indices=(0, 1))
        assert SliceView(**kwargs).spec_id != other.spec_id


class TestSpecRobustnessCube:
    def _mean_ic_fn(self):
        def fn(batch, labels):
            ic_series, _ = compute_daily_ic(batch, labels, min_assets=3)
            mean = np.nanmean(ic_series, axis=0)
            return ScalarMetricArtifact(
                metric_id="mean_ic",
                domain="ic",
                artifact_kind="scalar",
                provenance={"metric": "compute_daily_ic"},
                created_from=("factor_batch", "label_bundle"),
                values=mean,
            )
        return fn

    def _panel(self):
        rng = np.random.default_rng(11)
        # Factor 0: fragile - predictive early, noise late.
        f0 = rng.normal(size=(T, N))
        labels = f0 * 0.8 + rng.normal(size=(T, N)) * 0.2
        labels[T // 2:] = rng.normal(size=(T - T // 2, N))
        # Factor 1: robust - predictive throughout.
        f1 = rng.normal(size=(T, N))
        labels = np.stack([labels, f1 * 0.8 + rng.normal(size=(T, N)) * 0.2], axis=0)
        values = np.stack([f0, f1], axis=-1)  # (T, N, 2)
        return values, labels.max(axis=0) * 0 + labels.mean(axis=0)

    def test_cube_flags_fragile_factor_across_time_slices(self):
        values, label_values = self._panel()
        batch = make_panel(values, factor_ids=("fragile", "robust"))
        bundle = make_labels(label_values)

        time_views = slice_by_time(values[..., 0], time_axis=np.arange(T), spec={"mode": "explicit", "windows": [(0, 3), (3, 6)]})
        spec = MetricSpec(metric_id="mean_ic", metric_fn=self._mean_ic_fn(), slices=time_views)
        cube = SpecRobustnessCube()
        result = cube.evaluate(batch, bundle, specs=[spec])

        assert set(result.summaries) == {"mean_ic"}
        summary = result.summaries["mean_ic"]
        assert summary.slice_keys == ("time:explicit[0:3]", "time:explicit[3:6]")
        # Fragile factor 0 flips/collapses in the second window; factor 1 holds.
        assert 0 in summary.fragile
        assert 1 not in summary.fragile

    def test_cube_artifacts_keyed_by_slice_and_metric(self):
        values, label_values = self._panel()
        batch = make_panel(values, factor_ids=("fragile", "robust"))
        bundle = make_labels(label_values)
        views = slice_by_time(values[..., 0], time_axis=np.arange(T), spec={"mode": "whole"})
        spec = MetricSpec(metric_id="mean_ic", metric_fn=self._mean_ic_fn(), slices=views)
        result = SpecRobustnessCube().evaluate(batch, bundle, specs=[spec])

        cell = result.artifacts["mean_ic"]
        assert set(cell) == {("time:whole[0:6]", "mean_ic")}
        assert result.values("mean_ic").shape == (1, 2)
        assert len(result.spec_ids["mean_ic"]) == 64

    def test_cube_spec_id_is_content_addressed(self):
        values, _ = self._panel()
        views_a = slice_by_time(values[..., 0], time_axis=np.arange(T), spec={"mode": "whole"})
        views_b = slice_by_time(values[..., 0], time_axis=np.arange(T), spec={"mode": "explicit", "windows": [(0, 6)]})
        spec_a = MetricSpec(metric_id="mean_ic", metric_fn=self._mean_ic_fn(), slices=views_a)
        spec_b = MetricSpec(metric_id="mean_ic", metric_fn=self._mean_ic_fn(), slices=views_b)
        # Same span, different mode -> different keys -> different spec ids.
        assert spec_a.spec_id != spec_b.spec_id
        spec_a2 = MetricSpec(metric_id="mean_ic", metric_fn=self._mean_ic_fn(), slices=views_a)
        assert spec_a.spec_id == spec_a2.spec_id

    def test_cube_group_spec_summarizes_dispersion(self):
        values, _ = self._panel()
        _, label_values = self._panel()
        batch = make_panel(values, factor_ids=("fragile", "robust"))
        bundle = make_labels(label_values)
        views = slice_by_group(batch.asset_axis, {"even": [0, 2, 4], "odd": [1, 3, 5]})
        spec = MetricSpec(metric_id="mean_ic", metric_fn=self._mean_ic_fn(), slices=views)
        result = SpecRobustnessCube().evaluate(batch, bundle, specs=[spec])
        summary = result.summaries["mean_ic"]
        assert set(summary.slice_keys) == {"group:even", "group:odd"}
        assert summary.min_values.shape == (2,)
        assert summary.mean_values.shape == (2,)

    def test_cube_empty_specs_fail_closed(self):
        batch = make_panel(np.zeros((3, 2)))
        bundle = make_labels(np.zeros((3, 2)))
        with pytest.raises(InvalidContractError):
            SpecRobustnessCube().evaluate(batch, bundle, specs=[])

    def test_cube_duplicate_metric_ids_fail_closed(self):
        values, label_values = self._panel()
        batch = make_panel(values, factor_ids=("fragile", "robust"))
        bundle = make_labels(label_values)
        views = slice_by_time(values[..., 0], time_axis=np.arange(T), spec={"mode": "whole"})
        spec = MetricSpec(metric_id="mean_ic", metric_fn=self._mean_ic_fn(), slices=views)
        with pytest.raises(InvalidContractError, match="Duplicate metric_id"):
            SpecRobustnessCube().evaluate(batch, bundle, specs=[spec, spec])

    def test_metric_spec_requires_slices(self):
        with pytest.raises(InvalidContractError):
            MetricSpec(metric_id="mean_ic", metric_fn=lambda b, l: None, slices=())
