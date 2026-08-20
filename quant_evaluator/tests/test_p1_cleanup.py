"""
QE-P1 post-P0 cleanup tests (2026-08-20 audit).

Covers:
- QE-P1-27: the public evaluate facade's ic_series wrapper uses the
  spec-declared ic_method (spearman for rank-family metrics), not a
  hardcoded pearson.
- QE-P1-28: drawdown max_drawdown_idx is NaN-safe (points at the true
  trough, not the post-wipeout NaN).
- QE-P1-29: mean_ic vs pearson_ic observation_count semantics are pinned
  and documented.
- QE-P2: compute_coverage DeprecationWarning; compute_per_time_coverage
  consistency with the shared _valid_pair_mask.
"""

import warnings

import numpy as np
import pytest

from quant_evaluator import AxisRef, FactorBatch, LabelBundle, evaluate


def _batch_labels(values, label_values, factor_ids=("f",)):
    T, N, F = values.shape
    batch = FactorBatch(
        factor_ids=tuple(factor_ids),
        time_axis=AxisRef(name="time", dtype="int64", size=T, values=np.arange(T)),
        asset_axis=AxisRef(name="asset", dtype="int64", size=N, values=np.arange(N)),
        values=values,
    )
    labels = LabelBundle(
        target_id="ret",
        values=label_values,
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )
    return batch, labels


def _rank_family_panel():
    """Panel where spearman IC is +1 but pearson IC is NOT 1 (concave link)."""
    T, N = 40, 12
    rng = np.random.default_rng(21)
    values = rng.normal(size=(T, N, 1))
    label_values = np.sign(values[:, :, 0]) * np.abs(values[:, :, 0]) ** 2
    return _batch_labels(values, label_values, ("f",))


class TestICMethodFollowsMetric:
    """QE-P1-27: wrapper IC method follows the metric spec."""

    def test_rank_family_metric_gets_spearman_series(self, monkeypatch):
        import quant_evaluator.metrics.ic as ic_module

        batch, labels = _rank_family_panel()

        seen_methods = []
        real_compute_daily_ic = ic_module.compute_daily_ic

        def spy(factor_batch, label_bundle, method="pearson", min_assets=10):
            seen_methods.append(method)
            return real_compute_daily_ic(
                factor_batch, label_bundle, method=method, min_assets=min_assets
            )

        # The facade imports compute_daily_ic lazily inside the wrapper, so
        # patch the module attribute it resolves from.
        monkeypatch.setattr(ic_module, "compute_daily_ic", spy)

        result = evaluate(batch, labels, metrics=["ic.rank.hac_p"])
        assert seen_methods, "wrapper never called compute_daily_ic"
        assert seen_methods == ["spearman"], (
            f"rank-family metric must receive a spearman IC series, "
            f"got methods {seen_methods}"
        )
        metric = result.get_metric("ic.rank.hac_p")
        assert metric.metric_id == "ic.rank.hac_p"

    def test_pearson_family_metric_still_gets_pearson(self, monkeypatch):
        import quant_evaluator.metrics.ic as ic_module

        batch, labels = _batch_labels(
            np.random.default_rng(4).normal(size=(40, 12, 1)),
            np.random.default_rng(5).normal(size=(40, 12)),
        )

        seen_methods = []
        real_compute_daily_ic = ic_module.compute_daily_ic

        def spy(factor_batch, label_bundle, method="pearson", min_assets=10):
            seen_methods.append(method)
            return real_compute_daily_ic(
                factor_batch, label_bundle, method=method, min_assets=min_assets
            )

        monkeypatch.setattr(ic_module, "compute_daily_ic", spy)

        evaluate(batch, labels, metrics=["ic.pearson.ir"])
        assert seen_methods, "wrapper never called compute_daily_ic"
        assert seen_methods == ["pearson"], (
            f"pearson-family metric must receive a pearson IC series, "
            f"got methods {seen_methods}"
        )

    def test_registry_specs_declare_ic_method(self):
        from quant_evaluator.registry.metrics import get_metric

        for name in ("hac_pvalue", "hac_tstat", "ic_median"):
            spec = get_metric(name)
            assert spec.ic_method == "spearman", name
        for name in ("mean_ic", "ic_std", "ic_ir"):
            spec = get_metric(name)
            assert spec.ic_method == "pearson", name

    def test_metric_spec_rejects_invalid_ic_method(self):
        from quant_evaluator.registry import MetricSpec, MetricStatus, MetricTier

        with pytest.raises(ValueError, match="ic_method"):
            MetricSpec(
                name="bad_ic_method",
                display_name="Bad",
                description="bad",
                status=MetricStatus.EXPERIMENTAL,
                tier=MetricTier.RESEARCH,
                ic_method="kendall",
            )

    def test_rank_median_value_matches_spearman_kernel(self):
        """ic.rank.median through the facade equals a hand-built spearman series."""
        from quant_evaluator.metrics.ic import compute_daily_ic, compute_mean_ic

        batch, labels = _rank_family_panel()
        series, _ = compute_daily_ic(batch, labels, method="spearman")
        # median of the finite daily spearman ICs
        finite = series[np.isfinite(series)]
        expected = float(np.median(finite))

        got = evaluate(batch, labels, metrics=["ic.rank.median"]).get_metric(
            "ic.rank.median"
        )
        assert got.valid
        assert got.value == pytest.approx(expected, rel=1e-9)


class TestDrawdownTroughNanSafe:
    """QE-P1-28: max_drawdown_idx ignores post-wipeout NaNs."""

    def test_mid_series_wipeout_trough_points_at_true_trough(self):
        from quant_evaluator.metrics.risk.drawdown_analysis import (
            compute_drawdown_statistics,
        )

        # Index 5: deep drawdown (true trough); index 7: total wipeout,
        # after which the dd series is NaN.
        returns = np.array(
            [
                0.10,   # 0
                -0.10,  # 1
                -0.10,  # 2
                0.05,   # 3
                0.05,   # 4
                -0.30,  # 5  <- true max drawdown trough
                0.10,   # 6
                -1.00,  # 7  <- wipeout: NaN from here onward
                0.50,   # 8
                0.20,   # 9
            ]
        )
        stats = compute_drawdown_statistics(returns, min_periods=5)

        dd_series, _, _ = compute_drawdown_statistics.__globals__[
            "compute_drawdown_series"
        ](returns)
        expected_trough = int(
            np.nonzero(np.isfinite(dd_series) & (dd_series == np.nanmin(dd_series)))[0][0]
        )
        assert expected_trough == 5
        assert stats["max_drawdown_idx"] == expected_trough
        assert stats["max_drawdown"] == pytest.approx(-np.nanmin(dd_series))

    def test_wipeout_from_start_has_no_trough(self):
        from quant_evaluator.metrics.risk.drawdown_analysis import (
            compute_drawdown_statistics,
        )

        # First return already wipes the portfolio out (r = -1.0 at t=0):
        # the whole dd series is NaN, so there is no finite trough at all.
        returns = np.array([-1.0, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        stats = compute_drawdown_statistics(returns, min_periods=5)

        # All-NaN drawdown: no finite trough, max_dd stays NaN, idx -1.
        assert np.isnan(stats["max_drawdown"])
        assert stats["max_drawdown_idx"] == -1

    def test_normal_series_trough_unchanged(self):
        from quant_evaluator.metrics.risk.drawdown_analysis import (
            compute_drawdown_statistics,
        )

        returns = np.array([0.10, -0.20, -0.10, 0.05, 0.10, 0.15])
        stats = compute_drawdown_statistics(returns, min_periods=5)

        dd_series, _, _ = compute_drawdown_statistics.__globals__[
            "compute_drawdown_series"
        ](returns)
        assert stats["max_drawdown_idx"] == int(np.argmin(dd_series))

    def test_multi_factor_wipeout_column(self):
        from quant_evaluator.metrics.risk.drawdown_analysis import (
            compute_drawdown_statistics,
        )

        returns = np.array(
            [
                [0.10, 0.05],
                [-0.30, -0.10],  # column 0 true trough at index 1
                [-1.00, 0.02],  # column 0 wipes out here
                [0.50, -0.20],  # column 1 trough at index 3
            ]
        )
        stats = compute_drawdown_statistics(returns, min_periods=2)

        assert stats["max_drawdown_idx"][0] == 1
        assert stats["max_drawdown_idx"][1] == 3


class TestObservationCountSemantics:
    """QE-P1-29: mean_ic vs pearson_ic observation bases pinned."""

    def test_mean_ic_counts_panel_cells_pearson_ic_counts_ic_days(self):
        # T=40, N=12, one factor: 480 panel cells; every day has >= 10 valid
        # assets so both daily ICs are finite on all 40 days.
        batch, labels = _rank_family_panel()
        T = 40

        mean_ic = evaluate(batch, labels, metrics=["mean_ic"]).get_metric("mean_ic")
        pearson = evaluate(batch, labels, metrics=["pearson_ic"]).get_metric(
            "pearson_ic"
        )

        assert mean_ic.observation_count == 40 * 12
        assert pearson.observation_count == T

    def test_dotted_aliases_expose_consistent_counts(self):
        batch, labels = _rank_family_panel()

        plain = evaluate(batch, labels, metrics=["mean_ic"]).get_metric("mean_ic")
        dotted = evaluate(batch, labels, metrics=["ic.pearson.mean"]).get_metric(
            "ic.pearson.mean"
        )
        pearson = evaluate(batch, labels, metrics=["pearson_ic"]).get_metric(
            "pearson_ic"
        )
        # The three spellings are the SAME kernel (values identical), but the
        # observation_count bases are two, by documented contract:
        #   - mean_ic and its dotted spelling ic.pearson.mean report
        #     whole-batch valid panel cells (historical facade behaviour);
        #   - pearson_ic reports finite daily IC days.
        assert plain.observation_count == 480
        assert dotted.observation_count == 480
        assert pearson.observation_count == 40
        assert dotted.value == pytest.approx(pearson.value)
        assert dotted.value == pytest.approx(plain.value)

    def test_registry_documents_the_two_bases(self):
        from quant_evaluator.registry.metrics import get_metric

        mean_ic_desc = get_metric("mean_ic").description
        pearson_desc = get_metric("pearson_ic").description
        assert "observation_count" in mean_ic_desc
        assert "panel" in mean_ic_desc
        assert "finite daily IC" in pearson_desc


class TestQuickWins:
    """QE-P2 quick wins."""

    def test_compute_coverage_deprecation_warning(self):
        from quant_evaluator.metrics.quality import (
            compute_coverage,
            compute_coverage_per_factor,
        )

        batch, labels = _batch_labels(np.ones((3, 4, 1)), np.ones((3, 4)))

        with pytest.warns(DeprecationWarning, match="compute_coverage_per_factor"):
            coverage, num_valid, num_total = compute_coverage(batch, labels)

        assert coverage == 1.0
        assert num_valid == 12
        assert num_total == 12
        # The replacement does not warn.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            report = compute_coverage_per_factor(batch, labels)
        assert report["f"]["coverage"] == 1.0

    def test_per_time_coverage_matches_valid_pair_mask(self):
        from quant_evaluator.metrics.quality import (
            _valid_pair_mask,
            compute_per_time_coverage,
        )

        rng = np.random.default_rng(9)
        values = rng.normal(size=(6, 8, 2))
        values[0, :3, 0] = np.nan
        values[3, :, 1] = np.nan
        label_values = rng.normal(size=(6, 8))
        label_values[2, 5:] = np.nan
        batch, labels = _batch_labels(values, label_values, ("f", "g"))

        got = compute_per_time_coverage(batch, labels)

        mask = _valid_pair_mask(batch, labels)
        expected = np.sum(mask, axis=1) / batch.num_assets
        assert got.shape == (6, 2)
        np.testing.assert_allclose(got, expected)

    def test_per_time_coverage_with_validity_masks(self):
        from quant_evaluator.metrics.quality import compute_per_time_coverage

        values = np.ones((2, 4, 1))
        validity = np.ones((2, 4, 1), dtype=bool)
        validity[0, :2, 0] = False
        batch, labels = _batch_labels(values, np.ones((2, 4)))
        batch = type(batch)(
            factor_ids=batch.factor_ids,
            time_axis=batch.time_axis,
            asset_axis=batch.asset_axis,
            values=values,
            validity=validity,
        )

        got = compute_per_time_coverage(batch, labels)
        assert got[0, 0] == pytest.approx(2 / 4)
        assert got[1, 0] == pytest.approx(1.0)
