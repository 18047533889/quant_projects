"""Optimizer-level exact RAW summary reuse preserves the research result."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


def _fixture():
    # Continuous cross-sections make all six ROBUST_SCALE branches strictly
    # monotone within each date, so they produce repeated joint metric panels.
    rng = np.random.default_rng(20261004)
    n_time, n_assets = 240, 40
    signal = rng.normal(size=(n_time, n_assets))
    labels_values = .002 * signal + rng.normal(scale=.01, size=signal.shape)
    time_axis = AxisRef("time", "int", n_time, np.arange(n_time))
    asset_axis = AxisRef("asset", "str", n_assets,
                         np.array([f"asset-{i}" for i in range(n_assets)]))
    batch = FactorBatch(("signal",), time_axis, asset_axis, signal[:, :, None])
    labels = LabelBundle(
        "robust-scale-reuse", labels_values, 1,
        decision_time=tuple(range(n_time)),
        label_start_time=tuple(range(1, n_time + 1)),
        label_end_time=tuple(range(2, n_time + 2)),
        asset_axis=asset_axis,
    )
    return batch, labels


def _optimize(batch, labels):
    from factor_optimizer.research_batch import (
        BatchOptimizationConfig, optimize_factor_batch,
    )

    config = BatchOptimizationConfig(
        families=("ROBUST_SCALE",), maximum_candidates=16,
        bootstrap_draws=99, seed=20261004,
    )
    return optimize_factor_batch(batch, labels, config=config, allow_research=True)


def _assert_same_research_result(actual, expected):
    left, right = actual.factors["signal"], expected.factors["signal"]
    assert left.status == right.status
    assert left.selected_family == right.selected_family
    assert left.plan_identity == right.plan_identity
    assert left.train_gain == right.train_gain
    assert left.validation_lower_bound == right.validation_lower_bound
    assert left.validation_candidate_identity == right.validation_candidate_identity
    assert left.candidates == right.candidates
    assert dict(left.joint_diagnostics) == dict(right.joint_diagnostics)
    np.testing.assert_array_equal(
        np.isfinite(actual.optimized.values), np.isfinite(expected.optimized.values))
    np.testing.assert_array_equal(
        actual.optimized.values, expected.optimized.values)


def test_joint_optimizer_reuses_repeated_raw_summaries_without_changing_result(monkeypatch):
    """A dropped cache integration or changed summary must be caught at the optimizer seam."""
    from factor_optimizer import research_fitness, research_summary_cache

    batch, labels = _fixture()
    original_cache = research_summary_cache.RawMetricSummaryCache
    original_summarize = research_fitness.summarize
    uncached_count = {"construct": 0, "summarize": 0}
    cached_count = {"construct": 0, "summarize": 0}
    underlying_count = {"uncached": 0, "cached": 0}
    phase = ["uncached"]

    def counted_summarize(values, *, periods_per_year=252):
        underlying_count[phase[0]] += 1
        return original_summarize(values, periods_per_year=periods_per_year)

    class UncachedReference(original_cache):
        def __init__(self):
            uncached_count["construct"] += 1

        def summarize(self, values, *, periods_per_year=252):
            uncached_count["summarize"] += 1
            return research_fitness.summarize(
                values, periods_per_year=periods_per_year)

    class CachedSpy(original_cache):
        def __init__(self):
            cached_count["construct"] += 1
            super().__init__()

        def summarize(self, values, *, periods_per_year=252):
            cached_count["summarize"] += 1
            return super().summarize(values, periods_per_year=periods_per_year)

    monkeypatch.setattr(research_fitness, "summarize", counted_summarize)
    monkeypatch.setattr(research_summary_cache, "RawMetricSummaryCache", UncachedReference)
    reference = _optimize(batch, labels)

    phase[0] = "cached"
    monkeypatch.setattr(research_summary_cache, "RawMetricSummaryCache", CachedSpy)
    cached = _optimize(batch, labels)

    _assert_same_research_result(cached, reference)
    assert len(cached.factors["signal"].candidates) == 6
    assert cached_count["construct"] > 0
    assert cached_count["summarize"] > 0
    assert underlying_count["cached"] < underlying_count["uncached"]


def _run_cached_and_uncached(monkeypatch, batch, labels):
    """Run the optimizer with a direct-summary reference and the real cache."""
    from factor_optimizer import research_fitness, research_summary_cache

    original_cache = research_summary_cache.RawMetricSummaryCache
    original_summarize = research_fitness.summarize
    uncached_count = {"construct": 0, "summarize": 0}
    cached_count = {"construct": 0, "summarize": 0}
    underlying_count = {"uncached": 0, "cached": 0}
    phase = ["uncached"]

    def counted_summarize(values, *, periods_per_year=252):
        underlying_count[phase[0]] += 1
        return original_summarize(values, periods_per_year=periods_per_year)

    class UncachedReference(original_cache):
        def __init__(self):
            uncached_count["construct"] += 1

        def summarize(self, values, *, periods_per_year=252):
            uncached_count["summarize"] += 1
            return research_fitness.summarize(
                values, periods_per_year=periods_per_year)

    class CachedSpy(original_cache):
        def __init__(self):
            cached_count["construct"] += 1
            super().__init__()

        def summarize(self, values, *, periods_per_year=252):
            cached_count["summarize"] += 1
            return super().summarize(values, periods_per_year=periods_per_year)

    monkeypatch.setattr(research_fitness, "summarize", counted_summarize)
    monkeypatch.setattr(research_summary_cache, "RawMetricSummaryCache", UncachedReference)
    reference = _optimize(batch, labels)
    phase[0] = "cached"
    monkeypatch.setattr(research_summary_cache, "RawMetricSummaryCache", CachedSpy)
    cached = _optimize(batch, labels)
    return reference, cached, uncached_count, cached_count, underlying_count


def _assert_same_batch_result(actual, expected):
    assert tuple(actual.factors) == tuple(expected.factors)
    for factor_id in actual.factors:
        left, right = actual.factors[factor_id], expected.factors[factor_id]
        assert left.status == right.status
        assert left.selected_family == right.selected_family
        assert left.plan_identity == right.plan_identity
        assert left.train_gain == right.train_gain
        assert left.validation_lower_bound == right.validation_lower_bound
        assert left.validation_candidate_identity == right.validation_candidate_identity
        assert left.candidates == right.candidates
        assert dict(left.joint_diagnostics) == dict(right.joint_diagnostics)
    np.testing.assert_array_equal(
        np.isfinite(actual.optimized.values), np.isfinite(expected.optimized.values))
    np.testing.assert_array_equal(actual.optimized.values, expected.optimized.values)


def test_optimizer_cache_distinguishes_factors_with_different_missingness(monkeypatch):
    """A same-shape stale summary for different missing rows corrupts joint evidence."""
    batch, labels = _fixture()
    values = np.repeat(batch.values, 2, axis=2)
    values[37, :2, 0] = np.nan
    values[41, :3, 1] = np.nan
    batch = FactorBatch(("mask_a", "mask_b"), batch.time_axis,
                        batch.asset_axis, values)

    reference, cached, uncached_count, cached_count, underlying_count = (
        _run_cached_and_uncached(monkeypatch, batch, labels))

    _assert_same_batch_result(cached, reference)
    assert cached_count["construct"] > 0
    assert cached_count["summarize"] > 0
    assert underlying_count["cached"] < underlying_count["uncached"]
    left = cached.factors["mask_a"].candidates[0]["train_raw_metrics"]
    right = cached.factors["mask_b"].candidates[0]["train_raw_metrics"]
    assert left != right


def test_optimizer_does_not_cache_natural_typed_summary_failures(monkeypatch):
    """One naturally missing TRAIN label day makes each joint RAW summary unavailable."""
    batch, labels = _fixture()
    label_values = labels.values.copy()
    label_values[70, :] = np.nan
    labels = replace(labels, values=label_values)

    reference, cached, uncached_count, cached_count, underlying_count = (
        _run_cached_and_uncached(monkeypatch, batch, labels))

    _assert_same_batch_result(cached, reference)
    assert cached_count["construct"] > 0
    assert cached_count["summarize"] == 6
    assert uncached_count["summarize"] == 6
    assert underlying_count["cached"] == underlying_count["uncached"]
    assert underlying_count["cached"] >= cached_count["summarize"]
    factor = cached.factors["signal"]
    assert factor.status == "raw_retained"
    assert len(factor.candidates) == 6
    assert all(record["status"] == "ineligible" for record in factor.candidates)
    assert all(record["reason"].startswith("JointMetricsUnavailable:")
               for record in factor.candidates)

