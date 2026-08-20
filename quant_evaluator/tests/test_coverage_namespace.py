"""QE-METRIC-P0-03: canonical coverage/IC namespace and registry single truth.

Covers:
- per-factor coverage with min_assets actually used (hand-computed)
- dotted canonical aliases resolving to identical results
- rank_ic having exactly ONE meaning (time-mean daily Spearman IC)
- the public facade no longer shadowing registry compute_fns
- quantile_returns_full being a per-factor vector, not a scalar
"""

import numpy as np
import pytest

from quant_evaluator import AxisRef, FactorBatch, LabelBundle


def _batch_labels(values, label_values, factor_ids, validity=None, label_validity=None):
    T, N, F = values.shape
    batch = FactorBatch(
        factor_ids=tuple(factor_ids),
        time_axis=AxisRef(name="time", dtype="int64", size=T, values=np.arange(T)),
        asset_axis=AxisRef(name="asset", dtype="int64", size=N, values=np.arange(N)),
        values=values,
        validity=validity,
    )
    labels = LabelBundle(
        target_id="ret",
        values=label_values,
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
        validity=label_validity,
    )
    return batch, labels


# --- P0-01: per-factor coverage -------------------------------------------


def test_coverage_per_factor_hand_computed():
    from quant_evaluator.metrics.quality import compute_coverage_per_factor

    # T=3, N=4, F=2. Factor "good" is valid everywhere; factor "bad" is
    # invalid on day 1 for assets 0 and 1. Labels are valid except asset 3
    # on day 2 (so day 2 contributes at most 3 valid pairs per factor).
    values = np.ones((3, 4, 2))
    validity = np.ones((3, 4, 2), dtype=bool)
    validity[1, 0:2, 1] = False  # day 1: factor "bad" loses assets 0,1
    label_values = np.ones((3, 4))
    label_validity = np.ones((3, 4), dtype=bool)
    label_validity[2, 3] = False  # day 2: label invalid for asset 3
    batch, labels = _batch_labels(
        values, label_values, ("good", "bad"),
        validity=validity, label_validity=label_validity,
    )

    report = compute_coverage_per_factor(batch, labels, min_assets=3)

    # good: day0 4, day1 4, day2 3 -> 11/12; bad: day0 4, day1 2, day2 3 -> 9/12
    assert report["good"]["num_valid"] == 11
    assert report["good"]["num_total"] == 12
    assert report["good"]["coverage"] == pytest.approx(11 / 12)
    assert report["bad"]["num_valid"] == 9
    assert report["bad"]["coverage"] == pytest.approx(9 / 12)


def test_min_assets_actually_used():
    from quant_evaluator.metrics.quality import compute_coverage_per_factor

    # N=4 assets; min_assets=4 makes day 1 (2 valid) fall below the gate
    # for factor "bad", while min_assets=2 keeps every day above it.
    values = np.ones((2, 4, 1))
    validity = np.ones((2, 4, 1), dtype=bool)
    validity[1, 0:2, 0] = False
    batch, labels = _batch_labels(values, np.ones((2, 4)), ("bad",), validity=validity)

    strict = compute_coverage_per_factor(batch, labels, min_assets=4)["bad"]
    loose = compute_coverage_per_factor(batch, labels, min_assets=2)["bad"]

    assert strict["valid_days"] == 1
    assert strict["days_below_min_assets"] == 1
    assert loose["valid_days"] == 2
    assert loose["days_below_min_assets"] == 0
    # coverage itself is unaffected by the gate — only the day counts change
    assert strict["coverage"] == loose["coverage"] == pytest.approx(6 / 8)


# --- P0-03: canonical alias namespace --------------------------------------


def test_alias_table_maps_dotted_names_to_registry_ids():
    from quant_evaluator.registry.metrics import (
        CANONICAL_METRIC_ALIASES,
        get_metric,
        resolve_alias,
    )

    assert resolve_alias("ic.pearson.mean") == "mean_ic"
    assert resolve_alias("ic.rank.mean") == "rank_ic"
    assert resolve_alias("unheard_of") == "unheard_of"
    for dotted, canonical in CANONICAL_METRIC_ALIASES.items():
        spec = get_metric(canonical)  # every alias target must be registered
        assert spec is not None
        assert resolve_alias(dotted) == canonical


def test_pearson_mean_alias_equals_mean_ic_result():
    from quant_evaluator import evaluate

    rng = np.random.default_rng(7)
    T, N = 40, 12
    values = rng.normal(size=(T, N, 1))
    label_values = values[:, :, 0] + rng.normal(scale=0.1, size=(T, N))
    batch, labels = _batch_labels(values, label_values, ("f",))

    canonical = evaluate(batch, labels, metrics=["mean_ic"]).get_metric("mean_ic")
    dotted = evaluate(batch, labels, metrics=["ic.pearson.mean"]).get_metric("ic.pearson.mean")

    assert dotted.valid and canonical.valid
    assert dotted.value == pytest.approx(canonical.value)
    assert dotted.observation_count == canonical.observation_count


def test_rank_alias_equals_rank_ic_result():
    from quant_evaluator import evaluate

    rng = np.random.default_rng(11)
    T, N = 40, 12
    values = rng.normal(size=(T, N, 1))
    label_values = rng.normal(size=(T, N))
    batch, labels = _batch_labels(values, label_values, ("f",))

    canonical = evaluate(batch, labels, metrics=["rank_ic"]).get_metric("rank_ic")
    dotted = evaluate(batch, labels, metrics=["ic.rank.mean"]).get_metric("ic.rank.mean")

    assert dotted.value == pytest.approx(canonical.value)


def test_rank_ic_is_time_mean_of_daily_spearman_series():
    from quant_evaluator.metrics.ic import compute_daily_ic, compute_mean_ic

    rng = np.random.default_rng(3)
    T, N = 50, 15
    values = rng.normal(size=(T, N, 1))
    label_values = rng.normal(size=(T, N))
    batch, labels = _batch_labels(values, label_values, ("f",))

    series, _ = compute_daily_ic(batch, labels, method="spearman")
    expected_mean, _ = compute_mean_ic(series, min_periods=1)

    from quant_evaluator import evaluate

    got = evaluate(batch, labels, metrics=["rank_ic"]).get_metric("rank_ic")
    assert got.value == pytest.approx(float(expected_mean[0]), nan_ok=True)
    # ...and the series alias exposes exactly those daily values
    series_metric = evaluate(batch, labels, metrics=["ic.rank.daily"]).get_metric("ic.rank.daily")
    assert series_metric.value == pytest.approx(float(np.nanmean(series[:, 0])))


# --- P0-02: facade no longer overrides the registry -------------------------


def test_facade_defers_to_registry_compute_fn(monkeypatch):
    import quant_evaluator.registry.metrics as registry_module
    from quant_evaluator import evaluate

    def fake_coverage(factor_batch, label_bundle, min_assets=10):
        # Deliberately different from the real kernel: proof the facade
        # executes whatever compute_fn the registry currently holds.
        return np.full(len(factor_batch.factor_ids), 0.5)

    spec = registry_module.get_metric("coverage")
    # The spec is shared global state. monkeypatch cannot restore a frozen
    # dataclass field (setattr raises), so restore manually — otherwise the
    # fake compute_fn leaks into every later test in the same process.
    original_compute_fn = spec.compute_fn
    monkeypatch.setattr(registry_module, "get_metric", lambda name: spec)
    object.__setattr__(spec, "compute_fn", fake_coverage)
    try:
        batch, labels = _batch_labels(np.ones((3, 4, 1)), np.ones((3, 4)), ("f",))
        got = evaluate(batch, labels, metrics=["coverage"]).get_metric("coverage")
        assert got.value == pytest.approx(0.5)
    finally:
        object.__setattr__(spec, "compute_fn", original_compute_fn)


# --- P0-02/P0-04: quantile_returns_full is a vector -------------------------


def test_quantile_returns_full_is_vector_shaped():
    from quant_evaluator import evaluate

    # Need N >= n_quantiles * kernel min_assets(=10) valid cells per day for
    # the quantile kernel to emit finite per-quantile means; use a wide
    # cross-section so every quantile is populated on every day.
    rng = np.random.default_rng(5)
    T, N, F, Q = 60, 60, 2, 5
    values = rng.normal(size=(T, N, F))
    label_values = rng.normal(size=(T, N))
    batch, labels = _batch_labels(values, label_values, ("f", "g"))

    # The registry adapter (single truth) yields the full (Q, F) vector.
    from quant_evaluator.metrics.registry_adapters import (
        compute_quantile_returns_full_value,
    )

    vector = compute_quantile_returns_full_value(
        batch, labels, min_periods=20, n_quantiles=Q
    )
    assert vector.shape == (Q, F)
    # min_periods=20 finite days per quantile -> every entry survives the gate
    assert np.isfinite(vector).all()

    # The artifact wrapper carries the same (Q, F) vector, read-only.
    from quant_evaluator.metrics.registry_adapters import (
        compute_quantile_returns_full_artifact,
    )

    artifact = compute_quantile_returns_full_artifact(
        batch, labels, min_periods=20, n_quantiles=Q
    )
    assert artifact.values.shape == (Q, F)
    assert not artifact.values.flags.writeable

    # The public facade reduces each factor's vector to a scalar via
    # nanmean (documented adaptation) — per-factor MetricValues must be finite.
    bundle = evaluate(batch, labels, metrics=["quantile_returns_full"])
    assert bundle.metric_values == {}  # no scalar aggregation across factors
    for factor_id in ("f", "g"):
        got = bundle.get_metric("quantile_returns_full", factor_id=factor_id)
        assert got is not None
        assert got.valid
        assert np.isfinite(float(got.value))
