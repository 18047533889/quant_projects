"""Tests for multi-horizon forward-return labels and multi-horizon summaries.

Covers: formula oracle, causality (truncation/append invariance), NaN and
validity propagation, integration with evaluate_horizons (bit-level parity of
the H=1 daily IC with the single-horizon rank_ic_series kernel), summary
correctness against hand-computed examples, and dtype fail-closed policy.
"""
import numpy as np
import pytest

from quant_evaluator import evaluate
from quant_evaluator.api.horizons import (
    HorizonEvaluationBundle,
    evaluate_factor_multi_horizon,
    evaluate_horizons,
    build_forward_return_label_bundles,
    summarize_horizons,
)
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_artifacts import SeriesMetricArtifact


def factor_batch(t, n, seed=77001):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(t, n))
    times = np.arange(t)
    assets = np.asarray([f"S{i:03d}" for i in range(n)])
    return FactorBatch(
        ("predictor",),
        AxisRef("time", "int64", t, times),
        AxisRef("asset", "str", n, assets),
        values[:, :, None],
        value_hash="test-factor-content",
    ), values


def build_labels(returns, horizons=(1, 3, 5), validity=None):
    t, n = returns.shape
    return build_forward_return_label_bundles(
        returns, horizons,
        time_axis_values=tuple(np.arange(t)),
        asset_axis_values=[f"S{i:03d}" for i in range(n)],
        validity=validity,
    )


# ---------------------------------------------------------------------------
# 1. Formula oracle
# ---------------------------------------------------------------------------

def test_compound_formula_matches_hand_computed_oracle():
    r = np.array([[0.01], [0.02], [-0.005], [0.03], [0.04], [-0.02]])
    labels = build_labels(r, horizons=(1, 3))[3]
    assert labels.horizon == 3
    # H=3 at t=0: (1.02 * 0.995 * 1.03) - 1 = 0.045347 (hand-computed)
    assert labels.values[0, 0] == pytest.approx(0.045347, abs=1e-12)
    # H=3 at t=1: (0.995 * 1.03 * 1.04) - 1 = 0.065844 (hand-computed)
    assert labels.values[1, 0] == pytest.approx(0.065844, abs=1e-12)
    # H=3 at t=2: (1.03 * 1.04 * 0.98) - 1 = 0.049776 (hand-computed)
    assert labels.values[2, 0] == pytest.approx(0.049776, abs=1e-12)
    # window [t+1, t+1+3] crosses the end for t >= 3 (T=6)
    assert np.isnan(labels.values[3:]).all()
    assert labels.label_end_time[0] == 4  # t+1+H
    assert labels.label_start_time[0] == 1  # t+1


def test_h1_label_is_exact_single_period_return():
    rng = np.random.default_rng(33007)
    r = rng.normal(size=(40, 6))
    labels = build_labels(r, horizons=(1,))[1]
    # H=1 identity is exact (no (1+r)-1 round-trip): bitwise equal.
    assert np.array_equal(labels.values[:-1], r[1:], equal_nan=True)
    assert np.isnan(labels.values[-1]).all()


def test_h2_compound_matches_explicit_product():
    rng = np.random.default_rng(33008)
    r = rng.normal(size=(30, 4))
    labels = build_labels(r, horizons=(2,))[2]
    shifted1 = np.full_like(r, np.nan)
    shifted1[:29] = r[1:]  # shifted1[t] = r[t+1]
    shifted2 = np.full_like(r, np.nan)
    shifted2[:28] = r[2:]  # shifted2[t] = r[t+2]; t >= 28 has no r[t+2]
    expected = (1.0 + shifted1) * (1.0 + shifted2) - 1.0
    np.testing.assert_allclose(labels.values, expected, equal_nan=True, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 2. Causality: truncation / append invariance, window overflow
# ---------------------------------------------------------------------------

def test_truncation_and_append_invariance():
    rng = np.random.default_rng(33009)
    r = rng.normal(size=(60, 8))
    full = build_labels(r, horizons=(1, 3, 5, 10))
    t0 = 40
    prefix = build_labels(r[:t0], horizons=(1, 3, 5, 10))
    for h, bundle in full.items():
        # rows whose window fits entirely in the prefix (t <= t0-1-h) must be
        # bitwise identical: label_H(t) only reads r(t+1..t+H).  Prefix tail
        # rows (window beyond the truncated panel) are NaN there — that is
        # maturity, not a causality violation.
        np.testing.assert_allclose(
            bundle.values[: t0 - h], prefix[h].values[: t0 - h],
            equal_nan=True, rtol=0, atol=0,
        )
        assert np.isnan(prefix[h].values[t0 - h:]).all()
    appended = build_labels(np.r_[r, rng.normal(size=(7, 8))], horizons=(1, 3, 5, 10))
    for h, bundle in appended.items():
        # appending data may only FILL previously-immature tail rows; every
        # row that was finite in the base run stays bitwise identical
        was_finite = np.isfinite(full[h].values)
        np.testing.assert_allclose(
            appended[h].values[:60][was_finite], full[h].values[was_finite],
            rtol=0, atol=0,
        )
        # previously-immature rows become finite exactly when their window
        # now fits inside the appended panel (t <= T_appended-1-h)
        fillable = (np.arange(60) <= 67 - 1 - h)[:, None]
        assert np.isfinite(appended[h].values[:60][~was_finite & fillable]).all()
        assert np.isnan(appended[h].values[:60][~was_finite & ~fillable]).all()


def test_window_beyond_end_is_nan_not_zero():
    r = np.ones((10, 3)) * 0.001  # a naive zero-fill would give finite labels
    for h in (1, 3, 5, 10, 20):
        labels = build_labels(r, horizons=(h,))[h]
        if h >= 9:  # no decision index has a full window: t <= T-1-h < 0
            assert np.isnan(labels.values).all()
            continue
        last_mature = 10 - 1 - h  # last t with r(t+1..t+h) all inside the panel
        assert np.isnan(labels.values[last_mature + 1:]).all()
        assert np.isfinite(labels.values[: last_mature + 1]).all()


# ---------------------------------------------------------------------------
# 3. NaN / validity propagation
# ---------------------------------------------------------------------------

def test_single_nan_day_poisons_whole_window():
    rng = np.random.default_rng(33010)
    r = rng.normal(size=(40, 5))
    labels = build_labels(r, horizons=(1, 3, 7))
    r_bad = r.copy()
    r_bad[20, 2] = np.nan
    bad = build_labels(r_bad, horizons=(1, 3, 7))
    for h, bundle in bad.items():
        # windows covering row 20: t in [20-h, 19]
        poisoned = bundle.values[max(0, 20 - h):20, 2]
        assert np.isnan(poisoned).all()
        # row 20 itself is *not* poisoned (its window starts at 21)
        col = bundle.values[: 40 - 1 - h, 2]
        assert np.isfinite(col[-1])
    # H=1: exactly row 19 poisoned within the mature range
    assert np.isnan(bad[1].values[19, 2])
    assert np.isfinite(np.delete(bad[1].values[:38, 2], 19)).all()
    # other assets untouched
    np.testing.assert_allclose(
        bad[3].values[:, [0, 1, 3, 4]], labels[3].values[:, [0, 1, 3, 4]],
        equal_nan=True, rtol=0, atol=0,
    )


def test_inf_return_propagates_as_nan():
    r = np.zeros((8, 2))
    r[4, 1] = np.inf
    labels = build_labels(r, horizons=(3,))[3]
    assert np.isnan(labels.values[1:4, 1]).all()  # windows covering row 4
    assert np.isfinite(labels.values[: 8 - 1 - 3, 0]).all()


def test_validity_mask_propagates_like_nan():
    rng = np.random.default_rng(33011)
    r = rng.normal(size=(40, 6))
    validity = np.ones(r.shape, dtype=bool)
    validity[15, 3] = False
    masked = build_labels(r, horizons=(1, 5), validity=validity)
    nan_equivalent = build_labels(
        np.where(validity, r, np.nan), horizons=(1, 5)
    )
    for h in (1, 5):
        np.testing.assert_allclose(
            masked[h].values, nan_equivalent[h].values, equal_nan=True, rtol=0, atol=0
        )
    assert np.isnan(masked[5].values[11:15, 3]).all()


def test_nan_cell_baked_into_values_not_into_a_mask():
    r = np.zeros((6, 2))
    r[2, 0] = np.nan
    labels = build_labels(r, horizons=(1,))[1]
    # the builder encodes validity as NaN in the values (LabelBundle.validity
    # stays None); evaluate_horizons derives cell validity from isfinite
    assert labels.validity is None
    assert np.isnan(labels.values[1, 0])  # window [2, 2] covers the NaN return
    assert np.isfinite(labels.values[0, 0])


# ---------------------------------------------------------------------------
# 4. Integration with evaluate_horizons (bit-level H=1 parity)
# ---------------------------------------------------------------------------

def test_multi_horizon_h1_matches_single_horizon_rank_ic_series_bitwise():
    t, n = 48, 30
    batch, factor_values = factor_batch(t, n)
    rng = np.random.default_rng(33012)
    # plant cross-sectional signal: tomorrow's return follows today's factor
    r = rng.normal(scale=0.02, size=(t, n))
    rank = (factor_values - factor_values.mean(axis=1, keepdims=True))
    r[1:] += 0.05 * (rank[:-1] / (rank[:-1].std(axis=1, keepdims=True) + 1e-12))
    labels = build_labels(r, horizons=(1, 3, 5))
    # as_of covers the last label endpoint bar (t+1+H <= T), so every row
    # with a finite label is mature; this isolates label semantics from the
    # maturity policy (whose tail NaN behavior is covered by
    # test_immature_rows_cannot_leak_into_evaluation).
    as_of = t
    bundle = evaluate_horizons(
        batch, labels, as_of=as_of, sample_policy="per_horizon",
        min_assets=10, min_periods=5,
    )
    assert isinstance(bundle, HorizonEvaluationBundle)
    single = evaluate(
        batch, labels[1],
        metrics=("rank_ic_series",),
        metric_parameters={"rank_ic_series": {"min_assets": 10}},
    )
    single_values = np.asarray(single.artifacts["rank_ic_series"].values)
    h1_values = bundle.daily_ic_artifacts[1].values
    # bit-level equality (same kernel, same inputs); measured, not assumed
    assert np.array_equal(h1_values, single_values, equal_nan=True)
    # mean rank_ic scalar also matches the summary mean (same finite mask)
    scalar = evaluate(batch, labels[1], metrics=("rank_ic",))
    summary = summarize_horizons(bundle, icir_min_periods=5)
    mean_ic = summary.rows[1].mean_ic[0]
    assert mean_ic == pytest.approx(float(scalar.artifacts["rank_ic"].values[0]), rel=1e-12)


def test_one_call_entry_point_returns_bundle_and_summary():
    t, n = 40, 25
    batch, factor_values = factor_batch(t, n)
    rng = np.random.default_rng(33013)
    r = rng.normal(scale=0.01, size=(t, n))
    r[1:] += 0.03 * factor_values[:-1]
    bundle, summary = evaluate_factor_multi_horizon(
        batch, r, horizons=(1, 3, 5),
        as_of=t - 1, sample_policy="per_horizon", min_assets=8, min_periods=5,
        hac=True, icir_min_periods=5,
    )
    assert bundle.horizons == (1, 3, 5)
    assert set(summary.rows) == {1, 3, 5}
    assert summary.rows[1].t_stat_hac is not None
    assert summary.rows[1].se_hac is not None
    # planted signal: H1 mean IC must be strongly positive
    assert summary.rows[1].mean_ic[0] > 0.2
    # default axes come from the factor batch
    labels_direct = build_labels(r, horizons=(1, 3, 5))
    direct = evaluate_horizons(batch, labels_direct, as_of=t - 1,
                               sample_policy="per_horizon", min_assets=8,
                               min_periods=5)
    np.testing.assert_allclose(
        bundle.daily_ic_artifacts[3].values,
        direct.daily_ic_artifacts[3].values, equal_nan=True, rtol=0, atol=0,
    )


def test_one_call_entry_point_fail_closed_on_bad_kwargs():
    batch, _ = factor_batch(24, 20)
    r = np.full((24, 20), 0.001)
    with pytest.raises(ValueError, match="as_of"):
        evaluate_factor_multi_horizon(batch, r, horizons=(1, 3))
    with pytest.raises(ValueError, match="at least two horizons"):
        evaluate_factor_multi_horizon(batch, r, horizons=(1,), as_of=23)
    with pytest.raises(TypeError, match="unexpected keyword"):
        evaluate_factor_multi_horizon(batch, r, horizons=(1, 3), as_of=23, bogus=1)


# ---------------------------------------------------------------------------
# 5. Summary correctness against hand-computed examples
# ---------------------------------------------------------------------------

def handcrafted_bundle(ic_values):
    """Craft a HorizonEvaluationBundle with an exact daily IC series."""
    values = np.asarray(ic_values, dtype=np.float64)
    t, f = values.shape
    artifact = SeriesMetricArtifact(
        metric_id="rank_ic_series", domain="ic", values=values,
        time_index=tuple(np.arange(t)),
    )
    n = 2
    return HorizonEvaluationBundle(
        factor_ids=("f0",)[:f] if f == 1 else tuple(f"f{i}" for i in range(f)),
        horizons=(1,),
        sample_policy="common", as_of=t - 1,
        daily_ic_artifacts={1: artifact},
        curve_evidence={"f0": None},
        maturity_masks={1: np.ones(t, dtype=bool)},
        evaluation_masks={1: np.ones((t, n), dtype=bool)},
        common_mask=np.ones((t, n), dtype=bool),
        maturity_counts={1: t}, sample_counts={1: t},
        label_content_refs={1: "ref"},
    )


def test_summary_statistics_hand_computed():
    # factor 0: [0.1, 0.2, NaN, 0.3]; factor 1: [-0.1, 0.0, 0.05, 0.25]
    ic = [[0.1, -0.1], [0.2, 0.0], [np.nan, 0.05], [0.3, 0.25]]
    summary = summarize_horizons(handcrafted_bundle(ic), icir_min_periods=2)
    row = summary.rows[1]

    # mean of finite values
    np.testing.assert_allclose(row.mean_ic, [0.2, 0.05], rtol=0, atol=1e-15)
    # sample std ddof=1 over finite values
    np.testing.assert_allclose(row.ic_std[0], np.std([0.1, 0.2, 0.3], ddof=1), rtol=1e-12)
    np.testing.assert_allclose(row.ic_std[1], np.std([-0.1, 0.0, 0.05, 0.25], ddof=1), rtol=1e-12)
    # rank ICIR = mean / std(ddof=1) (compute_icir convention)
    np.testing.assert_allclose(
        row.rank_icir,
        [row.mean_ic[0] / row.ic_std[0], row.mean_ic[1] / row.ic_std[1]],
        rtol=1e-12,
    )
    # annualized IR = ICIR * sqrt(252), documented 252-trading-day assumption
    np.testing.assert_allclose(row.annualized_ir, row.rank_icir * np.sqrt(252.0), rtol=1e-15)
    # positive ratio over finite values (0.0 is not positive)
    np.testing.assert_allclose(row.positive_ic_ratio, [1.0, 0.5], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(row.sample_counts, [3, 4])
    # cumulative IC: nancumsum (NaN days skipped, never zero-filled retroactively)
    np.testing.assert_allclose(
        summary.cumulative_ic[1],
        np.nancumsum(np.asarray(ic, dtype=np.float64), axis=0),
        rtol=0, atol=0,
    )
    assert summary.cumulative_ic[1].shape == (4, 2)
    # HAC reuses the library kernel; it requires >= max_lag+10 contiguous
    # finite observations, so a 4-row series is correctly NaN (fail-closed).
    summary_hac = summarize_horizons(handcrafted_bundle(ic), icir_min_periods=2, hac=True, hac_max_lag=1)
    assert np.isnan(summary_hac.rows[1].t_stat_hac).all()
    rng = np.random.default_rng(33015)
    long_ic = np.stack([0.05 * np.sin(np.arange(30) * 0.3 + 0.7) + rng.normal(scale=0.01, size=30),
                        0.05 * np.cos(np.arange(30) * 0.2) + rng.normal(scale=0.01, size=30)], axis=1)
    long_hac = summarize_horizons(handcrafted_bundle(long_ic), icir_min_periods=2, hac=True, hac_max_lag=1)
    assert np.isfinite(long_hac.rows[1].t_stat_hac).all()
    assert np.isfinite(long_hac.rows[1].se_hac).all()
    # content identity: equal content -> equal hash; frozen and hashable
    again = summarize_horizons(handcrafted_bundle(ic), icir_min_periods=2)
    assert again == summary
    assert hash(again) == hash(summary)
    with pytest.raises(Exception):
        summary.rows[1].mean_ic[0] = 1.0
    with pytest.raises(Exception):
        summary.cumulative_ic[1][0, 0] = 1.0


def test_summary_positive_ratio_and_counts_edge_cases():
    ic = [[np.nan, 0.0], [np.nan, 0.0]]
    summary = summarize_horizons(handcrafted_bundle(ic), icir_min_periods=2)
    row = summary.rows[1]
    assert np.isnan(row.mean_ic[0]) and np.isnan(row.ic_std[0])
    assert np.isnan(row.rank_icir[0])
    np.testing.assert_array_equal(row.sample_counts, [0, 2])
    np.testing.assert_allclose(row.positive_ic_ratio[1], 0.0, rtol=0, atol=1e-15)
    # constant series: zero dispersion -> NaN ICIR (compute_icir convention)
    ic_const = [[0.05, 0.1], [0.05, 0.1], [0.05, 0.1]]
    summary_const = summarize_horizons(handcrafted_bundle(ic_const), icir_min_periods=2)
    assert np.isnan(summary_const.rows[1].rank_icir).all()
    assert np.isfinite(summary_const.rows[1].mean_ic).all()


def test_summary_per_horizon_portfolio_metrics_is_structural_carrier():
    ic = [[0.1], [0.2], [0.3]]
    summary = summarize_horizons(
        handcrafted_bundle(ic),
        per_horizon_portfolio_metrics={1: {"max_underwater_duration": 4.0,
                                           "mean_underwater_duration": 1.5,
                                           "time_to_recovery": 2.0}},
    )
    carried = summary.per_horizon_portfolio_metrics[1]
    assert carried["max_underwater_duration"] == 4.0
    assert carried["mean_underwater_duration"] == 1.5
    assert carried["time_to_recovery"] == 2.0
    with pytest.raises(ValueError, match="evaluated horizons"):
        summarize_horizons(handcrafted_bundle(ic),
                           per_horizon_portfolio_metrics={9: {"x": 1.0}})
    with pytest.raises(ValueError, match="real numbers"):
        summarize_horizons(handcrafted_bundle(ic),
                           per_horizon_portfolio_metrics={1: {"x": "bad"}})


# ---------------------------------------------------------------------------
# 6. dtype policy (fail-closed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make_panel,match", [
    (lambda t, n: np.ones((t, n), dtype=bool), "boolean"),
    (lambda t, n: np.ones((t, n), dtype=np.complex128), "complex"),
    (lambda t, n: np.full((t, n), 0.1, dtype=object), "object"),
    (lambda t, n: np.ones((t, n), dtype=np.int64), "integer"),
])
def test_return_panel_dtypes_are_rejected_fail_closed(make_panel, match):
    with pytest.raises(ValueError, match=match):
        build_labels(make_panel(10, 4), horizons=(1, 3))


def test_float32_panel_is_accepted_and_upcast():
    r = np.full((10, 3), 0.01, dtype=np.float32)
    labels = build_labels(r, horizons=(1,))[1]
    assert labels.values.dtype == np.float64


def test_label_axis_and_shape_validation_fail_closed():
    r = np.zeros((10, 4))
    with pytest.raises(ValueError, match="time_axis_values"):
        build_forward_return_label_bundles(
            r, (1, 3), time_axis_values=tuple(range(9)),
            asset_axis_values=[f"S{i}" for i in range(4)])
    with pytest.raises(ValueError, match="asset_axis_values"):
        build_forward_return_label_bundles(
            r, (1, 3), time_axis_values=tuple(range(10)),
            asset_axis_values=[f"S{i}" for i in range(5)])
    with pytest.raises(ValueError, match="validity"):
        build_forward_return_label_bundles(
            r, (1, 3), time_axis_values=tuple(range(10)),
            asset_axis_values=[f"S{i}" for i in range(4)],
            validity=np.ones((10, 4), dtype=float))
    with pytest.raises(ValueError, match="unique positive"):
        build_labels(r, horizons=(1, 1, 3))
    with pytest.raises(ValueError, match="unique positive"):
        build_labels(r, horizons=(0, 3))
    with pytest.raises(ValueError, match="nonnegative"):
        build_forward_return_label_bundles(
            r, (1, 3), time_axis_values=tuple(range(10)),
            asset_axis_values=[f"S{i}" for i in range(4)], execution_delay=-1)


def test_immature_rows_cannot_leak_into_evaluation():
    """The existing test_horizons convention: immature tail must stay NaN."""
    t, n = 36, 20
    batch, _ = factor_batch(t, n)
    rng = np.random.default_rng(33014)
    r = rng.normal(scale=0.01, size=(t, n))
    labels = build_labels(r, horizons=(1, 5, 10))
    bundle = evaluate_horizons(batch, labels, as_of=t - 1,
                               sample_policy="per_horizon", min_assets=8,
                               min_periods=5)
    for h in (1, 5, 10):
        # mature decisions: t+1+h <= as_of = t-1 -> last index t-2-h;
        # NaN rows (immature labels / beyond maturity) start at t-1-h
        assert np.isnan(bundle.daily_ic_artifacts[h].values[t - 1 - h:]).all()
        assert bundle.maturity_counts[h] == t - 1 - h
