"""Catalog binding contract tests (2026-09-24 legacy-id binding round).

The 16 legacy 10-domain catalog ids bound in this round must, through the
PUBLIC ``evaluate`` facade:

1. evaluate successfully and route to the correct output channel
   (per-factor scalar, time-axis series, or the daily TQF quantile artifact);
2. produce byte-identical output on identical requests (determinism);
3. agree with the canonical twin metric when one exists
   (spearman_ic == rank_ic, rank_ic_time_series == rank_ic_series, ...);
4. match hand-computed known-value oracles derived from the documented
   formula in ``docs/METRIC_REFERENCE.md`` (``<a id="metric-XXX">`` anchors);
5. keep the kernel dtype rejection paths (bool/object inputs raise).

The ten ids that remain deliberately unbound (``_DECLARED_UNBOUND`` in
``test_all_metrics_ab_contract.py``) are covered by the AB-contract suite.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_artifacts import (
    ScalarMetricArtifact,
    SeriesMetricArtifact,
)
from quant_evaluator.metrics.ic import compute_daily_ic
from quant_evaluator.metrics.ic_summary import compute_rolling_ic_stats
from quant_evaluator.metrics.risk.drawdown_analysis import compute_drawdown_duration
from quant_evaluator.registry.metrics import get_metric
from quant_evaluator.runtime.evaluator import evaluate

_T, _N, _F = 60, 25, 2

# ids bound in this round, grouped by their documented output channel.
_SCALAR_IDS = (
    "spearman_ic", "turnover_rate", "factor_coverage", "joint_coverage",
    "coverage_stability", "drawdown_duration", "var_95", "var_99",
    "cvar_95", "cvar_99", "skewness", "kurtosis",
)
_SERIES_IDS = ("rank_ic_time_series", "rank_ic_cross_section", "rolling_ic")
_QUANTILE_IDS = ("quantile_returns",)
_BOUND_IDS = tuple(sorted(_SCALAR_IDS + _SERIES_IDS + _QUANTILE_IDS))


def _label(values: np.ndarray, assets: AxisRef | None = None) -> LabelBundle:
    t = values.shape[0]
    if assets is None:
        assets = AxisRef("a", "str", values.shape[1], tuple(f"a{i}" for i in range(values.shape[1])))
    return LabelBundle(
        target_id="r",
        values=np.ascontiguousarray(values, dtype=np.float64),
        horizon=1,
        decision_time=tuple(range(t)),
        label_start_time=tuple(range(t)),
        label_end_time=tuple(range(1, t + 1)),
        asset_axis=assets,
    )


@pytest.fixture(scope="module")
def qe():
    """Random main fixture: all-finite panels so the daily IC wrappers
    (min_assets=20) have full support, plus a probe portfolio artifact."""
    rng = np.random.default_rng(20260924)
    times = AxisRef("t", "int", _T, np.arange(_T))
    assets = AxisRef("a", "str", _N, tuple(f"s{i}" for i in range(_N)))
    fb = FactorBatch(("f0", "f1"), times, assets, rng.normal(size=(_T, _N, _F)))
    lb = _label(rng.normal(size=(_T, _N)), assets)
    first = evaluate(fb, lb, metrics=("long_short_returns",))
    series = first.artifacts["long_short_returns"]
    probe = ProbePortfolioArtifact(
        np.ascontiguousarray(np.asarray(series.values, dtype=np.float64)),
        time_index=tuple(series.time_axis.time_index),
        factor_ids=("f0", "f1"),
    )
    return fb, lb, probe


# ---------------------------------------------------------------------------
# Registry-level sanity: bound ids are STABLE with a compute function.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric_id", _BOUND_IDS)
def test_bound_catalog_id_has_compute_fn(metric_id):
    spec = get_metric(metric_id)
    assert spec.status.value == "stable"
    assert spec.compute_fn is not None


# ---------------------------------------------------------------------------
# Public-facade executability and output channel.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric_id", _SCALAR_IDS)
def test_scalar_binding_routes_to_factor_axis(qe, metric_id):
    fb, lb, probe = qe
    bundle = evaluate(fb, lb, metrics=(metric_id,), portfolio_returns=probe)
    artifact = bundle.artifacts[metric_id]
    assert isinstance(artifact, ScalarMetricArtifact)
    values = np.asarray(artifact.values, dtype=np.float64)
    assert values.shape == (_F,)
    assert np.isfinite(values).all()
    for fid in fb.factor_ids:
        metric = bundle.grouped_metrics[fid][metric_id]
        assert metric.valid is True


@pytest.mark.parametrize("metric_id", _SERIES_IDS)
def test_series_binding_routes_to_time_axis(qe, metric_id):
    fb, lb, probe = qe
    bundle = evaluate(fb, lb, metrics=(metric_id,), portfolio_returns=probe)
    artifact = bundle.artifacts[metric_id]
    assert isinstance(artifact, SeriesMetricArtifact)
    values = np.asarray(artifact.values, dtype=np.float64)
    assert values.shape == (_T, _F)
    assert values.shape[0] == len(tuple(lb.decision_time))


def test_quantile_binding_routes_to_daily_tqf_artifact(qe):
    fb, lb, probe = qe
    bundle = evaluate(fb, lb, metrics=("quantile_returns",), portfolio_returns=probe)
    artifact = bundle.artifacts["quantile_returns"]
    values = np.asarray(artifact.values, dtype=np.float64)
    assert values.ndim == 3
    assert values.shape[0] == _T and values.shape[2] == _F
    assert tuple(artifact.factor_axis) == tuple(fb.factor_ids)


@pytest.mark.parametrize("metric_id", _BOUND_IDS)
def test_binding_is_deterministic(qe, metric_id):
    fb, lb, probe = qe
    first = evaluate(fb, lb, metrics=(metric_id,), portfolio_returns=probe)
    second = evaluate(fb, lb, metrics=(metric_id,), portfolio_returns=probe)
    a1 = first.artifacts[metric_id].values
    a2 = second.artifacts[metric_id].values
    if isinstance(a1, np.ndarray):
        h1 = stable_content_hex(tag="bindings.determinism", fields={"v": a1})
        h2 = stable_content_hex(tag="bindings.determinism", fields={"v": a2})
        assert h1 == h2
    else:
        assert repr(a1) == repr(a2)


# ---------------------------------------------------------------------------
# Known-value oracles (hand-computed from the documented formulas).
# ---------------------------------------------------------------------------


def test_skewness_known_value():
    # doc metric-skewness: scipy.stats.skew(bias=False), min_obs=10.
    # r = [0]*9 + [3]: mean=0.3, m2=0.81, m3=1.944 -> g1=8/3,
    # G1 = g1*sqrt(n(n-1))/(n-2) = (8/3)*sqrt(90)/8 = sqrt(10) approx 3.16227766.
    spec = get_metric("skewness")
    value = spec.compute_fn(returns=np.array([0.0] * 9 + [3.0]))
    assert value == pytest.approx(3.1622776601683795)


def test_kurtosis_known_value():
    # doc metric-kurtosis: scipy.stats.kurtosis(fisher=True, bias=False).
    # For [0]*9 + [3]: g2 = m4/m2^2 - 3 = 46/9 and the unbiased correction
    # (n-1)/((n-2)(n-3)) * ((n+1)g2 + 6) = 10.0 exactly.
    spec = get_metric("kurtosis")
    value = spec.compute_fn(returns=np.array([0.0] * 9 + [3.0]))
    assert value == pytest.approx(10.0)


def test_var_95_known_value():
    # doc metric-var_95: VaR = max(0, -Q_{0.05}(r)), linear-interpolated
    # empirical quantile.  Sorted r: [-0.20, -0.04 x9, 0.01 x10];
    # Q_{0.05}: index 0.05*19 = 0.95 -> -0.20 + 0.95*0.16 = -0.048.
    spec = get_metric("var_95")
    returns = np.array([-0.20] + [-0.04] * 9 + [0.01] * 10)
    assert float(spec.compute_fn(returns=returns)) == pytest.approx(0.048)


def test_var_99_known_value():
    # doc metric-var_99: fixed confidence 0.99.  Q_{0.01}: index 0.19
    # -> -0.20 + 0.19*0.16 = -0.1696.
    spec = get_metric("var_99")
    returns = np.array([-0.20] + [-0.04] * 9 + [0.01] * 10)
    assert float(spec.compute_fn(returns=returns)) == pytest.approx(0.1696)


def test_cvar_95_known_value():
    # doc metric-cvar_95: tail mass m = (1-0.95)*20 = 1 -> ES = worst loss.
    spec = get_metric("cvar_95")
    returns = np.array([-0.20] + [-0.04] * 9 + [0.01] * 10)
    assert float(spec.compute_fn(returns=returns)) == pytest.approx(0.20)


def test_cvar_99_known_value():
    # doc metric-cvar_99: m = 0.2 < 1 observation -> degrades to worst loss.
    spec = get_metric("cvar_99")
    returns = np.array([-0.20] + [-0.04] * 9 + [0.01] * 10)
    assert float(spec.compute_fn(returns=returns)) == pytest.approx(0.20)


def test_drawdown_duration_known_value():
    # doc metric-drawdown_duration: D_max = max_j(e_j - s_j + 1) over
    # underwater periods.  NAV of [0.10,-0.05,0.00,0.02,0.05]:
    # 1.10, 1.045, 1.045, 1.0659, 1.1192 -> one underwater spell of 3 periods.
    spec = get_metric("drawdown_duration")
    returns = np.array([0.10, -0.05, 0.00, 0.02, 0.05])
    assert float(spec.compute_fn(returns=returns, min_periods=1)) == 3.0
    # No drawdown at all -> documented 0.
    assert float(spec.compute_fn(returns=np.array([0.01, 0.02]), min_periods=1)) == 0.0


def _small_batch():
    """(T=4, N=3, F=1) hand-built batch: factor NaN exactly at (t=0,i=0) and
    (t=1,i=0); labels all finite.  Factor/validity masks are None."""
    values = np.ones((4, 3, 1), dtype=np.float64)
    values[0, 0, 0] = np.nan
    values[1, 0, 0] = np.nan
    times = AxisRef("t", "int", 4, np.arange(4))
    assets = AxisRef("a", "str", 3, ("a0", "a1", "a2"))
    fb = FactorBatch(("f0",), times, assets, values)
    lb = _label(np.ones((4, 3), dtype=np.float64), assets)
    return fb, lb


def test_factor_coverage_known_value():
    # doc metric-factor_coverage: per-factor fraction of valid factor cells
    # over the raw T x N panel.  10 of 12 cells finite -> 10/12.
    fb, lb = _small_batch()
    spec = get_metric("factor_coverage")
    values = np.asarray(spec.compute_fn(factor_batch=fb), dtype=np.float64)
    assert values[0] == pytest.approx(10.0 / 12.0)


def test_joint_coverage_known_value():
    # doc metric-joint_coverage: jointly valid (factor, label) fraction = 10/12.
    fb, lb = _small_batch()
    spec = get_metric("joint_coverage")
    values = np.asarray(spec.compute_fn(factor_batch=fb, label_bundle=lb), dtype=np.float64)
    assert values[0] == pytest.approx(10.0 / 12.0)


def test_coverage_stability_known_value():
    # doc metric-coverage_stability: V_f = (1/T) sum_t (c_{t,f} - mean)^2 over
    # per-day joint coverage.  Per-day coverage: 2/3, 2/3, 1, 1 -> mean 5/6,
    # population variance = 4*(1/6)^2 / 4 = 1/36.
    fb, lb = _small_batch()
    spec = get_metric("coverage_stability")
    values = np.asarray(spec.compute_fn(factor_batch=fb, label_bundle=lb), dtype=np.float64)
    assert values[0] == pytest.approx(1.0 / 36.0)


def test_spearman_ic_known_value():
    # Perfectly monotone cross-sections -> every daily Spearman IC is exactly
    # 1.0, so the time mean is 1.0 (doc metric-spearman_ic formula IC_t).
    n = 25
    grid = np.tile(np.arange(n, dtype=np.float64), (5, 1))
    times = AxisRef("t", "int", 5, np.arange(5))
    assets = AxisRef("a", "str", n, tuple(f"s{i}" for i in range(n)))
    fb = FactorBatch(("f0",), times, assets, grid[:, :, None])
    lb = _label(grid, assets)
    bundle = evaluate(fb, lb, metrics=("spearman_ic",))
    values = np.asarray(bundle.artifacts["spearman_ic"].values, dtype=np.float64)
    assert values[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Equality with the canonical twin metric (same documented kernel).
# ---------------------------------------------------------------------------


def test_spearman_ic_equals_rank_ic(qe):
    fb, lb, probe = qe
    bundle = evaluate(fb, lb, metrics=("spearman_ic", "rank_ic"), portfolio_returns=probe)
    a = np.asarray(bundle.artifacts["spearman_ic"].values)
    b = np.asarray(bundle.artifacts["rank_ic"].values)
    assert np.array_equal(a, b, equal_nan=True)


@pytest.mark.parametrize("metric_id", ["rank_ic_time_series", "rank_ic_cross_section"])
def test_rank_ic_series_ids_equal_rank_ic_series(qe, metric_id):
    fb, lb, probe = qe
    bundle = evaluate(fb, lb, metrics=(metric_id, "rank_ic_series"), portfolio_returns=probe)
    a = np.asarray(bundle.artifacts[metric_id].values)
    b = np.asarray(bundle.artifacts["rank_ic_series"].values)
    assert a.shape == b.shape == (_T, _F)
    assert np.array_equal(a, b, equal_nan=True)


def test_turnover_rate_equals_turnover(qe):
    fb, lb, probe = qe
    bundle = evaluate(fb, lb, metrics=("turnover_rate", "turnover"), portfolio_returns=probe)
    a = np.asarray(bundle.artifacts["turnover_rate"].values)
    b = np.asarray(bundle.artifacts["turnover"].values)
    assert np.array_equal(a, b, equal_nan=True)


def test_joint_coverage_equals_coverage(qe):
    fb, lb, probe = qe
    bundle = evaluate(fb, lb, metrics=("joint_coverage", "coverage"), portfolio_returns=probe)
    a = np.asarray(bundle.artifacts["joint_coverage"].values)
    b = np.asarray(bundle.artifacts["coverage"].values)
    assert np.array_equal(a, b, equal_nan=True)


def test_quantile_returns_equals_quantile_returns_daily(qe):
    fb, lb, probe = qe
    bundle = evaluate(
        fb, lb, metrics=("quantile_returns", "quantile_returns_daily"),
        portfolio_returns=probe,
    )
    a = bundle.artifacts["quantile_returns"]
    b = bundle.artifacts["quantile_returns_daily"]
    assert np.array_equal(np.asarray(a.values), np.asarray(b.values), equal_nan=True)
    assert np.array_equal(np.asarray(a.counts), np.asarray(b.counts))


def test_rolling_ic_matches_documented_rolling_ic_mean(qe):
    fb, lb, probe = qe
    bundle = evaluate(fb, lb, metrics=("rolling_ic",), portfolio_returns=probe)
    got = np.asarray(bundle.artifacts["rolling_ic"].values, dtype=np.float64)
    ic_series, _ = compute_daily_ic(fb, lb, method="pearson", min_assets=20)
    expected = compute_rolling_ic_stats(ic_series, window=60, min_periods=20)["rolling_ic_mean"]
    assert np.array_equal(got, expected, equal_nan=True)


def test_drawdown_duration_matches_documented_field(qe):
    fb, lb, probe = qe
    bundle = evaluate(fb, lb, metrics=("drawdown_duration",), portfolio_returns=probe)
    got = np.asarray(bundle.artifacts["drawdown_duration"].values, dtype=np.float64)
    series = np.asarray(probe.values, dtype=np.float64)
    if series.ndim == 1:
        series = series[:, None]
    expected = np.array([
        compute_drawdown_duration(series[:, i])["max_drawdown_duration"]
        for i in range(series.shape[1])
    ], dtype=np.float64)
    assert np.array_equal(got, expected, equal_nan=True)


@pytest.mark.parametrize(
    "metric_id, kernel, kwargs",
    [
        ("var_95", "quant_evaluator.metrics.risk.var_cvar.compute_var", {}),
        ("var_99", "quant_evaluator.metrics.risk.var_cvar.compute_var", {"confidence_level": 0.99}),
        ("cvar_95", "quant_evaluator.metrics.risk.var_cvar.compute_cvar", {}),
        ("cvar_99", "quant_evaluator.metrics.risk.var_cvar.compute_cvar", {"confidence_level": 0.99}),
    ],
)
def test_tail_risk_bindings_match_kernel(qe, metric_id, kernel, kwargs):
    import importlib

    fb, lb, probe = qe
    module_path, _, attr = kernel.rpartition(".")
    fn = getattr(importlib.import_module(module_path), attr)
    bundle = evaluate(fb, lb, metrics=(metric_id,), portfolio_returns=probe)
    got = np.asarray(bundle.artifacts[metric_id].values, dtype=np.float64)
    series = np.asarray(probe.values, dtype=np.float64)
    if series.ndim == 1:
        series = series[:, None]
    expected = np.array([
        float(fn(series[:, i], **kwargs)) for i in range(series.shape[1])
    ], dtype=np.float64)
    assert np.array_equal(got, expected, equal_nan=True)


# ---------------------------------------------------------------------------
# dtype / shape rejection paths preserved.
# ---------------------------------------------------------------------------


def test_factor_batch_rejects_bool_and_object_values():
    times = AxisRef("t", "int", 1, (0,))
    assets = AxisRef("a", "str", 1, ("a0",))
    with pytest.raises(ValueError):
        FactorBatch(("f0",), times, assets, np.array([[[True]]]))
    with pytest.raises(ValueError):
        FactorBatch(("f0",), times, assets, np.array([[["x"]]], dtype=object))


def test_skewness_kurtosis_reject_bool_and_object():
    bad_bool = np.array([True, False] * 6)
    bad_object = np.array(["x"] * 12, dtype=object)
    for metric_id in ("skewness", "kurtosis"):
        spec = get_metric(metric_id)
        with pytest.raises(TypeError):
            spec.compute_fn(returns=bad_bool)
        with pytest.raises(TypeError):
            spec.compute_fn(returns=bad_object)


def test_var_95_rejects_object():
    spec = get_metric("var_95")
    with pytest.raises(TypeError):
        spec.compute_fn(returns=np.array(["x"] * 12, dtype=object))
