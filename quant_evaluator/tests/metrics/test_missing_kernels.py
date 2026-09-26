"""Missing-kernel round tests (2026-09-24).

Covers the ten legacy catalog ids that were still unbound after the
catalog-binding round (see tests/metrics/test_catalog_bindings.py):

- new pure kernels (metrics/ic_summary.py, metrics/turnover.py,
  metrics/data_quality.py): known-value hand oracles derived from the
  documented formulas in docs/METRIC_REFERENCE.md, >=3-seed equivalence
  against naive in-test references, NaN / short-sample / min_periods
  boundaries, dtype rejection (bool / complex / object), and byte-level
  determinism;
- the nine ids bound through the public ``evaluate`` facade
  (ic_summary, ic_stability, quantile_stability, hhi_concentration,
  hhi_effective_n, return_coverage, turnover_adjusted_ic,
  turnover_stability, autocorrelation_ic): executability, output channel,
  kernel agreement, and facade-level determinism;
- ic_decay: the kernel works on pre-computed per-horizon mean ICs while the
  id stays deliberately unavailable through evaluate() (the facade cannot
  synthesise one LabelBundle per forward horizon), failing with its declared
  ``HorizonMeanIC`` requirement.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.stats import rankdata, ttest_1samp

warnings.filterwarnings("ignore")

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.metric_artifacts import (
    ScalarMetricArtifact,
    VectorMetricArtifact,
)
from quant_evaluator.metrics.data_quality import compute_return_coverage
from quant_evaluator.metrics.exposure import compute_concentration_hhi
from quant_evaluator.metrics.ic import compute_daily_ic
from quant_evaluator.metrics.ic_summary import (
    compute_ic_decay_from_mean_ics,
    compute_ic_stability,
    compute_ic_stability_scalar,
    compute_ic_summary_stats,
    compute_quantile_rank_stability,
)
from quant_evaluator.metrics.quantile import compute_quantile_returns_fast
from quant_evaluator.metrics.temporal import compute_ic_autocorrelation
from quant_evaluator.metrics.turnover import (
    compute_turnover_adjusted_ic_from_series,
    compute_turnover_stability_from_series,
    estimate_turnover_from_ranks,
)
from quant_evaluator.registry.metrics import get_metric
from quant_evaluator.runtime.evaluator import evaluate

_T, _N, _F = 60, 25, 2

# ids bound through the public facade in this round.
_FACADE_SCALAR_IDS = (
    "hhi_concentration", "hhi_effective_n", "ic_stability", "ic_summary",
    "quantile_stability", "return_coverage", "turnover_adjusted_ic",
    "turnover_stability",
)


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
    rng = np.random.default_rng(20260924)
    times = AxisRef("t", "int", _T, np.arange(_T))
    assets = AxisRef("a", "str", _N, tuple(f"s{i}" for i in range(_N)))
    fb = FactorBatch(("f0", "f1"), times, assets, rng.normal(size=(_T, _N, _F)))
    lb = _label(rng.normal(size=(_T, _N)), assets)
    return fb, lb


# ---------------------------------------------------------------------------
# Kernels: known-value hand oracles.
# ---------------------------------------------------------------------------


def test_return_coverage_known_value():
    panel = np.array([[1.0, np.nan], [np.nan, 2.0]])
    assert compute_return_coverage(panel) == pytest.approx(0.5)
    assert compute_return_coverage(np.ones((3, 4))) == pytest.approx(1.0)
    assert np.isnan(compute_return_coverage(np.empty((0, 5))))


def test_ic_summary_stats_known_values():
    # Column 0: finite values [0.1, 0.3, -0.2, 0.4]; column 1 mostly NaN so it
    # fails min_periods=4 and stays all-NaN.
    ic = np.array([
        [0.1, np.nan],
        [0.3, 0.5],
        [-0.2, np.nan],
        [0.4, np.nan],
    ])
    stats = compute_ic_summary_stats(ic, min_periods=4)
    finite = ic[:, 0]
    assert stats["mean"][0] == pytest.approx(np.mean(finite))
    assert stats["std"][0] == pytest.approx(np.std(finite, ddof=1))
    assert stats["icir"][0] == pytest.approx(
        np.mean(finite) / np.std(finite, ddof=1)
    )
    t_ref, p_ref = ttest_1samp(finite, 0.0)
    assert stats["t_stat"][0] == pytest.approx(t_ref)
    assert stats["p_value"][0] == pytest.approx(p_ref)
    for key in ("mean", "std", "icir", "t_stat", "p_value"):
        assert np.isnan(stats[key][1])
    # Constant series: mean/std finite, icir/t/p NaN (documented rule).
    const = compute_ic_summary_stats(np.ones((25, 1)), min_periods=20)
    assert const["mean"][0] == pytest.approx(1.0)
    assert const["std"][0] == 0.0
    assert np.isnan(const["icir"][0]) and np.isnan(const["t_stat"][0])


def test_ic_stability_scalar_known_value():
    # window_size=40 on a 40-length series: exactly one window, whose value is
    # the pairwise-complete correlation between the first and second half.
    rng = np.random.default_rng(7)
    series = rng.normal(size=(40, 1))
    series[3, 0] = np.nan
    got = compute_ic_stability_scalar(series, window_size=40, min_periods=5)
    a, b = series[:20, 0], series[20:, 0]
    valid = np.isfinite(a) & np.isfinite(b)
    expected = np.corrcoef(a[valid], b[valid])[0, 1]
    assert got[0] == pytest.approx(expected)
    # Too short for any window -> NaN.
    assert np.isnan(compute_ic_stability_scalar(series[:30], 40, 5)[0])


def test_quantile_rank_stability_known_value():
    # Three days, three buckets: corr(d0,d1)=-1, corr(d0,d2)=+1, corr(d1,d2)=-1
    # -> mean = -1/3.
    daily = np.array([
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
        [1.0, 2.0, 3.0],
    ])
    got = compute_quantile_rank_stability(daily, min_periods=2)
    assert got[0] == pytest.approx(-1.0 / 3.0)
    # Perfectly stable rankings -> +1.
    same = np.tile(np.array([1.0, 2.0, 3.0]), (4, 1))
    assert compute_quantile_rank_stability(same)[0] == pytest.approx(1.0)
    # Single valid day fails min_periods=2 -> NaN.
    assert np.isnan(compute_quantile_rank_stability(daily[:1])[0])


def test_turnover_stability_known_value():
    series = np.array([[0.1], [0.3], [np.nan]])
    # Population variance over the finite values [0.1, 0.3].
    assert compute_turnover_stability_from_series(series)[0] == pytest.approx(0.01)
    assert np.isnan(compute_turnover_stability_from_series(series, min_periods=3)[0])


def test_turnover_adjusted_ic_known_value():
    ic = np.array([[0.2], [0.4], [np.nan]])
    tau = np.array([[0.1], [0.3], [0.2]])
    # mean_ic = 0.3, mean_tau = 0.2 -> 1.5.
    assert compute_turnover_adjusted_ic_from_series(ic, tau, min_periods=2)[0] == pytest.approx(1.5)
    # IC gate: fewer than min_periods finite days -> NaN.
    assert np.isnan(compute_turnover_adjusted_ic_from_series(ic, tau, min_periods=3)[0])
    assert np.isnan(compute_turnover_adjusted_ic_from_series(ic, tau)[0])
    # Non-positive mean turnover -> NaN.
    zero = compute_turnover_adjusted_ic_from_series(
        np.ones((3, 1)), np.zeros((3, 1)), min_periods=2
    )
    assert np.isnan(zero[0])


def test_ic_decay_from_mean_ics_passthrough():
    mean_ics = np.array([[0.05, 0.03], [np.nan, 0.01], [0.02, np.inf]])
    out = compute_ic_decay_from_mean_ics(mean_ics, horizons=(1, 5, 20))
    assert out.shape == (3, 2)
    assert out[0, 0] == pytest.approx(0.05)
    assert np.isnan(out[1, 0]) and np.isnan(out[2, 1])
    assert out[2, 0] == pytest.approx(0.02)
    with pytest.raises(ValueError):
        compute_ic_decay_from_mean_ics(mean_ics, horizons=(5, 1, 20))
    with pytest.raises(ValueError):
        compute_ic_decay_from_mean_ics(mean_ics, horizons=(1, 5))


def test_hhi_bindings_known_values():
    # T=2, N=2, F=1: equal absolute exposure every day -> HHI_t = 0.5,
    # N_eff_t = 2.0; a day whose total gross exposure is 0 is NaN and excluded.
    values = np.ones((2, 2, 1))
    times = AxisRef("t", "int", 2, np.arange(2))
    assets = AxisRef("a", "str", 2, ("a0", "a1"))
    fb = FactorBatch(("f0",), times, assets, values)
    assert get_metric("hhi_concentration").compute_fn(factor_batch=fb)[0] == pytest.approx(0.5)
    assert get_metric("hhi_effective_n").compute_fn(factor_batch=fb)[0] == pytest.approx(2.0)

    values_zero = values.copy()
    values_zero[1, :, 0] = 0.0  # second day: zero total gross exposure -> NaN
    fb_zero = FactorBatch(("f0",), times, assets, values_zero)
    conc = get_metric("hhi_concentration").compute_fn(factor_batch=fb_zero)
    assert conc[0] == pytest.approx(0.5)  # only day 0 survives, mean unchanged
    eff = get_metric("hhi_effective_n").compute_fn(factor_batch=fb_zero)
    assert eff[0] == pytest.approx(2.0)


def test_hhi_binding_matches_daily_kernel(qe):
    fb, _ = qe
    values = np.asarray(fb.values, dtype=np.float64)
    for f in range(values.shape[2]):
        hhi = compute_concentration_hhi(values[:, :, f])
        expected = np.mean(hhi[np.isfinite(hhi)])
        got = np.asarray(get_metric("hhi_concentration").compute_fn(factor_batch=fb))[f]
        assert got == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Kernels: seed equivalence against naive in-test references.
# ---------------------------------------------------------------------------


def _reference_quantile_rank_stability(daily: np.ndarray, min_common: int = 3) -> float:
    panel = daily if daily.ndim == 2 else daily[:, :, 0]
    days = [t for t in range(panel.shape[0]) if np.isfinite(panel[t]).sum() >= min_common]
    corrs = []
    for a in range(len(days)):
        for b in range(a + 1, len(days)):
            ra, rb = panel[days[a]], panel[days[b]]
            common = np.isfinite(ra) & np.isfinite(rb)
            if common.sum() < min_common:
                continue
            ranks_a = rankdata(ra[common], method="average")
            ranks_b = rankdata(rb[common], method="average")
            if np.std(ranks_a) == 0 or np.std(ranks_b) == 0:
                continue
            rho = float(np.corrcoef(ranks_a, ranks_b)[0, 1])
            if np.isfinite(rho):
                corrs.append(rho)
    return float(np.mean(corrs)) if corrs else np.nan


@pytest.mark.parametrize("seed", [11, 2026, 90924])
def test_quantile_rank_stability_seed_equivalence(seed):
    rng = np.random.default_rng(seed)
    daily = rng.normal(size=(30, 5, 3))
    mask = rng.random(daily.shape) < 0.15
    daily[mask] = np.nan
    got = compute_quantile_rank_stability(daily)
    for f in range(3):
        expected = _reference_quantile_rank_stability(daily[:, :, f])
        if np.isnan(expected):
            assert np.isnan(got[f])
        else:
            assert got[f] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("seed", [11, 2026, 90924])
def test_ic_stability_scalar_seed_equivalence(seed):
    rng = np.random.default_rng(seed)
    ic = rng.normal(size=(120, 2))
    ic[rng.random(ic.shape) < 0.2] = np.nan
    got = compute_ic_stability_scalar(ic, window_size=60, min_periods=10)
    series, _ = compute_ic_stability(ic, window_size=60, min_periods=10)
    for f in range(2):
        finite = series[np.isfinite(series[:, f]), f]
        if finite.size == 0:
            assert np.isnan(got[f])
        else:
            assert got[f] == pytest.approx(float(np.mean(finite)), abs=1e-15)


@pytest.mark.parametrize("seed", [3, 42, 777])
def test_turnover_stability_and_adjusted_ic_seed_equivalence(seed):
    rng = np.random.default_rng(seed)
    times = AxisRef("t", "int", _T, np.arange(_T))
    assets = AxisRef("a", "str", _N, tuple(f"s{i}" for i in range(_N)))
    values = rng.normal(size=(_T, _N, 2))
    values[rng.random(values.shape) < 0.05] = np.nan
    fb = FactorBatch(("f0", "f1"), times, assets, np.ascontiguousarray(values))
    tau = estimate_turnover_from_ranks(fb)
    got_var = compute_turnover_stability_from_series(tau)
    for f in range(2):
        finite = tau[np.isfinite(tau[:, f]), f]
        expected = np.var(finite, ddof=0) if finite.size >= 2 else np.nan
        assert got_var[f] == pytest.approx(expected, nan_ok=True)

    ic = rng.normal(size=(_T, 2))
    ic[rng.random(ic.shape) < 0.1] = np.nan
    got_adj = compute_turnover_adjusted_ic_from_series(ic, tau)
    for f in range(2):
        ic_fin = ic[np.isfinite(ic[:, f]), f]
        tau_fin = tau[np.isfinite(tau[:, f]), f]
        if ic_fin.size < 20 or tau_fin.size < 2 or np.mean(tau_fin) <= 0:
            assert np.isnan(got_adj[f])
        else:
            assert got_adj[f] == pytest.approx(np.mean(ic_fin) / np.mean(tau_fin))


@pytest.mark.parametrize("seed", [5, 55, 555])
def test_return_coverage_seed_equivalence(seed):
    rng = np.random.default_rng(seed)
    panel = rng.normal(size=(40, 12))
    panel[rng.random(panel.shape) < 0.3] = np.nan
    assert compute_return_coverage(panel) == pytest.approx(np.isfinite(panel).mean())


# ---------------------------------------------------------------------------
# Kernels: dtype rejection, shape validation, determinism.
# ---------------------------------------------------------------------------


_BAD_INPUTS = {
    "bool": lambda shape: np.zeros(shape, dtype=bool),
    "object": lambda shape: np.array(["x"] * int(np.prod(shape)), dtype=object).reshape(shape),
    "complex": lambda shape: np.zeros(shape, dtype=complex),
}


# bool inputs hit the IC-family ValueError gate first; object/complex hit the
# TypeError gate.  Both rejection paths count as dtype rejection.
_DTYPE_ERRORS = (TypeError, ValueError)


@pytest.mark.parametrize("kind", sorted(_BAD_INPUTS))
def test_new_kernels_reject_bool_object_complex(kind):
    make = _BAD_INPUTS[kind]
    cases = (
        ("ic_summary", lambda a: compute_ic_summary_stats(a), (10, 2)),
        ("ic_stability_scalar", lambda a: compute_ic_stability_scalar(a, 40, 2), (40, 2)),
        ("quantile_rank_stability", lambda a: compute_quantile_rank_stability(a), (10, 3, 2)),
        ("ic_decay", lambda a: compute_ic_decay_from_mean_ics(a), (3, 2)),
        ("turnover_stability", lambda a: compute_turnover_stability_from_series(a), (10, 2)),
        ("turnover_adjusted_ic", lambda a: compute_turnover_adjusted_ic_from_series(a, a), (10, 2)),
        ("return_coverage", lambda a: compute_return_coverage(a), (5, 4)),
    )
    for name, fn, shape in cases:
        with pytest.raises(_DTYPE_ERRORS):
            fn(make(shape))
            pytest.fail(f"{name} accepted {kind} input")


def test_new_kernels_reject_bad_min_periods_and_shapes():
    ic = np.ones((10, 2))
    with pytest.raises(TypeError):
        compute_ic_summary_stats(ic, min_periods=True)
    with pytest.raises(ValueError):
        compute_ic_summary_stats(ic, min_periods=1)
    with pytest.raises(ValueError):
        compute_ic_summary_stats(ic.ravel())
    with pytest.raises(ValueError):
        compute_quantile_rank_stability(np.ones((4, 3)), min_periods=1)
    with pytest.raises(ValueError):
        compute_turnover_stability_from_series(ic, min_periods=1)
    with pytest.raises(ValueError):
        compute_turnover_adjusted_ic_from_series(ic, np.ones((5, 2)))
    with pytest.raises(ValueError):
        compute_return_coverage(np.ones(10))


@pytest.mark.parametrize(
    "call",
    [
        lambda ic: compute_ic_summary_stats(ic)["mean"],
        lambda ic: compute_ic_stability_scalar(ic, 40, 5),
        lambda ic: compute_quantile_rank_stability(ic[:, :, None]),
        lambda ic: compute_turnover_stability_from_series(np.abs(ic) * 0.01),
        lambda ic: compute_turnover_adjusted_ic_from_series(ic, np.abs(ic) * 0.01 + 0.05, min_periods=5),
    ],
)
def test_new_kernels_are_bit_deterministic(call):
    rng = np.random.default_rng(20260924)
    ic = rng.normal(size=(80, 1))
    ic[rng.random(ic.shape) < 0.2] = np.nan
    first, second = call(ic), call(ic)
    assert isinstance(first, np.ndarray)
    assert first.tobytes() == second.tobytes()


def test_ic_summary_stats_dict_is_bit_deterministic():
    rng = np.random.default_rng(20260924)
    ic = rng.normal(size=(80, 2))
    ic[rng.random(ic.shape) < 0.2] = np.nan
    first = compute_ic_summary_stats(ic)
    second = compute_ic_summary_stats(ic)
    assert sorted(first) == sorted(second)
    for key in first:
        assert first[key].tobytes() == second[key].tobytes()


def test_quantile_rank_stability_kernel_is_bit_deterministic():
    rng = np.random.default_rng(1)
    daily = rng.normal(size=(25, 4, 2))
    daily[rng.random(daily.shape) < 0.2] = np.nan
    a = compute_quantile_rank_stability(daily)
    b = compute_quantile_rank_stability(daily)
    assert a.tobytes() == b.tobytes()


# ---------------------------------------------------------------------------
# Registry bindings.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric_id, implementation_suffix",
    [
        ("ic_summary", "compute_ic_summary_stats"),
        ("ic_stability", "compute_ic_stability_scalar"),
        ("quantile_stability", "compute_quantile_rank_stability"),
        ("hhi_concentration", "compute_concentration_hhi"),
        ("hhi_effective_n", "compute_concentration_hhi"),
        ("return_coverage", "compute_return_coverage"),
        ("turnover_adjusted_ic", "compute_turnover_adjusted_ic_from_series"),
        ("turnover_stability", "compute_turnover_stability_from_series"),
        ("autocorrelation_ic", "compute_ic_autocorrelation"),
        ("ic_decay", "compute_ic_decay_from_mean_ics"),
    ],
)
def test_missing_kernel_ids_are_bound(metric_id, implementation_suffix):
    spec = get_metric(metric_id)
    assert spec.compute_fn is not None
    assert spec.implementation_id.endswith(implementation_suffix)
    assert spec.status.value == "stable"


# ---------------------------------------------------------------------------
# Public-facade reachability for the nine bound ids.
# ---------------------------------------------------------------------------


def test_facade_scalar_ids_route_to_factor_axis(qe):
    fb, lb = qe
    bundle = evaluate(fb, lb, metrics=_FACADE_SCALAR_IDS)
    for metric_id in _FACADE_SCALAR_IDS:
        artifact = bundle.artifacts[metric_id]
        assert isinstance(artifact, ScalarMetricArtifact), metric_id
        values = np.asarray(artifact.values, dtype=np.float64)
        assert values.shape == (_F,), metric_id


def test_autocorrelation_ic_routes_to_lag_axis_vector(qe):
    fb, lb = qe
    bundle = evaluate(fb, lb, metrics=("autocorrelation_ic",))
    artifact = bundle.artifacts["autocorrelation_ic"]
    assert isinstance(artifact, VectorMetricArtifact)
    values = np.asarray(artifact.values, dtype=np.float64)
    assert values.shape == (21, _F)  # max_lag (20) + 1 rows over the lag axis
    ic, _ = compute_daily_ic(fb, lb, method="pearson", min_assets=20)
    expected = compute_ic_autocorrelation(ic)
    assert np.array_equal(values, expected, equal_nan=True)


def test_facade_values_match_kernels(qe):
    fb, lb = qe
    bundle = evaluate(fb, lb, metrics=_FACADE_SCALAR_IDS)

    ic, _ = compute_daily_ic(fb, lb, method="pearson", min_assets=20)

    got = np.asarray(bundle.artifacts["ic_summary"].values)
    assert np.array_equal(got, compute_ic_summary_stats(ic)["mean"], equal_nan=True)

    got = np.asarray(bundle.artifacts["ic_stability"].values)
    assert np.array_equal(
        got, compute_ic_stability_scalar(ic), equal_nan=True
    )

    daily, _ = compute_quantile_returns_fast(fb, lb, n_quantiles=5, min_assets=10)
    got = np.asarray(bundle.artifacts["quantile_stability"].values)
    assert np.array_equal(
        got, compute_quantile_rank_stability(daily), equal_nan=True
    )

    got = np.asarray(bundle.artifacts["return_coverage"].values)
    expected = compute_return_coverage(lb.values)
    assert np.allclose(got, expected)

    tau = estimate_turnover_from_ranks(fb)
    got = np.asarray(bundle.artifacts["turnover_stability"].values)
    assert np.array_equal(
        got, compute_turnover_stability_from_series(tau), equal_nan=True
    )

    got = np.asarray(bundle.artifacts["turnover_adjusted_ic"].values)
    assert np.array_equal(
        got, compute_turnover_adjusted_ic_from_series(ic, tau), equal_nan=True
    )


def test_facade_bound_ids_are_deterministic(qe):
    fb, lb = qe
    ids = _FACADE_SCALAR_IDS + ("autocorrelation_ic",)
    first = evaluate(fb, lb, metrics=ids)
    second = evaluate(fb, lb, metrics=ids)
    for metric_id in ids:
        a = first.artifacts[metric_id].values
        b = second.artifacts[metric_id].values
        h1 = stable_content_hex(tag="missing-kernels.determinism", fields={"v": a})
        h2 = stable_content_hex(tag="missing-kernels.determinism", fields={"v": b})
        assert h1 == h2, metric_id


# ---------------------------------------------------------------------------
# ic_decay: kernel available, facade auto-dispatch deliberately unavailable.
# ---------------------------------------------------------------------------


def test_ic_decay_not_auto_dispatchable_through_evaluate(qe):
    fb, lb = qe
    with pytest.raises(InvalidContractError, match="HorizonMeanIC"):
        evaluate(fb, lb, metrics=("ic_decay",))


def test_ic_decay_kernel_matches_documented_formula(qe):
    fb, lb = qe
    # Pre-compute per-horizon mean ICs exactly as the doc formula defines
    # (finite-day mean of the daily IC per horizon) and check passthrough.
    rng = np.random.default_rng(20260924)
    mean_ics = np.vstack([
        np.nanmean(compute_daily_ic(fb, lb, method="pearson", min_assets=20)[0], axis=0),
        rng.normal(size=(1, _F)),
    ])
    out = compute_ic_decay_from_mean_ics(mean_ics, horizons=(1, 5))
    assert np.array_equal(out, np.where(np.isfinite(mean_ics), mean_ics, np.nan), equal_nan=True)
