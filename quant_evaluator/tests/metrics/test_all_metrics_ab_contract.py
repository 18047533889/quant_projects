"""All-metric A/B contract suite.

Goal: every catalog metric is *exercised* by the default pytest run, and each
one must satisfy the invariants that make an evaluation trustworthy:

1. **Determinism** — the same request twice yields byte-identical metric
   values and byte-identical evidence hashes (no hidden state, no iteration
   order dependence, no salted hashing).
2. **Batch == single** — evaluating a batch of metrics yields the same value
   per metric as evaluating each metric alone (shared intermediates must not
   change results).
3. **Executable path** — every catalog metric either computes successfully or
   fails with a *declared contract requirement* (missing artifact input,
   unresolved parameter that the public API expects the caller to bind).
   A metric that fails with "function not registered" or an unexpected
   internal error is a real defect and fails this test.
4. **No catastrophic slowdown** — hot kernels have generous wall-clock
   ceilings so a regression back to a Python per-day loop is caught here.

The suite runs at S scale (T=250, N=50) to stay fast; heavy equivalence
testing for refactored kernels lives in the dedicated
``test_*_perf_equivalence.py`` / ``*_vectorized_equivalence.py`` modules.
"""

from __future__ import annotations

import re
import time
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.portfolio_inputs import HoldingReturnPanel, PortfolioSpec
from quant_evaluator.metrics.catalog import list_all_metric_ids
from quant_evaluator.metrics.generalization_evidence import build_train_validation_artifact
from quant_evaluator.runtime.evaluator import evaluate

_T, _N = 250, 50

# Metrics whose kernel was refactored in the 2026-09-22 performance round and
# therefore carry dedicated bit-level equivalence suites.  They are still
# exercised here for determinism, but their numbers are pinned elsewhere.
_REFACTORED = {
    "rank_ic", "pearson_ic", "ic_ir", "long_short_returns", "max_drawdown",
    "sharpe_ratio", "quantile_spread", "adaptive_quantile_count",
    "turnover_estimate", "calmar_ratio", "sortino_ratio",
}

# Generous ceilings (ms) for hot kernels at S scale.  These are ~5-10x the
# measured values; they exist to catch O(T*N*F) Python loops creeping back in,
# not to benchmark the machine.
_HOT_CEILINGS_MS = {
    "rank_ic": 120.0,
    "long_short_returns": 120.0,
    "adaptive_quantile_count": 300.0,
    "quantile_spread": 200.0,
    "max_drawdown": 120.0,
    "sharpe_ratio": 120.0,
}

# Contract failures that are legitimate (the metric needs an input the
# all-metric fixture does not, or a parameter the caller must bind).
_DECLARED_FAILURES = (
    "requires missing",
    "unresolved required parameters",
    "requires typed",
    "requires authoritative",
    "must be a positive integer",
    "Generalization evidence factor axis must match",
    "time_index entries must be timezone-aware",
    # The probe/calendar coordinate pair must be produced by the caller's own
    # data pipeline (session-aligned instants + matching snapshot); the
    # synthetic fixture is deliberately coordinate-mismatched.
    "Portfolio time coordinates do not match",
    "requires an explicit non-default metric_instance",
    "requires explicit factor identities",
)

# Catalog ids that carry ``status=STABLE`` but have NO compute function bound
# in the registry (``compute_fn is None``) — verified 2026-09-24 against
# registry/metrics.py after the catalog-binding round (16 of the 26 legacy
# unbound ids were bound to their documented kernels; see
# tests/metrics/test_catalog_bindings.py) and the 2026-09-24 missing-kernel
# round, which bound nine of the remaining ten (ic_summary, ic_stability,
# quantile_stability, hhi_concentration, hhi_effective_n, return_coverage,
# turnover_adjusted_ic, turnover_stability, autocorrelation_ic; see
# tests/metrics/test_missing_kernels.py).
#
# The single remaining declared gap:
#   - ic_decay: the bound kernel (compute_ic_decay_from_mean_ics) consumes
#     the pre-computed per-horizon mean-IC matrix via
#     ``requires=["HorizonMeanIC"]``; the documented formula needs one
#     LabelBundle per forward horizon, which the facade cannot synthesise,
#     so the id stays deliberately unavailable through evaluate() (API-layer
#     callers, e.g. api/horizons summarize_horizons, supply the means).
#
# The suite asserts the *actual* unbound set is a SUBSET of this list: a newly
# unbound/never-bound catalog metric fails the run, while fixing any of these
# (binding compute_fn) passes silently.  Do not extend this list without a
# recorded reason; that is the whole point of the guard.
_DECLARED_UNBOUND = {
    "ic_decay",
}


@pytest.fixture(scope="module")
def qe_inputs():
    rng = np.random.default_rng(20260923)
    times = AxisRef("t", "int", _T, np.arange(_T))
    assets = AxisRef("a", "str", _N, tuple(f"s{i}" for i in range(_N)))
    fvals = rng.normal(size=(_T, _N, 1))
    labels = rng.normal(size=(_T, _N))
    fvals[rng.random(fvals.shape) < 0.05] = np.nan
    labels[rng.random(labels.shape) < 0.05] = np.nan
    fb = FactorBatch(("prof",), times, assets, np.ascontiguousarray(fvals))
    lb = LabelBundle(
        target_id="r",
        values=np.ascontiguousarray(labels),
        horizon=1,
        decision_time=tuple(range(_T)),
        label_start_time=tuple(range(_T)),
        label_end_time=tuple(range(1, _T + 1)),
        asset_axis=assets,
    )

    first = evaluate(fb, lb, metrics=("long_short_returns",))
    series = first.artifacts["long_short_returns"]
    probe = ProbePortfolioArtifact(
        np.ascontiguousarray(np.asarray(series.values, dtype=np.float64)),
        time_index=tuple(series.time_axis.time_index),
        factor_ids=("prof",),
    )
    holding = HoldingReturnPanel(
        rng.normal(size=(_T, _N)) * 0.01, times, assets,
        "synthetic:ab-contract", "close_to_close",
    )
    spec = PortfolioSpec(
        holding=2, n_quantiles=2, min_bucket_size=1,
        long_weight=1.0, short_weight=0.0, per_side_cost=0.001,
        terminal_position_policy="liquidate_at_end",
    )

    generalization = build_train_validation_artifact(
        np.array([1.2, 0.8]), np.array([1.0, 0.4]),
        train_rankic=np.array([0.03, 0.02]), validation_rankic=np.array([0.02, 0.01]),
        train_icir=np.array([0.5, 0.3]), validation_icir=np.array([0.4, 0.2]),
        train_sharpe=np.array([1.5, 1.0]), validation_sharpe=np.array([1.2, 0.8]),
        train_shape=np.array([1.0, 0.9]), validation_shape=np.array([0.9, 0.8]),
        train_factor_ids=("f0", "f1"), validation_factor_ids=("f0", "f1"),
        train_factor_versions=("v1", "v1"), validation_factor_versions=("v1", "v1"),
        metric_instance="rank_ic:h1:daily",
        metric_instance_refs={
            "rankic": "ab:rankic", "icir": "ab:icir",
            "sharpe": "ab:sharpe", "shape": "ab:shape",
        },
    )

    # Calendar metrics need timezone-aware instants on the probe time axis.
    import datetime as _dt
    base = _dt.datetime(2024, 1, 1, 15, 0, tzinfo=_dt.timezone(_dt.timedelta(hours=8)))
    instants = tuple(base + _dt.timedelta(days=i) for i in range(_T))
    probe_cal = ProbePortfolioArtifact(
        np.ascontiguousarray(np.asarray(series.values, dtype=np.float64)),
        time_index=instants,
        factor_ids=("prof",),
    )
    from data_access.r30.calendar_snapshot import CalendarSnapshot
    days = tuple(d.date().isoformat() for d in instants)
    sessions = tuple((d, "09:30:00", "15:00:00", 1) for d in days)
    calendar = CalendarSnapshot(
        market="CN", source_version="ab-contract", timezone="Asia/Shanghai",
        trading_days=days, sessions=sessions, early_close=(), snapshot_id="ab-snap",
    )

    probe_group = dict(portfolio_returns=probe, generalization_evidence=generalization)
    calendar_group = dict(
        portfolio_returns=probe_cal, calendar_snapshot=calendar,
        generalization_evidence=generalization,
    )
    return fb, lb, probe_group, dict(
        holding_returns=holding, portfolio_spec=spec,
        generalization_evidence=generalization,
    ), calendar_group


def _fingerprint(bundle) -> str:
    """Content fingerprint of everything the caller can observe."""
    payload = {}
    for key, metric in sorted(bundle.metric_values.items()):
        payload[f"value::{key}"] = repr(getattr(metric, "value", None))
    for key, artifact in sorted(bundle.artifacts.items()):
        values = getattr(artifact, "values", None)
        if isinstance(values, np.ndarray):
            payload[f"artifact::{key}"] = stable_content_hex(
                tag="ab.artifact", fields={"values": values}
            )
        else:
            payload[f"artifact::{key}"] = repr(values)
    return stable_content_hex(tag="ab.fingerprint", fields=payload)


def _run(fb, lb, metric_id, qe_inputs):
    """Try each declared input group; return (bundle, None) or (None, error)."""
    fb, lb, probe_kw, holding_kw, calendar_kw = qe_inputs
    seen = []
    for kw in (probe_kw, holding_kw, calendar_kw, {}):
        try:
            return evaluate(fb, lb, metrics=(metric_id,), **kw), None
        except Exception as exc:  # noqa: BLE001
            text = f"{type(exc).__name__}: {exc}"
            if text not in seen:
                seen.append(text)
    # Report every distinct reason so a declared-requirement token is not
    # masked by the last (input-less) attempt.
    return None, " | ".join(seen)


@pytest.mark.parametrize("metric_id", sorted(list_all_metric_ids()))
def test_every_metric_is_executable_or_declares_its_requirement(metric_id, qe_inputs):
    """Every catalog metric must compute, or fail with a declared contract
    requirement (or be in the monitored unbound-legacy list).  Undeclared
    internal errors and *new* unbound catalog entries fail here."""
    fb, lb, _, _, _ = qe_inputs
    bundle, err = _run(fb, lb, metric_id, qe_inputs)
    if bundle is not None:
        assert metric_id not in _DECLARED_UNBOUND, (
            f"{metric_id} is listed in _DECLARED_UNBOUND but now evaluates "
            f"successfully - remove it from the declared-gap list"
        )
        return
    assert err is not None
    if metric_id in _DECLARED_UNBOUND:
        return
    assert not re.search(r"not registered", err), (
        f"{metric_id} has a catalog entry but no registered compute function, "
        f"and is NOT in _DECLARED_UNBOUND: {err}"
    )
    assert any(token in err for token in _DECLARED_FAILURES), (
        f"{metric_id} failed with an undeclared error, which is a defect: {err}"
    )


@pytest.mark.parametrize("metric_id", sorted(list_all_metric_ids()))
def test_metric_is_deterministic(metric_id, qe_inputs):
    """Two identical requests must produce byte-identical observable output."""
    fb, lb, _, _, _ = qe_inputs
    first, err = _run(fb, lb, metric_id, qe_inputs)
    if first is None:
        pytest.skip(f"{metric_id}: needs a specialised input ({err})")
    second, err2 = _run(fb, lb, metric_id, qe_inputs)
    assert err2 is None, f"{metric_id} second run failed: {err2}"
    assert _fingerprint(first) == _fingerprint(second), (
        f"{metric_id} is not deterministic: identical requests differ"
    )


_BATCH = sorted(
    {
        "rank_ic", "pearson_ic", "ic_ir", "ic_positive_ratio",
        "long_short_returns", "max_drawdown", "sharpe_ratio", "calmar_ratio",
        "quantile_spread", "quantile_monotonicity", "adaptive_quantile_count",
        "turnover_estimate", "effective_n", "missing_ratio", "hit_rate",
        "quantile_returns_daily", "factor_turnover_rate",
    }
    & set(list_all_metric_ids())
)


def test_batch_evaluation_equals_single_metric_evaluation(qe_inputs):
    """A batch request must return exactly what per-metric requests return.

    Shared intermediates inside one ``evaluate`` call (the IC series cache,
    exposure loading cache, quantile cache, ...) must never leak between
    metrics or change a metric's value.  Only metrics that are evaluable with
    this fixture take part, so the invariant is tested for whatever the
    runtime actually supports.
    """
    fb, lb, probe_kw, _, _ = qe_inputs
    evaluable = []
    for candidate in _BATCH:
        try:
            evaluate(fb, lb, metrics=(candidate,), **probe_kw)
            evaluable.append(candidate)
        except Exception:  # noqa: BLE001
            continue
    assert len(evaluable) >= 5, (
        f"expected at least 5 evaluable batch metrics, got {evaluable}"
    )
    batch = evaluate(fb, lb, metrics=tuple(evaluable), **probe_kw)
    for metric_id in evaluable:
        single = evaluate(fb, lb, metrics=(metric_id,), **probe_kw)
        for key, metric in single.metric_values.items():
            assert key in batch.metric_values, f"{metric_id}: {key} missing from batch"
            a = getattr(metric, "value", None)
            b = getattr(batch.metric_values[key], "value", None)
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True), (
                    f"{metric_id}: batch value differs from single value"
                )
            else:
                assert a == b or (a is None and b is None), (
                    f"{metric_id}: batch {b!r} != single {a!r}"
                )


@pytest.mark.parametrize("metric_id", sorted(_HOT_CEILINGS_MS))
def test_hot_kernel_has_no_python_loop_regression(metric_id, qe_inputs):
    """Hot kernels must stay well below the loop-era cost at S scale."""
    fb, lb, _, _, _ = qe_inputs
    bundle, err = _run(fb, lb, metric_id, qe_inputs)
    if bundle is None:
        pytest.skip(f"{metric_id}: unavailable in this fixture ({err})")
    best = min(
        _timeit(fb, lb, metric_id, qe_inputs) for _ in range(2)
    )
    ceiling = _HOT_CEILINGS_MS[metric_id]
    assert best * 1000 < ceiling, (
        f"{metric_id} took {best*1000:.1f}ms at T={_T}/N={_N} "
        f"(ceiling {ceiling}ms) - a Python per-day loop likely crept back in"
    )


def _timeit(fb, lb, metric_id, qe_inputs) -> float:
    _, _, probe_kw, holding_kw, calendar_kw = qe_inputs
    for kw in (probe_kw, holding_kw, calendar_kw, {}):
        try:
            start = time.perf_counter()
            evaluate(fb, lb, metrics=(metric_id,), **kw)
            return time.perf_counter() - start
        except Exception:  # noqa: BLE001
            continue
    return float("inf")


def test_evidence_hash_is_stable_across_calls(qe_inputs):
    """Artifact evidence hashes must be reproducible (content-addressed)."""
    fb, lb, _, _, _ = qe_inputs
    first = evaluate(fb, lb, metrics=("long_short_returns",))
    second = evaluate(fb, lb, metrics=("long_short_returns",))
    for key, artifact in first.artifacts.items():
        values = getattr(artifact, "values", None)
        if not isinstance(values, np.ndarray):
            continue
        h1 = stable_content_hex(tag="evidence", fields={"values": values})
        h2 = stable_content_hex(
            tag="evidence", fields={"values": second.artifacts[key].values}
        )
        assert h1 == h2, f"{key}: evidence hash is not reproducible"
