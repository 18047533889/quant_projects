"""Dispatch regression: probe-pnl metrics must be reachable through the
public ``evaluate`` facade.

Background (found 2026-09-23): ``_prepare_metric_call``'s unresolved-required
-parameter gate ran before the metric-facade wrapper, so any metric whose
compute function takes a required ``returns`` argument was rejected outright
when it declared ``requires=["probe_pnl"]``.  Eleven catalog metrics were
unreachable through the public API as a result, even with a perfectly good
``ProbePortfolioArtifact`` supplied:

    worst_quarter, worst_month, worst_12m, time_to_recovery,
    rolling_1y_sharpe_q10, rolling_1y_sharpe_min, return_skew,
    max_underwater_duration, mean_underwater_duration,
    cvar_expected_shortfall, downside_deviation

``runtime/evaluator.py`` now exempts facade-bound parameter names from that
gate (``_FACADE_BOUND_PARAMETERS``).  These tests pin both halves of the
contract: metrics compute when the artifact is supplied, and they still fail
closed (with the precise artifact error, not a confusing parameter error)
when it is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate

# The eleven metrics that were unreachable before the dispatch fix.
PROBE_PNL_METRICS = (
    "worst_quarter",
    "worst_month",
    "worst_12m",
    "time_to_recovery",
    "rolling_1y_sharpe_q10",
    "rolling_1y_sharpe_min",
    "return_skew",
    "max_underwater_duration",
    "mean_underwater_duration",
    "cvar_expected_shortfall",
    "downside_deviation",
)

_T, _N = 400, 40


def _fixtures():
    rng = np.random.default_rng(20260923)
    times = AxisRef("t", "int", _T, np.arange(_T))
    assets = AxisRef("a", "str", _N, tuple(f"s{i}" for i in range(_N)))
    fvals = rng.normal(size=(_T, _N, 1))
    labels = rng.normal(size=(_T, _N)) * 0.01
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
    # A well-formed probe portfolio: a real long/short return stream.
    returns = rng.normal(size=_T) * 0.01
    probe = ProbePortfolioArtifact(
        np.ascontiguousarray(returns[:, None].astype(np.float64)),
        time_index=tuple(range(_T)),
        factor_ids=("prof",),
    )
    return fb, lb, probe


@pytest.mark.parametrize("metric_id", PROBE_PNL_METRICS)
def test_probe_pnl_metric_is_reachable_with_artifact(metric_id):
    """Each metric must compute through the public facade given the artifact."""
    fb, lb, probe = _fixtures()
    bundle = evaluate(fb, lb, metrics=(metric_id,), portfolio_returns=probe)
    assert metric_id in bundle.metric_values, (
        f"{metric_id} produced no metric value through the public facade"
    )
    value = bundle.metric_values[metric_id].value
    # A 400-day stream clears every min_periods in this family; require a
    # real number rather than a silently-missing value.
    assert value is not None
    assert np.isfinite(float(value)), f"{metric_id} produced a non-finite value"


@pytest.mark.parametrize("metric_id", PROBE_PNL_METRICS)
def test_probe_pnl_metric_fails_closed_without_artifact(metric_id):
    """Without the artifact the failure must name the missing artifact."""
    fb, lb, _ = _fixtures()
    with pytest.raises(InvalidContractError) as excinfo:
        evaluate(fb, lb, metrics=(metric_id,))
    message = str(excinfo.value)
    assert "ProbePortfolioArtifact" in message or "probe_pnl" in message, (
        f"{metric_id} failed with a confusing message instead of the missing "
        f"artifact: {message}"
    )
    # The old defect surfaced as "unresolved required parameters: returns".
    assert "unresolved required parameters" not in message


def test_metrics_with_defaulted_returns_still_resolve():
    """Metrics whose ``returns`` parameter is bound by the wrapper (no
    ``probe_pnl`` declaration) must keep working - no regression from the
    gate change."""
    fb, lb, probe = _fixtures()
    for metric_id in ("max_drawdown", "sharpe_ratio", "calmar_ratio", "sortino_ratio"):
        bundle = evaluate(fb, lb, metrics=(metric_id,), portfolio_returns=probe)
        assert metric_id in bundle.metric_values
