"""Public-entrypoint A/B coverage for registry metrics outside the domain catalog.

The older all-metric suite enumerates the 140 domain-catalog IDs. These 24
additional registered IDs are classified explicitly so a new gap, a newly
bound input, or a changed registration cannot silently disappear from testing.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.catalog import list_all_metric_ids
from quant_evaluator.registry.metrics import get_metric, list_metrics
from quant_evaluator.runtime.evaluator import evaluate


_COMPUTABLE = {
    "block_bootstrap_ci", "coverage", "factor_turnover_rate", "hac_pvalue",
    "hac_tstat", "half_life", "ic_autocorr_lag1", "ic_ir", "ic_median",
    "ic_std", "mean_ic", "pearson_ic_ir", "pearson_ic_series",
    "pearson_ic_std", "quantile_returns_daily", "quantile_returns_full",
    "rank_ic_series", "rank_stability", "subsample_stability", "turnover",
}
_NEEDS_INPUT = {
    "information_ratio": "active",
    "mean_investment_fraction": "investment_fraction",
    "relative_max_drawdown": "relative_return",
    "tracking_error": "active",
}
_NO_IMPLEMENTATION = set()
_EXPECTED = _COMPUTABLE | set(_NEEDS_INPUT) | _NO_IMPLEMENTATION


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(20260926)
    t, n, f = 60, 25, 2
    times = AxisRef("t", "int", t, np.arange(t))
    assets = AxisRef("a", "str", n, tuple(f"s{i}" for i in range(n)))
    factors = FactorBatch(
        ("f0", "f1"), times, assets,
        np.ascontiguousarray(rng.normal(size=(t, n, f))),
    )
    labels = LabelBundle(
        target_id="r",
        values=np.ascontiguousarray(rng.normal(size=(t, n))),
        horizon=1,
        decision_time=tuple(range(t)),
        label_start_time=tuple(range(t)),
        label_end_time=tuple(range(1, t + 1)),
        asset_axis=assets,
    )
    series = evaluate(factors, labels, metrics=("long_short_returns",)).artifacts[
        "long_short_returns"
    ]
    probe = ProbePortfolioArtifact(
        np.ascontiguousarray(np.asarray(series.values, dtype=np.float64)),
        time_index=tuple(series.time_axis.time_index),
        factor_ids=("f0", "f1"),
    )
    return factors, labels, probe


def test_registry_only_inventory_and_declared_ability():
    actual = set(list_metrics()) - set(list_all_metric_ids())
    assert actual == _EXPECTED
    assert len(actual) == 24
    assert not (_COMPUTABLE & set(_NEEDS_INPUT))
    for metric_id in _COMPUTABLE | set(_NEEDS_INPUT):
        assert get_metric(metric_id).compute_fn is not None, metric_id
    for metric_id in _NO_IMPLEMENTATION:
        assert get_metric(metric_id).compute_fn is None, metric_id


@pytest.mark.parametrize("metric_id,leg", sorted(_NEEDS_INPUT.items()))
def test_explicit_portfolio_leg_requirement(panel, metric_id, leg):
    factors, labels, probe = panel
    with pytest.raises(Exception, match=f"requires explicitly bound portfolio leg {leg}"):
        evaluate(
            factors, labels, metrics=(metric_id,), portfolio_returns=probe,
        )


@pytest.fixture(scope="module")
def batch(panel):
    factors, labels, probe = panel
    return evaluate(
        factors, labels, metrics=tuple(sorted(_COMPUTABLE)),
        portfolio_returns=probe,
    )


@pytest.mark.parametrize("metric_id", sorted(_COMPUTABLE))
def test_registry_only_single_vs_batch_artifact_values(panel, batch, metric_id):
    factors, labels, probe = panel
    single = evaluate(
        factors, labels, metrics=(metric_id,), portfolio_returns=probe,
    )
    assert metric_id in batch.artifacts and metric_id in single.artifacts
    left = np.asarray(single.artifacts[metric_id].values)
    right = np.asarray(batch.artifacts[metric_id].values)
    assert left.shape == right.shape
    np.testing.assert_array_equal(left, right)
