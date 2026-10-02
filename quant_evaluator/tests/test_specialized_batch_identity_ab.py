"""Batch/single/repeat identity contracts for typed specialized inputs."""
from __future__ import annotations

import numpy as np
import pytest

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.portfolio_inputs import HoldingReturnPanel, PortfolioSpec
from quant_evaluator.metrics.catalog import list_all_metric_ids, get_metric_spec
from quant_evaluator.metrics.exposure_evidence import ExposurePanel
from quant_evaluator.runtime.evaluator import evaluate

T, N = 400, 40
EXPOSURE_IDS = tuple(sorted(i for i in list_all_metric_ids()
                            if "exposure_panel" in (get_metric_spec(i).requires or ())))
PROBE_IDS = tuple(sorted(i for i in list_all_metric_ids()
                         if "probe_pnl" in (get_metric_spec(i).requires or ()) and get_metric_spec(i).required_portfolio_leg is None))


@pytest.fixture(scope="module")
def typed_inputs():
    rng = np.random.default_rng(20261002)
    times = AxisRef("t", "int", T, np.arange(T, dtype=np.int64))
    assets = AxisRef("a", "str", N, tuple(f"s{i}" for i in range(N)))
    factor_ids = ("f0", "f1")
    factors = np.ascontiguousarray(rng.normal(size=(T, N, 2)))
    labels = np.ascontiguousarray(rng.normal(size=(T, N)) * .01)
    fb = FactorBatch(factor_ids, times, assets, factors)
    lb = LabelBundle(
        target_id="r", values=labels, horizon=1,
        decision_time=tuple(range(T)), label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)), asset_axis=assets,
    )
    exposures = np.ascontiguousarray(rng.normal(size=(T, N, 6)))
    panel = ExposurePanel(
        exposures, style_names=("size", "momentum", "volatility", "liquidity",
                                "industry", "beta"),
        source_ref="synthetic:ab", provider="test", date_index=tuple(range(T)),
        security_ids=tuple(assets.values), factor_ids=factor_ids,
        universe_snapshot_ref="synthetic:universe",
    )
    probe_values = np.ascontiguousarray(rng.normal(size=(T, 2)) * .01)
    probe = ProbePortfolioArtifact(probe_values, time_index=tuple(range(T)), factor_ids=factor_ids)
    holding = HoldingReturnPanel(
        np.ascontiguousarray(rng.normal(size=(T, N)) * .01), times, assets,
        "synthetic:ab", "close_to_close",
    )
    spec = PortfolioSpec(holding=2, n_quantiles=2, min_bucket_size=1,
                         long_weight=1.0, short_weight=0.0, per_side_cost=.001,
                         terminal_position_policy="liquidate_at_end")
    return fb, lb, panel, probe, holding, spec


def _fingerprint(bundle):
    payload = {}
    for key, item in sorted(bundle.metric_values.items()):
        value = getattr(item, "value", None)
        if isinstance(value, np.ndarray):
            payload["value::" + key] = stable_content_hex(tag="value", fields={"v": value})
        else:
            payload["value::" + key] = repr(value)
    for key, item in sorted(bundle.artifacts.items()):
        value = getattr(item, "values", None)
        if isinstance(value, np.ndarray):
            payload["artifact::" + key] = stable_content_hex(tag="artifact", fields={"v": value})
        else:
            payload["artifact::" + key] = repr(item)
    return stable_content_hex(tag="specialized.ab", fields=payload)


def _observable(bundle, metric_id):
    if metric_id in bundle.metric_values:
        return bundle.metric_values[metric_id].value
    artifact = bundle.artifacts.get(metric_id)
    assert artifact is not None, f"{metric_id}: no value or artifact"
    return getattr(artifact, "values", None)


def _assert_finite_output(value, metric_id):
    assert value is not None, f"{metric_id}: missing output"
    arr = np.asarray(value)
    assert arr.size and np.isfinite(arr).all(), f"{metric_id}: output contains missing/non-finite factor values: {arr}"


def _metric_evidence(bundle, metric_id):
    if metric_id in bundle.metric_values:
        metric = bundle.metric_values[metric_id]
        return (getattr(metric, "valid", None), getattr(metric, "observation_count", None),
                getattr(metric, "status", None))
    artifact = bundle.artifacts[metric_id]
    provenance = getattr(artifact, "provenance", {})
    return (getattr(artifact, "valid", None), getattr(artifact, "observation_count", None),
            getattr(artifact, "status", None), provenance.get("observation_counts"))


def _assert_same_evidence(single, batch, metric_id):
    a, b = _metric_evidence(single, metric_id), _metric_evidence(batch, metric_id)
    assert a == b, (metric_id, a, b)


def _config_hash(bundle):
    return bundle.metadata["execution_receipt"]["config_hash"]


def _assert_equal_output(a, b, metric_id):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True), metric_id
    else:
        assert a == b or (a is None and b is None), (metric_id, a, b)


def _assert_single_batch_repeat(ids, kwargs):
    fb, lb = kwargs.pop("_fb_lb")
    batch = evaluate(fb, lb, metrics=ids, **kwargs)
    repeated = evaluate(fb, lb, metrics=ids, **kwargs)
    assert _fingerprint(batch) == _fingerprint(repeated)
    for metric_id in ids:
        single = evaluate(fb, lb, metrics=(metric_id,), **kwargs)
        assert _fingerprint(single) == _fingerprint(evaluate(fb, lb, metrics=(metric_id,), **kwargs))
        a, b = _observable(single, metric_id), _observable(batch, metric_id)
        _assert_finite_output(a, metric_id)
        _assert_equal_output(a, b, metric_id)
        _assert_same_evidence(single, batch, metric_id)


def test_exposure_metrics_batch_single_repeat_and_source_perturbation(typed_inputs):
    fb, lb, panel, *_ = typed_inputs
    assert len(EXPOSURE_IDS) == 11, EXPOSURE_IDS
    kw = {"exposure_panel": panel}
    batch = evaluate(fb, lb, metrics=EXPOSURE_IDS, **kw)
    assert _fingerprint(batch) == _fingerprint(evaluate(fb, lb, metrics=EXPOSURE_IDS, **kw))
    for metric_id in EXPOSURE_IDS:
        single = evaluate(fb, lb, metrics=(metric_id,), **kw)
        single_again = evaluate(fb, lb, metrics=(metric_id,), **kw)
        assert _fingerprint(single) == _fingerprint(single_again), metric_id
        a, b = _observable(single, metric_id), _observable(batch, metric_id)
        _assert_finite_output(a, metric_id)
        _assert_equal_output(a, b, metric_id)
        _assert_same_evidence(single, batch, metric_id)
    changed = ExposurePanel(
        np.ascontiguousarray(panel.values + .25), style_names=panel.style_names,
        source_ref=panel.source_ref, provider=panel.provider, date_index=panel.date_index,
        security_ids=panel.security_ids, factor_ids=panel.factor_ids,
        universe_snapshot_ref=panel.universe_snapshot_ref,
    )
    changed_bundle = evaluate(fb, lb, metrics=("size_exposure",), exposure_panel=changed)
    original_bundle = evaluate(fb, lb, metrics=("size_exposure",), exposure_panel=panel)
    assert _fingerprint(changed_bundle) != _fingerprint(original_bundle)
    assert _config_hash(changed_bundle) != _config_hash(original_bundle)


def test_probe_metrics_batch_single_repeat_and_source_perturbation(typed_inputs):
    fb, lb, _, probe, *_ = typed_inputs
    assert len(PROBE_IDS) >= 11, PROBE_IDS
    kw = {"portfolio_returns": probe}
    batch = evaluate(fb, lb, metrics=PROBE_IDS, **kw)
    assert _fingerprint(batch) == _fingerprint(evaluate(fb, lb, metrics=PROBE_IDS, **kw))
    for metric_id in PROBE_IDS:
        one = evaluate(fb, lb, metrics=(metric_id,), **kw)
        again = evaluate(fb, lb, metrics=(metric_id,), **kw)
        assert _fingerprint(one) == _fingerprint(again), metric_id
        a, b = _observable(one, metric_id), _observable(batch, metric_id)
        _assert_finite_output(a, metric_id)
        _assert_equal_output(a, b, metric_id)
        _assert_same_evidence(one, batch, metric_id)
    changed_values = np.array(probe.values, copy=True)
    changed_values[0, :] += 0.1
    changed = ProbePortfolioArtifact(
        np.ascontiguousarray(changed_values), time_index=probe.time_index,
        factor_ids=probe.factor_ids,
    )
    changed_bundle = evaluate(fb, lb, metrics=("return_skew",), portfolio_returns=changed)
    original_bundle = evaluate(fb, lb, metrics=("return_skew",), portfolio_returns=probe)
    assert _fingerprint(changed_bundle) != _fingerprint(original_bundle)
    assert _config_hash(changed_bundle) != _config_hash(original_bundle)


def test_holding_input_batch_single_repeat(typed_inputs):
    fb, lb, _, _, holding, spec = typed_inputs
    fb = FactorBatch((fb.factor_ids[0],), fb.time_axis, fb.asset_axis, fb.values[:, :, :1])
    kw = {"holding_returns": holding, "portfolio_spec": spec}
    ids = tuple(i for i in list_all_metric_ids() if i in {
        "max_drawdown", "sharpe_ratio", "calmar_ratio", "sortino_ratio",
    })
    assert len(ids) >= 3
    _assert_single_batch_repeat(ids, {**kw, "_fb_lb": (fb, lb)})


def test_turnover_cost_repeat_matches_mean_cost_drag_oracle(typed_inputs):
    fb, lb, _, probe, *_ = typed_inputs
    cost_drag = ProbePortfolioArtifact(
        np.ascontiguousarray(np.abs(probe.values)), time_index=probe.time_index,
        factor_ids=probe.factor_ids, provenance={"leg": "cost_drag"},
    )
    metric_id = "turnover_cost"
    first = evaluate(fb, lb, metrics=(metric_id,), portfolio_returns=cost_drag)
    again = evaluate(fb, lb, metrics=(metric_id,), portfolio_returns=cost_drag)
    assert _fingerprint(first) == _fingerprint(again)
    observed = _observable(first, metric_id)
    _assert_finite_output(observed, metric_id)
    expected = np.mean(cost_drag.values, axis=0) * 10_000.0
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)


def test_holding_input_source_perturbation(typed_inputs):
    fb, lb, _, _, holding, spec = typed_inputs
    fb = FactorBatch((fb.factor_ids[0],), fb.time_axis, fb.asset_axis, fb.values[:, :, :1])
    changed = HoldingReturnPanel(
        np.ascontiguousarray(holding.values * 1.25), holding.time_axis, holding.asset_axis,
        holding.source_ref + ":changed", holding.price_basis,
    )
    a = evaluate(fb, lb, metrics=("sharpe_ratio",), holding_returns=holding, portfolio_spec=spec)
    b = evaluate(fb, lb, metrics=("sharpe_ratio",), holding_returns=changed, portfolio_spec=spec)
    assert _fingerprint(a) != _fingerprint(b)
    assert _config_hash(a) != _config_hash(b)
