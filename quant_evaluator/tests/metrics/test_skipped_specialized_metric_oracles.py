"""Independent public oracles for style exposure metrics in typed fixtures."""
from __future__ import annotations

import numpy as np

from quant_evaluator.contracts.artifact_types import ExecutablePortfolioArtifact
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.exposure_evidence import ExposurePanel
from quant_evaluator.runtime.evaluator import evaluate

_STYLES = ("industry", "size", "beta", "liquidity", "volatility", "momentum")
_METRICS = {
    "beta_exposure": "beta",
    "liquidity_exposure": "liquidity",
    "volatility_exposure": "volatility",
    "momentum_exposure": "momentum",
}
_T, _N, _F = 6, 32, 2


def _hadamard(order: int) -> np.ndarray:
    """Construct a small Sylvester matrix with centered orthogonal columns."""
    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < order:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def _contracts(style_order=None):
    style_order = tuple(range(len(_STYLES))) if style_order is None else tuple(style_order)
    names = tuple(_STYLES[index] for index in style_order)
    basis = _hadamard(_N)
    styles = basis[:, 1:7]
    residual = basis[:, 7]
    coefficients = np.array([
        [[.35, -.20, .10, .45, -.15, .25], [-.10, .40, .20, -.30, .50, -.25]],
        [[.20, -.35, .45, -.10, .30, -.40], [.25, .15, -.45, .35, -.20, .55]],
        [[-.25, .30, .20, .15, -.40, .35], [-.30, -.25, .35, .20, .45, -.10]],
        [[.45, -.10, -.25, .30, .20, -.35], [.15, .35, .10, -.50, .25, .40]],
        [[-.15, .25, .40, -.20, .35, .10], [.40, -.15, -.20, .45, -.30, .25]],
        [[.30, .20, -.10, .25, -.45, .50], [-.20, .45, .30, .10, .35, -.40]],
    ], dtype=np.float64)
    residual_scale = np.array([[.30, .45], [.20, .35], [.40, .25],
                               [.35, .50], [.25, .30], [.45, .20]])
    factor_values = np.einsum("nk,tfk->tnf", styles, coefficients)
    factor_values += residual[None, :, None] * residual_scale[:, None, :]
    factor_validity = np.ones_like(factor_values, dtype=bool)
    # Different factors miss different dates; the evaluator must omit each.
    factor_validity[1, :, 0] = False
    factor_validity[4, :, 1] = False
    factor_values[~factor_validity] = np.nan

    times_values = np.arange(_T, dtype=np.int64)
    assets_values = np.asarray([f"A{i:02d}" for i in range(_N)], dtype=object)
    times = AxisRef("time", "int64", _T, times_values)
    assets = AxisRef("asset", "object", _N, assets_values)
    factor_ids = ("factor-a", "factor-b")
    batch = FactorBatch(factor_ids, times, assets, factor_values,
                        validity=factor_validity)
    labels = LabelBundle(
        "forward", np.zeros((_T, _N)), 1,
        decision_time=tuple(int(value) for value in times_values),
        label_start_time=tuple(int(value) + 1 for value in times_values),
        label_end_time=tuple(int(value) + 2 for value in times_values),
        asset_axis=assets,
    )
    risk = np.broadcast_to(styles[None, :, :], (_T, _N, len(_STYLES))).copy()
    risk = risk[:, :, list(style_order)]
    panel = ExposurePanel(
        risk, style_names=names, source_ref="synthetic:orthogonal-risk",
        provider="independent-oracle", date_index=tuple(int(v) for v in times_values),
        security_ids=tuple(assets_values.tolist()), factor_ids=factor_ids,
        universe_snapshot_ref="synthetic:universe",
    )
    return batch, labels, panel, coefficients, residual_scale


def _independent_expected(coefficients, residual_scale, factor_validity):
    """Use orthogonal-design dot products, not QE exposure helpers."""
    result = {metric: np.empty(_F, dtype=np.float64) for metric in _METRICS}
    for factor_index in range(_F):
        per_time = []
        for time_index in range(_T):
            if not factor_validity[time_index, 0, factor_index]:
                continue
            beta = coefficients[time_index, factor_index]
            # Every style and the residual are centered, mutually orthogonal,
            # and have population variance one. OLS slopes equal beta, and
            # factor SD is sqrt(sum(beta**2) + residual_scale**2).
            factor_sd = np.sqrt(
                np.dot(beta, beta) + residual_scale[time_index, factor_index] ** 2
            )
            per_time.append(beta / factor_sd)
        mean_by_style = np.mean(per_time, axis=0)
        for metric, style in _METRICS.items():
            result[metric][factor_index] = mean_by_style[_STYLES.index(style)]
    return result


def _values(bundle, metric):
    return np.asarray([bundle.get_metric(metric, factor_id).value
                       for factor_id in ("factor-a", "factor-b")], dtype=np.float64)


def test_style_exposures_match_independent_oracle_with_missing_days_and_style_reordering():
    batch, labels, panel, coefficients, residual_scale = _contracts()
    expected = _independent_expected(coefficients, residual_scale, batch.validity)
    metrics = tuple(_METRICS)
    result = evaluate(batch, labels, metrics=metrics, exposure_panel=panel)
    repeated = evaluate(batch, labels, metrics=metrics, exposure_panel=panel)
    for metric in metrics:
        actual = _values(result, metric)
        np.testing.assert_allclose(actual, expected[metric], rtol=1e-11, atol=1e-12)
        np.testing.assert_array_equal(actual, _values(repeated, metric))
        assert [result.get_metric(metric, factor_id).observation_count
                for factor_id in ("factor-a", "factor-b")] == [5, 5]
        single = evaluate(batch, labels, metrics=(metric,), exposure_panel=panel)
        np.testing.assert_array_equal(actual, _values(single, metric))

    reordered = _contracts((5, 2, 0, 4, 1, 3))[2]
    reordered_result = evaluate(batch, labels, metrics=metrics, exposure_panel=reordered)
    for metric in metrics:
        np.testing.assert_allclose(_values(reordered_result, metric), expected[metric],
                                   rtol=1e-11, atol=1e-12)


def test_turnover_cost_batches_with_rank_ic_and_matches_single_requests():
    time_count, asset_count = 8, 32
    time_values = np.arange(time_count, dtype=np.int64)
    asset_values = np.asarray([f"asset-{index}" for index in range(asset_count)], dtype=object)
    times = AxisRef("time", "int64", time_count, time_values)
    assets = AxisRef("asset", "object", asset_count, asset_values)
    factor_values = np.empty((time_count, asset_count, 1), dtype=np.float64)
    labels = np.empty((time_count, asset_count), dtype=np.float64)
    cross_section = np.linspace(-1.0, 1.0, asset_count)
    for time_index in range(time_count):
        factor_values[time_index, :, 0] = cross_section + time_index * 0.01
        labels[time_index, :] = cross_section + np.sin(np.arange(asset_count) + time_index) * 0.05
    batch = FactorBatch(("factor-a",), times, assets, factor_values)
    label_bundle = LabelBundle(
        "forward", labels, 1,
        decision_time=tuple(int(value) for value in time_values),
        label_start_time=tuple(int(value) + 1 for value in time_values),
        label_end_time=tuple(int(value) + 2 for value in time_values),
        asset_axis=assets,
    )
    cost = ExecutablePortfolioArtifact(
        np.full((time_count, 1), 0.0005, dtype=np.float64),
        time_index=tuple(int(value) for value in time_values),
        factor_ids=("factor-a",),
        provenance={
            "leg": "cost_drag", "execution_certified": True,
            "cost_scope": "NET_EXECUTABLE",
            "execution_ref": "synthetic:execution-ledger",
        },
    )

    batched = evaluate(batch, label_bundle, metrics=("turnover_cost", "rank_ic"),
                       portfolio_returns=cost)
    turnover_only = evaluate(batch, label_bundle, metrics=("turnover_cost",),
                              portfolio_returns=cost)
    rank_ic_only = evaluate(batch, label_bundle, metrics=("rank_ic",),
                            portfolio_returns=cost)

    np.testing.assert_equal(batched.metric_values["turnover_cost"],
                            turnover_only.metric_values["turnover_cost"])
    np.testing.assert_equal(batched.metric_values["rank_ic"],
                            rank_ic_only.metric_values["rank_ic"])
    assert batched.metric_values["turnover_cost"].valid
    assert batched.metric_values["rank_ic"].valid
    assert batched.metric_values["turnover_cost"].value == 5.0
