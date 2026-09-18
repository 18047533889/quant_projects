import numpy as np
import pytest
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.metrics.ic import compute_daily_ic
from quant_evaluator.metrics.ic_summary import compute_icir
from quant_evaluator.metrics.turnover import compute_turnover, compute_turnover_series, compute_turnover_matrix_batch


@pytest.mark.parametrize("mean_id,ir_id,method", [("pearson_ic", "pearson_ic_ir", "pearson"), ("rank_ic", "ic_ir", "spearman")])
def test_sparse_mean_and_ir_share_sample(mean_id, ir_id, method):
    rng = np.random.default_rng(724)
    x = rng.normal(size=(40, 30))
    y = x + rng.normal(scale=0.3, size=x.shape)
    y[20:] = -0.5*x[20:] + rng.normal(size=x[20:].shape)
    x[:20, 15:] = np.nan
    batch = FactorBatch(("f",), AxisRef("time", "int", 40), AxisRef("asset", "int", 30), x[..., None])
    label = LabelBundle("forward", y, 1, decision_time=tuple(range(40)), label_start_time=tuple(range(1,41)), label_end_time=tuple(range(2,42)))
    out = evaluate(batch, label, metrics=[mean_id, ir_id], backend="cpu")
    series, _ = compute_daily_ic(batch, label, method=method, min_assets=20)
    mean = out.get_metric(mean_id, "f")
    ir = out.get_metric(ir_id, "f")
    assert mean.observation_count == ir.observation_count == 20
    assert mean.value == pytest.approx(np.nanmean(series))
    assert ir.value == pytest.approx(compute_icir(series)[0])


def test_unknown_weight_is_not_zero_turnover():
    w = np.array([[0.5,0.5,np.nan], [0.5,np.nan,0.5]])
    assert np.isnan(compute_turnover(w[0], w[1]))
    assert np.isnan(compute_turnover_series(w)[1])
    for size in (2, 12):
        assert np.isnan(compute_turnover_matrix_batch(np.repeat(w[...,None],size,axis=2))[1]).all()
    z = np.nan_to_num(w)
    assert compute_turnover(z[0],z[1]) == pytest.approx(0.5)


def test_signed_monotonicity_keeps_direction_and_requires_complete_evidence():
    from quant_evaluator.metrics.quantile_shape import compute_quantile_rank_monotonicity as signed
    from quant_evaluator.metrics.quantile_shape import compute_quantile_monotonicity as adjacent
    assert signed(np.arange(10.0))[0] == pytest.approx(1)
    assert signed(-np.arange(10.0))[0] == pytest.approx(-1)
    assert np.isnan(signed(np.ones(10))[0])
    assert np.isnan(signed(np.array([0.,1.,np.nan,3.]))[0])
    tail = np.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,-100.])
    assert adjacent(tail)[0] == pytest.approx(8/9)
    assert signed(tail)[0] == pytest.approx(0.4545454545454545)


def test_public_signed_metric_and_explicit_ic_policy():
    from quant_evaluator.tests.test_v3_public_artifacts import inputs
    batch, label = inputs()
    out = evaluate(batch, label, metrics=["quantile_rank_monotonicity"], backend="cpu")
    assert out.get_metric("quantile_rank_monotonicity", "up").value == pytest.approx(1)
    assert out.get_metric("quantile_rank_monotonicity", "down").value == pytest.approx(-1)
    assert out.get_metric("quantile_rank_monotonicity", "up").observation_count == 5
    out = evaluate(batch, label, metrics=["rank_ic", "ic_ir"], backend="cpu",
        metric_parameters={"rank_ic": {"min_assets": 10}, "ic_ir": {"min_assets": 10}})
    assert out.get_metric("rank_ic", "up").observation_count == out.get_metric("ic_ir", "up").observation_count
