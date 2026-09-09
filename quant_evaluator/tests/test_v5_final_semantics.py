"""Independent final V5 public-entry numerical counterexamples."""
import json
from dataclasses import replace
import numpy as np
import pytest
from quant_evaluator.api.requests import EvaluationRequest, EvaluationBundle
from quant_evaluator.contracts.metric_instance import MetricInstance
from quant_evaluator.runtime.evaluator import evaluate
from quant_evaluator.tests.test_v3_public_artifacts import inputs


def test_t03_constant_ic_has_undefined_raw_ir_on_real_cpu_and_cuda():
    batch, label = inputs()
    values = np.broadcast_to(np.arange(batch.values.shape[1])[None,:,None], batch.values.shape).copy()
    batch = replace(batch, values=values)
    label = replace(label, values=values[:,:,0].copy())
    results = [evaluate(batch, label, metrics=["rank_ic", "ic_ir"], backend=b) for b in ("cpu", "cuda")]
    for result in results:
        np.testing.assert_allclose(result.artifacts["rank_ic"].values, 1.)
        assert np.isnan(result.artifacts["ic_ir"].values).all()
    for fid in batch.factor_ids:
        assert not results[0].get_metric("ic_ir",fid).valid
        assert not results[1].get_metric("ic_ir",fid).valid


@pytest.mark.parametrize("q", [5,10,20])
def test_daily_quantile_axes_survive_cpu_cuda_and_json_roundtrip(q):
    batch,label=inputs()
    item=MetricInstance("quantile_returns_daily",parameters={"n_quantiles":q})
    request=EvaluationRequest(batch,label,metric_instances=(item,),tier="research")
    cpu,gpu=(evaluate(request,backend=b) for b in ("cpu","cuda"))
    def artifact(bundle):
        return next(iter(bundle.instance_results.values())).artifacts[item.metric_id]
    assert artifact(cpu).values.size == len(batch.values)*q*len(batch.factor_ids)
    np.testing.assert_allclose(artifact(cpu).values,artifact(gpu).values,equal_nan=True,atol=1e-12)
    restored=EvaluationBundle.from_dict(json.loads(json.dumps(cpu.to_dict())))
    np.testing.assert_allclose(artifact(restored).values,artifact(cpu).values,equal_nan=True)
    for name in ("time_axis","quantile_axis","factor_axis"):
        assert getattr(artifact(restored),name)==getattr(artifact(cpu),name)
