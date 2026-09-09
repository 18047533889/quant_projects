import numpy as np
import pytest
from quant_evaluator.tests.test_v3_public_artifacts import inputs
from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.runtime.evaluator import evaluate


def test_batch_owns_snapshot_and_cannot_reenable_write():
    batch,labels = inputs()
    source = np.array(batch.values,copy=True)
    validity = np.ones(source.shape,dtype=bool)
    refs = {'source':{'version':'v1'}}
    snapshot = FactorBatch(batch.factor_ids,batch.time_axis,batch.asset_axis,source,validity,context_refs=refs)
    source[:] = 999
    validity[:] = False
    refs['source']['version'] = 'v2'
    np.testing.assert_equal(snapshot.values,batch.values)
    assert snapshot.validity.all()
    assert snapshot.context_refs['source']['version'] == 'v1'
    for array in [snapshot.values,snapshot.validity,labels.values]:
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_frozen_request_nested_policy_and_artifact_ownership():
    batch,labels = inputs()
    policy = {'rank_ic':{'min_assets':10}}
    request = EvaluationRequest(batch,labels,metric_ids=('rank_ic',),metric_parameters=policy)
    policy['rank_ic']['min_assets'] = 1000
    assert request.metric_parameters['rank_ic']['min_assets'] == 10
    with pytest.raises(TypeError):
        request.metric_parameters['rank_ic']['min_assets'] = 2
    bundle = evaluate(request)
    with pytest.raises(ValueError):
        bundle.artifacts['rank_ic'].values.setflags(write=True)
