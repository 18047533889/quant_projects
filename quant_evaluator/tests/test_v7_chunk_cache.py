import numpy as np
import pytest
from types import SimpleNamespace
from quant_evaluator.runtime.evaluator import Evaluator
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.errors import InvalidContractError

STATE = {'multiplier': 1.0}

def dynamic_metric(factor_batch):
    return STATE['multiplier']

def helper():
    return 1.0

def helper_metric(factor_batch):
    return helper()

def panel():
    values = np.arange(33., dtype=float).reshape(11, 1, 3)
    return (FactorBatch(factor_ids=('a','b','c'), time_axis=AxisRef('t','int',11),
                       asset_axis=AxisRef('n','int',1), values=values),
            LabelBundle(target_id='y', values=np.zeros((11,1)), horizon=1,
                        decision_time=tuple(range(11)), label_start_time=tuple(range(11)),
                        label_end_time=tuple(range(1,12))))

def test_dynamic_global_cannot_hit_stale_cache():
    ev=Evaluator(); ev.register_metric('dynamic',dynamic_metric)
    spec=[{'metric_id':'dynamic','metric_kind':'custom'}]
    args=panel()
    try:
        assert ev.evaluate(*args,spec).metrics['dynamic']==1
        STATE['multiplier']=2.
        assert ev.evaluate(*args,spec).metrics['dynamic']==2
    finally:
        STATE['multiplier']=1.

def test_helper_replacement_cannot_hit_stale_cache(monkeypatch):
    ev=Evaluator(); ev.register_metric('helper',helper_metric)
    spec=[{'metric_id':'helper','metric_kind':'custom'}]
    assert ev.evaluate(*panel(),spec).metrics['helper']==1
    monkeypatch.setitem(globals(),'helper',lambda: 2.)
    assert ev.evaluate(*panel(),spec).metrics['helper']==2

def test_underspecified_sum_fails_closed():
    node=SimpleNamespace(metadata={'partitionability':'PARTITIONABLE','aggregation':'sum'})
    graph=SimpleNamespace(get_node=lambda name:node)
    with pytest.raises(InvalidContractError):
        Evaluator()._aggregate_chunk_results([{'m':np.array([1,2])},{'m':np.array([3,4])}],['m'],graph)

def test_declared_time_sum_preserves_factor_axis():
    ev=Evaluator(max_chunk_memory_mb=.00015)
    ev.register_metric('m', lambda factor_batch: factor_batch.values.sum(axis=(0,1)))
    spec=[{'metric_id':'m','metric_kind':'custom','metadata':{
        'partitionability':'PARTITIONABLE', 'aggregation':'sum',
        'chunk_contract':{'version':1,'split_axis':'time','output_axes':['factor'],
                          'merge':'sum','halo':0}}}]
    full=ev.evaluate(*panel(),spec,use_chunking=False)
    chunk=ev.evaluate(*panel(),spec,use_chunking=True)
    assert chunk.chunks_processed>1
    np.testing.assert_array_equal(chunk.metrics['m'], full.metrics['m'])

def test_factor_concat_tail_and_reordered_descriptors():
    from quant_evaluator.planner.dependency_plan import resolve_metric_dependencies
    from quant_evaluator.planner.batch_plan import create_batch_plan
    ev=Evaluator(max_chunk_memory_mb=.0002)
    ev.register_metric('m',lambda factor_batch: factor_batch.values[:,0,:])
    spec=[{'metric_id':'m','metric_kind':'custom','metadata':{'partitionability':'PARTITIONABLE',
        'aggregation':'concat','chunk_contract':{'version':1,'split_axis':'factor',
        'output_axes':['time','factor'],'merge':'concat','halo':0}}}]
    batch,labels=panel()
    full=ev.evaluate(batch,labels,spec,use_chunking=False)
    tiled=ev.evaluate(batch,labels,spec)
    np.testing.assert_array_equal(tiled.metrics['m'],full.metrics['m'])
    assert tiled.chunks_processed==3
    graph=resolve_metric_dependencies(spec)
    plan=create_batch_plan(batch,time_chunk_size=11,asset_chunk_size=1,factor_chunk_size=2)
    plan.chunks.reverse()
    shuffled=ev._evaluate_chunked(batch,labels,graph,plan)
    np.testing.assert_array_equal(shuffled.metrics['m'],full.metrics['m'])

def test_undeclared_axis_split_rejected():
    from quant_evaluator.planner.dependency_plan import resolve_metric_dependencies
    from quant_evaluator.planner.batch_plan import create_batch_plan
    ev=Evaluator(); batch,_=panel()
    graph=resolve_metric_dependencies([{'metric_id':'m','metric_kind':'custom','metadata':{
        'partitionability':'PARTITIONABLE','aggregation':'concat','chunk_contract':{
        'version':1,'split_axis':'time','output_axes':['time','factor'],'merge':'concat','halo':0}}}])
    plan=create_batch_plan(batch,time_chunk_size=3,asset_chunk_size=1,factor_chunk_size=1)
    with pytest.raises(InvalidContractError,match='undeclared'):
        ev._validate_chunk_plan(plan,graph)

def test_factor_array_hash_once_per_batch_for_many_metrics(monkeypatch):
    import quant_evaluator.runtime.evaluator as module
    batch,labels=panel(); ev=Evaluator()
    calls=[]; real=module.authoritative_array_hash
    def counted(value):
        if value is batch.values:
            calls.append(1)
        return real(value)
    monkeypatch.setattr(module,'authoritative_array_hash',counted)
    specs=[]
    for i in range(8):
        name=f'm{i}'; ev.register_metric(name,lambda factor_batch: 1.)
        specs.append({'metric_id':name,'metric_kind':'custom','metadata':{
            'cache_contract':{'version':1,'semantic_ref':'constant-one.v1','dependency_refs':{}}}})
    result=ev.evaluate(batch,labels,specs)
    assert len(calls)==1
    assert result.metrics=={f'm{i}':1. for i in range(8)}
    assert ev._batch_identities=={}

def test_upstream_change_invalidates_cached_dependent():
    ev=Evaluator()
    ev.register_metric('up',lambda factor_batch: 1.)
    ev.register_metric('down',lambda up: up*10.)
    contract={'cache_contract':{'version':1,'semantic_ref':'pure-test.v1','dependency_refs':{}}}
    specs=[{'metric_id':'up','metric_kind':'custom','metadata':contract},
           {'metric_id':'down','metric_kind':'custom','dependencies':['up'],'metadata':contract}]
    assert ev.evaluate(*panel(),specs).metrics['down']==10.
    ev.register_metric('up',lambda factor_batch: 2.)
    assert ev.evaluate(*panel(),specs).metrics['down']==20.

def test_invalid_signature_rejected_even_if_cache_offers_value():
    class PoisonCache:
        def get(self,key): return 99.
        def put(self,*args): pass
    ev=Evaluator(cache=PoisonCache())
    ev.register_metric('m',lambda missing, **kwargs: missing)
    with pytest.raises(InvalidContractError,match='unresolved'):
        ev.evaluate(*panel(),[{'metric_id':'m','metric_kind':'custom'}])
