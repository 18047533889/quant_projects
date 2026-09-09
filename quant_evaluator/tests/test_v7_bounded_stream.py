import numpy as np
import pytest
from quant_evaluator.runtime.streaming_evaluator import StreamingEvaluator, streaming_ic_updater
from quant_evaluator.planner.dependency_plan import MetricKind
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle

def chunks(total, width):
    for lo in range(0,total,width):
        hi=min(total,lo+width)
        rng=np.random.default_rng(lo)
        x=rng.normal(size=(hi-lo,20,2)); y=rng.normal(size=(hi-lo,20))
        times=tuple(range(lo,hi))
        yield (FactorBatch(('a','b'),AxisRef('t','int',hi-lo),AxisRef('n','int',20),x),
               LabelBundle('y',y,1,decision_time=times,label_start_time=times,
                           label_end_time=tuple(range(lo+1,hi+1))))

def evaluator():
    ev=StreamingEvaluator(chunk_size_time=4,chunk_size_factors=2)
    ev.register_streaming_metric('ic',streaming_ic_updater,MetricKind.IC)
    return ev

def test_summary_only_and_acknowledged_sink_do_not_retain_series():
    expected=[]
    for b,l in chunks(101,4):
        expected.extend([[np.corrcoef(b.values[t,:,f],l.values[t])[0,1] for f in range(2)]
                         for t in range(b.num_times)])
    expected=np.array(expected)
    calls=[]
    def sink(metric_id,factor_ids,times,values):
        assert len(times)<=4 and values.shape[1]==2
        assert metric_id=='ic' and factor_ids==('a','b')
        calls.append(len(times))
    result=evaluator().evaluate_stream(chunks(101,4),[{'metric_id':'ic','metric_kind':'ic'}],
                                      output_mode='sink',series_sink=sink)
    assert sum(calls)==101
    assert result.metadata['capabilities']['series_sink_metrics'] == ['ic']
    assert result.metadata['capabilities']['remove'] == 'unsupported'
    assert result.metadata['capabilities']['window'] == 'all_observations_in_call'
    for f,fid in enumerate(('a','b')):
        actual=result.metrics['ic'][fid]
        assert actual['count']==101
        assert actual['mean']==pytest.approx(expected[:,f].mean(),abs=1e-14)
        assert actual['std']==pytest.approx(expected[:,f].std(ddof=1),abs=1e-14)
    short=evaluator().evaluate_stream(chunks(9,4),[{'metric_id':'ic','metric_kind':'ic'}],output_mode='summary')
    assert result.peak_memory_mb < short.peak_memory_mb * 1.2

def test_sink_failure_stops_source_before_reading_next_chunk():
    fetched=[]
    def source():
        for pair in chunks(20,4):
            fetched.append(1)
            yield pair
    def fail(*args):
        raise RuntimeError('sink full')
    with pytest.raises(RuntimeError,match='sink full'):
        evaluator().evaluate_stream(source(),[{'metric_id':'ic','metric_kind':'ic'}],output_mode='sink',series_sink=fail)
    assert len(fetched)==1

@pytest.mark.parametrize('async_sink', [False, True])
def test_sink_requires_synchronous_completion(async_sink):
    from quant_evaluator.contracts.errors import InvalidContractError
    fetched=[]
    def source():
        for pair in chunks(20,4):
            fetched.append(1)
            yield pair
    async def deferred(*args):
        pass
    def unacknowledged(*args):
        return object()
    with pytest.raises(InvalidContractError, match='finish synchronously'):
        evaluator().evaluate_stream(source(), [{'metric_id':'ic','metric_kind':'ic'}],
                                   output_mode='sink',
                                   series_sink=deferred if async_sink else unacknowledged)
    assert len(fetched)==1
