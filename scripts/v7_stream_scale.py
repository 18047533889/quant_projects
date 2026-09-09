"""Measured bounded-source/synchronous-sink receipt; synthetic CPU Pearson only."""
import argparse
import json
import resource
import time
from pathlib import Path
import numpy as np
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.streaming_evaluator import StreamingEvaluator, streaming_ic_updater
from quant_evaluator.planner.dependency_plan import MetricKind

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rows',type=int,required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    counters={'input_bytes':0,'sink_bytes':0,'sink_calls':0,'source_calls':0}
    width,n,f=16,20,8
    def source():
        for lo in range(0,args.rows,width):
            hi=min(args.rows,lo+width); times=tuple(range(lo,hi))
            rng=np.random.default_rng(20260908+lo)
            x=rng.normal(size=(hi-lo,n,f)); y=rng.normal(size=(hi-lo,n))
            counters['input_bytes']+=x.nbytes+y.nbytes
            counters['source_calls']+=1
            yield (FactorBatch(tuple(f'f{i}' for i in range(f)),AxisRef('time','int',hi-lo),
                    AxisRef('asset','int',n),x),
                   LabelBundle('h1',y,1,decision_time=times,label_start_time=times,
                               label_end_time=tuple(range(lo+1,hi+1))))
    def sink(metric,factors,times,values):
        path=out/f'{times[0]:08d}.npy'
        with path.open('xb') as stream:
            np.save(stream,values,allow_pickle=False)
        counters['sink_bytes']+=path.stat().st_size
        counters['sink_calls']+=1
    ev=StreamingEvaluator(chunk_size_time=width,chunk_size_factors=f)
    ev.register_streaming_metric('pearson_ic',streaming_ic_updater,MetricKind.IC)
    started=time.perf_counter()
    result=ev.evaluate_stream(source(),[{'metric_id':'pearson_ic','metric_kind':'ic'}],
                              output_mode='sink',series_sink=sink)
    payload={'schema':'v7-bounded-stream-benchmark.v1','shape':[args.rows,n,f],
        'tile':[width,n,f],'dtype':'float64','backend':'cpu','cache':'none/cold',
        'wall_seconds':time.perf_counter()-started,
        'rss_peak_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'tracked_input_state_peak_mib':result.peak_memory_mb,
        'vram_bytes':0,'host_device_transfer_bytes':0, 'error_count':0,'retry_count':0,
        'summary':result.metrics,**counters,
        'limitations':'Synthetic complete cross-section Pearson IC only; no production dataset, all-metric, or multi-GPU certification.'}
    print(json.dumps(payload,sort_keys=True))

if __name__=='__main__':
    main()
