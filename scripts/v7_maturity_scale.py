"""Synthetic rolling IC checkpoint scale receipt; isolated SQLite only."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import resource
import time

import numpy as np
from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.evaluation_refs import FactorValueRef, LabelBundleRef
from quant_evaluator.contracts.metric_instance import MetricInstance
from quant_evaluator.runtime.label_maturation import LabelMaturationQueue
from quant_platform.app.db.sqlite_backend import SqliteDb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--times', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    db_path = args.out / 'checkpoint.db'
    db = SqliteDb(str(db_path))
    queue = LabelMaturationQueue(db, max_factors_per_shard=8)
    start = datetime(2026,1,1,tzinfo=timezone.utc)
    factors = tuple(f'f{i}' for i in range(8))
    semantics = dict(factor_value_semantics='synthetic.v7', universe='synthetic20',
        clock='daily', window={'kind':'rolling_observations','size':32}, policy_version='research.v7')
    expected = []
    began = time.perf_counter()
    for lo in range(0,args.times,16):
        hi = min(args.times,lo+16)
        xs, ys = [], []
        for t in range(lo,hi):
            rng = np.random.default_rng(t)
            x = rng.normal(size=(20,8))
            y = .2*x[:,0]+rng.normal(size=20)
            xs.append(x); ys.append(y)
            if t >= args.times-32:
                expected.append([np.corrcoef(x[:,f],y)[0,1] for f in range(8)])
        dates = tuple(start+timedelta(days=t) for t in range(lo,hi))
        batch = FactorBatch(factors, AxisRef('time','datetime',hi-lo),
                            AxisRef('asset','int',20), np.stack(xs))
        labels = LabelBundle('y',np.stack(ys),1,decision_time=dates,
            label_start_time=dates,label_end_time=tuple(d+timedelta(days=1) for d in dates))
        ref = f'synthetic:{lo}:{hi}'
        request = EvaluationRequest(batch,labels,metric_ids=(),tier='research',
            metric_instances=(MetricInstance('pearson_ic_series',horizon=1),),
            factor_value_ref=FactorValueRef(ref,factors),label_bundle_ref=LabelBundleRef(ref,'y',1))
        stream = queue.enqueue(ref,request,stream_semantics=semantics,available_at=start)
        completed = queue.drain(start+timedelta(days=hi+1),lambda _:request)
        assert completed == (ref,)
    expected = np.asarray(expected)
    errors = []
    for (_,fid), summary in queue.summaries(stream).items():
        f = factors.index(fid)
        assert summary['count'] == min(args.times,32)
        errors.append(abs(summary['mean']-expected[:,f].mean()))
        assert np.isclose(summary['std'],expected[:,f].std(ddof=1),atol=1e-12)
    elapsed = time.perf_counter()-began
    state_rows = db.query('SELECT payload FROM qe_maturity_states')
    receipt = dict(scope='synthetic_single_CPU_rolling_Pearson_only',times=args.times,
        assets=20,factors=8,time_tile=16,window_observations=32,
        wall_seconds=elapsed,rss_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        state_rows=len(state_rows),state_json_bytes=sum(len(r['payload'].encode()) for r in state_rows),
        max_mean_absolute_error=max(errors),errors=0,retries=0,
        capabilities=dict(queue.capabilities),disk_scope='durable observations and event results intentionally retained')
    db.close()
    receipt['checkpoint_bytes'] = db_path.stat().st_size
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
