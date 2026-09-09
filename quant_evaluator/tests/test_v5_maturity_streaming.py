from dataclasses import replace
from datetime import datetime, timedelta, timezone
import numpy as np
import pytest

from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.evaluation_refs import FactorValueRef, LabelBundleRef
from quant_evaluator.contracts.metric_instance import MetricInstance
from quant_evaluator.runtime.label_maturation import LabelMaturationQueue
from quant_evaluator.runtime.online_moments import OnlineMoments
from quant_evaluator.runtime.streaming_evaluator import (StreamingMetricState, streaming_summary_updater,
    streaming_coverage_updater, StreamingEvaluator, streaming_ic_updater)
from quant_evaluator.planner.dependency_plan import MetricKind
from quant_platform.app.db.sqlite_backend import SqliteDb

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
SEMANTICS = dict(factor_value_semantics="raw.v1", universe="synthetic.40", clock="daily.complete",
                 window="all_history", policy_version="research.v5")


def request(start=0, end=12, reverse=False):
    rng = np.random.default_rng(812)
    x = rng.normal(size=(12,40))
    y = .2*x + rng.normal(size=x.shape)
    if reverse:
        y = -y
    dates = tuple(START+timedelta(days=i) for i in range(start,end))
    batch = FactorBatch(("positive","negative"), AxisRef("time","datetime",end-start), AxisRef("asset","int",40),
                        np.stack([x[start:end],-x[start:end]], axis=-1))
    label = LabelBundle("h10",y[start:end],10,decision_time=dates,label_start_time=dates,
                        label_end_time=tuple(d+timedelta(days=10) for d in dates))
    ref = f"r:{start}:{end}:{reverse}"
    return EvaluationRequest(batch,label,metric_ids=(),tier="research",
                             metric_instances=(MetricInstance("rank_ic_series",horizon=10),),
                             factor_value_ref=FactorValueRef(ref,batch.factor_ids),
                             label_bundle_ref=LabelBundleRef(ref,"h10",10))


def test_full_chunk_bar_restart_have_identical_per_factor_moments(tmp_path):
    summaries = []
    for width in (12,4,1):
        path = str(tmp_path/f"state-{width}.db")
        db = SqliteDb(path)
        queue = LabelMaturationQueue(db)
        refs = {}
        for lo in range(0,12,width):
            req = request(lo,min(lo+width,12))
            refs[req.factor_value_ref.factor_value_id] = req
            stream = queue.enqueue(f"event-{lo}",req,stream_semantics=SEMANTICS,available_at=START)
            assert stream == queue.enqueue(f"event-{lo}",req,stream_semantics=SEMANTICS,available_at=START)
        resolve = lambda ref: refs[ref.factor_value_ref.factor_value_id]
        assert queue.drain(START+timedelta(days=9),resolve) == ()
        assert queue.summaries(stream) == {}
        db.close()
        db = SqliteDb(path)
        queue = LabelMaturationQueue(db)
        assert len(queue.drain(START+timedelta(days=30),resolve)) == 12//width
        assert queue.drain(START+timedelta(days=30),resolve) == ()
        summaries.append(queue.summaries(stream))
        assert {v["count"] for v in summaries[-1].values()} == {12}
        db.close()
    for actual in summaries[1:]:
        for key, expected in summaries[0].items():
            for metric in expected:
                assert actual[key][metric] == pytest.approx(expected[metric],abs=1e-12)


def test_failure_rolls_back_checkpoint_and_revision_preserves_as_known(tmp_path):
    db = SqliteDb(str(tmp_path/"fault.db"))
    queue = LabelMaturationQueue(db)
    req = request()
    stream = queue.enqueue("a",req,stream_semantics=SEMANTICS,available_at=START)
    def crash(_):
        raise RuntimeError("crash before commit")
    with pytest.raises(RuntimeError,match="crash"):
        queue.drain(START+timedelta(days=40),lambda _:req,before_commit=crash)
    assert queue.summaries(stream) == {}
    assert db.query("SELECT status FROM qe_maturity_events")[0]["status"] == "PENDING"
    queue.drain(START+timedelta(days=40),lambda _:req)
    original = queue.summaries(stream)
    revised = request(reverse=True)
    other = queue.enqueue("revision",revised,stream_semantics=SEMANTICS,available_at=START+timedelta(days=40),
                          snapshot_mode="RESTATED_RESEARCH",revision_id="vendor-correction-v2")
    assert other != stream
    assert queue.drain(START+timedelta(days=39),lambda _:revised) == ()
    queue.drain(START+timedelta(days=41),lambda _:revised)
    assert queue.summaries(stream) == original
    for key in original:
        assert queue.summaries(other)[key]["mean"] == pytest.approx(-original[key]["mean"])
    with pytest.raises(ValueError,match="timezone"):
        queue.drain(datetime(2026,2,1),lambda _:req)


def test_overlap_cannot_duplicate_observations_or_silently_restate(tmp_path):
    queue = LabelMaturationQueue(SqliteDb(str(tmp_path/"overlap.db")))
    req = request()
    stream = queue.enqueue("first",req,stream_semantics=SEMANTICS,available_at=START)
    queue.drain(START+timedelta(days=40),lambda _:req)
    queue.enqueue("same-observations-new-envelope",req,stream_semantics=SEMANTICS,available_at=START)
    queue.drain(START+timedelta(days=40),lambda _:req)
    assert {v["count"] for v in queue.summaries(stream).values()} == {12}
    revised = request(reverse=True)
    queue.enqueue("bad-restatement",revised,stream_semantics=SEMANTICS,available_at=START)
    with pytest.raises(ValueError,match="immutable"):
        queue.drain(START+timedelta(days=40),lambda _:revised)


def test_centered_summary_and_coverage_are_per_factor():
    req = request()
    values = np.array(req.batch_or_factor_ids.values,copy=True)
    values[:,:,0] = 1e10 + np.arange(480).reshape(12,40)/8
    values[:,:,1] = np.nan
    batch = replace(req.batch_or_factor_ids,values=values)
    summary = streaming_summary_updater(StreamingMetricState("summary",MetricKind.SUMMARY),batch,req.label_bundle).finalize()
    assert summary["positive"]["std"] == pytest.approx(np.std(values[:,:,0],ddof=1),rel=1e-12)
    assert summary["negative"]["count"] == 0
    coverage = streaming_coverage_updater(StreamingMetricState("coverage",MetricKind.COVERAGE),batch,req.label_bundle).finalize()
    assert coverage == {"positive":1.,"negative":0.}
    whole, bars = OnlineMoments(), OnlineMoments()
    whole.update(values[:,:,0])
    for value in values[:,:,0].ravel():
        bars.update([value])
    assert bars.summary()["std"] == pytest.approx(whole.summary()["std"],rel=1e-12)


def test_arbitrary_tiny_spread_at_high_location_matches_extended_precision():
    values=1e12+np.random.default_rng(91).normal(0,.01,1000)
    stats=OnlineMoments()
    for value in values:
        stats.update([value])
    expected=float(np.std(values.astype(np.longdouble),ddof=1))
    assert stats.summary()["std"] == pytest.approx(expected,rel=1e-10)
