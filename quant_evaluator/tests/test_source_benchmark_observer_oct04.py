"""Observer lifecycle must not contaminate the full API wall timer."""
from types import SimpleNamespace

import pytest

from quant_evaluator.tests.test_benchmark_real_cos_source_batch import _fixtures
from quant_evaluator.scripts import benchmark_real_cos_source_batch as harness
from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy


@pytest.mark.parametrize("fail_phase", [None, "before", "after", "evaluate"])
def test_observer_order_timer_and_close(monkeypatch, fail_phase):
    dates, assets, _, labels, records, rows = _fixtures()
    events = []
    sources = []
    original = harness.RealCosSource

    def factory(*args, **kwargs):
        source = original(*args, **kwargs)
        sources.append(source)
        source.close = lambda: events.append("close")
        return source

    def observer(*, phase, source, labels, metrics, requested_tile_size, policy):
        events.append(phase)
        assert source is sources[0]
        assert labels is label_ref
        assert tuple(metrics) == ("pearson_ic",)
        assert requested_tile_size == 16
        assert isinstance(policy, GPUExecutionPolicy)
        if fail_phase == phase:
            raise RuntimeError(phase)
        return "captured-" + phase

    def evaluate(source, *args, **kwargs):
        events.append("evaluate")
        assert kwargs["max_tile_size"] == 16
        if fail_phase == "evaluate":
            raise RuntimeError("evaluate")
        source.reads.append((0, 2))
        return SimpleNamespace(metadata={"backend_used": "cpu",
            "effective_max_tile_size": 16, "factor_tiles_processed": 1})

    ticks = iter([10.0, 12.0])
    def timer():
        events.append("timer")
        return next(ticks)

    label_ref = labels
    monkeypatch.setattr(harness, "RealCosSource", factory)
    monkeypatch.setattr(harness, "evaluate_factor_source_batch", evaluate)
    monkeypatch.setattr(harness.time, "perf_counter", timer)
    def run():
        return harness.run_backend("cpu", records, rows, dates, assets.values,
            labels, "a" * 64, 16, 16, GPUExecutionPolicy(), ("pearson_ic",),
            context_observer=observer)

    if fail_phase:
        with pytest.raises(RuntimeError, match=fail_phase):
            run()
    else:
        _, report = run()
        assert report["seconds"] == 2.0
        assert report["context_before"] == "captured-before"
        assert report["context_after"] == "captured-after"
        assert events == ["before", "timer", "evaluate", "timer", "after", "close"]
    assert events[-1] == "close"
    if fail_phase == "before":
        assert events == ["before", "close"]


def test_auto_verification_forwards_exact_supplied_qualification(monkeypatch):
    dates, assets, _, labels, records, rows = _fixtures()
    token = (object(), object())
    observed = []

    def evaluate(source, *args, **kwargs):
        observed.append(kwargs)
        source.reads.append((0, 2))
        return SimpleNamespace(metadata={"backend_used": "cpu",
            "effective_max_tile_size": 16, "factor_tiles_processed": 1})

    monkeypatch.setattr(harness, "evaluate_factor_source_batch", evaluate)
    _, receipt = harness.run_backend("auto", records, rows, dates, assets.values,
        labels, "a" * 64, 16, 16, GPUExecutionPolicy(), ("pearson_ic",),
        source_qualification=token)
    assert observed[0]["source_qualification"] is token
    assert observed[0]["backend"] == "auto"
    assert receipt["backend_used"] == "cpu"
