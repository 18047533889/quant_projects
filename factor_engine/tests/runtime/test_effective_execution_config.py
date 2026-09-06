import pytest
from factor_engine.runtime.effective_execution_config import resolve_fast_execution_config


@pytest.mark.parametrize("parallel", ["thread", "auto", "hybrid"])
def test_parallel_is_honestly_resolved(parallel):
    config = resolve_fast_execution_config(parallel=parallel)
    assert config.requested_parallel == parallel
    assert config.parallel == "thread"


@pytest.mark.parametrize("kwargs", [dict(scheduler="serial"), dict(parallel="process"),
    dict(resource_profile="max"), dict(auto_shard=False), dict(result_policy="yield"),
    dict(write_results="false"), dict(native_fusion="false"), dict(auto_shard=1),
    dict(result_policy="return")])
def test_unsupported_or_untyped_requests_rejected(kwargs):
    with pytest.raises(ValueError):
        resolve_fast_execution_config(**kwargs)


def test_compute_only_sink_does_not_retain_and_explicit_return_does():
    assert not resolve_fast_execution_config(write_results=False).retain_results
    config = resolve_fast_execution_config(write_results=False, result_policy="return", native_fusion=False)
    assert config.retain_results and not config.native_fusion
    assert config.to_dict()["result_policy"] == "return"


def test_explicit_scheduler_policy_overrides_env(monkeypatch):
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
    monkeypatch.setenv("FACTOR_ENGINE_SCHEDULER", "process")
    scheduler = AdaptiveBatchScheduler(execution_policy="thread")
    try:
        assert scheduler._execution_policy == "thread"
    finally:
        scheduler.executor.shutdown()


def test_scheduler_failure_finishes_sink_without_masking_error(monkeypatch):
    from factor_engine.runtime.adaptive_batch_scheduler import AdaptiveBatchScheduler
    scheduler = AdaptiveBatchScheduler(execution_policy="thread")
    calls = []
    class Sink:
        def finish(self):
            calls.append("finish")
            raise RuntimeError("cleanup failure")
    original = ValueError("compute failure")
    def fail(*args, **kwargs):
        raise original
    monkeypatch.setattr(scheduler, "_run_impl", fail)
    try:
        with pytest.raises(ValueError) as raised:
            scheduler.run(None, backend=None, ctx=None, sink=Sink())
        assert raised.value is original
        assert calls == ["finish"]
        assert "cleanup failure" in str(raised.value.__notes__)
    finally:
        scheduler.executor.shutdown()
