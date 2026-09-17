from types import SimpleNamespace
import pytest
import smoke_catalog as smoke


def test_batch_smoke_uses_one_public_call_and_default_performance_options():
    factors = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
    calls = []
    def sink(name, value):
        calls.append((name, value))
    class Engine:
        def run_many(self, received, **kwargs):
            assert received is factors
            assert set(kwargs) == {"result_policy", "sink"}
            assert kwargs["result_policy"] == "sink"
            kwargs["sink"]("a", 1)
            kwargs["sink"]("b", 2)
            return {"dag": "actual"}
        def run(self, *args, **kwargs):
            raise AssertionError("must not silently use singleton API")
    assert smoke.execute_default_batch(Engine(), factors, sink) == {"dag": "actual"}
    assert calls == [("a", 1), ("b", 2)]


def test_strict_abort_keeps_success_and_never_attributes_shared_failure_to_a_factor():
    rows = {"a": {"status": "EXECUTED"}, "b": {}, "c": {"status": "COMPILE_FAILED"}}
    smoke.mark_batch_aborted(rows, ValueError("shared dependency failed"))
    assert rows["a"] == {"status": "EXECUTED"}
    assert rows["c"] == {"status": "COMPILE_FAILED"}
    assert rows["b"]["status"] == "BATCH_ABORTED"
    assert rows["b"]["retry_after_batch_abort"] is False
    assert rows["b"]["error_type"] == "ValueError"


def test_default_batch_propagates_failure_without_singleton_retry():
    class Engine:
        def run_many(self, *args, **kwargs):
            raise ValueError("failed")
        def run(self, *args, **kwargs):
            raise AssertionError("no retry")
    with pytest.raises(ValueError, match="failed"):
        smoke.execute_default_batch(Engine(), [object(), object()], lambda *a: None)
