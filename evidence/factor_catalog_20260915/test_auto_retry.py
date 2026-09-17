from types import SimpleNamespace
import pytest
import smoke_catalog as smoke


def test_auto_retry_preserves_physical_batch_api_and_route():
    factor = SimpleNamespace(name="one")
    class Engine:
        def run(self, factor):
            raise AssertionError("auto must not use logical run")
        def run_many(self, factors, *, enable_cse, result_policy, sink):
            assert factors == [factor] and enable_cse and result_policy == "sink"
            sink("one", [1.0, float("nan"), 3.0])
            return {"backend_paths": {"one": {"physical_plan": True, "primary_route": "polars_panel"}}}
    result = smoke.retry_factor_summary(Engine(), factor, requested_backend="auto")
    assert result["value_count"] == 3
    assert result["finite_count"] == 2
    assert result["backend_path"] == {"physical_plan": True, "primary_route": "polars_panel"}


def test_auto_retry_without_terminal_sink_is_not_success():
    class Engine:
        def run_many(self, *args, **kwargs):
            return {"backend_paths": {}}
    with pytest.raises(RuntimeError, match="terminal sink"):
        smoke.retry_factor_summary(Engine(), SimpleNamespace(name="one"), requested_backend="auto")


def test_explicit_pandas_retry_keeps_single_factor_api():
    class Engine:
        def run(self, factor):
            return [2.0, float("nan")]
    result = smoke.retry_factor_summary(Engine(), SimpleNamespace(name="one"), requested_backend="pandas")
    assert result["value_count"] == 2
    assert result["finite_count"] == 1
