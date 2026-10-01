"""Batch finalizers must not silently accept production Pandas fallbacks."""
from types import SimpleNamespace

import pytest

from factor_engine.runtime.production_policy import (
    ProductionPolicyViolation, assert_production_fastpath_runtime,
)


def _ctx(policy="error", fallback=True):
    return SimpleNamespace(
        production_fallback_policy=policy,
        runtime_stats={"production_pandas_fallbacks":
                       [{"op": "ts_mean", "requested": "polars", "actual": "pandas"}]
                       if fallback else []},
    )


@pytest.mark.parametrize("context", ["run_many:scheduler", "run_many", "run_many_parallel"])
def test_batch_finalizer_rejects_fallback_even_without_fastpath_gate(monkeypatch, context):
    monkeypatch.setenv("FACTOR_ENGINE_PRODUCTION_REQUIRE_FASTPATH", "0")
    with pytest.raises(ProductionPolicyViolation, match="ts_mean"):
        assert_production_fastpath_runtime(_ctx(), mode="production", context=context)


@pytest.mark.parametrize("mode,policy,fallback", [
    ("research", "error", True), ("production", "warn", True),
    ("production", "error", False),
])
def test_batch_finalizer_preserves_research_explicit_warn_and_native_paths(
    monkeypatch, mode, policy, fallback,
):
    monkeypatch.setenv("FACTOR_ENGINE_PRODUCTION_REQUIRE_FASTPATH", "0")
    assert_production_fastpath_runtime(_ctx(policy, fallback), mode=mode)
