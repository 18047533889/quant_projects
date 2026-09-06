import numpy as np
import pandas as pd
import pytest

from benchmarks.benchmark_run_many_streaming_20260906 import Source
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.production_policy import ProductionPolicyViolation


class CountingSource(Source):
    root = "/synthetic/public-audit"

    def __init__(self):
        index = pd.MultiIndex.from_product(
            [pd.date_range("2025-01-01", periods=3), ["A", "B"]],
            names=["timestamp", "instrument"],
        )
        super().__init__(pd.Series(np.arange(6, dtype=float), index=index))
        self.reads = 0

    def load_column(self, name):
        self.reads += 1
        return super().load_column(name)


def test_real_public_run_rejects_foreign_plan_before_source_io():
    source = CountingSource()
    engine = FactorEngine(backend=PandasBackend(), data_source=source, run_mode="research")
    factor = Factor(name="a", expr=col("close") + 1)
    other = Factor(name="b", expr=col("close") + 2)
    foreign_plan, foreign_analysis = engine.compile(other)
    with pytest.raises(ProductionPolicyViolation, match="binding mismatch"):
        engine.run(factor, plan=foreign_plan, analysis=foreign_analysis)
    assert source.reads == 0
    plan, analysis = engine.compile(factor)
    actual = engine.run(factor, plan=plan, analysis=analysis)["result"]
    pd.testing.assert_series_equal(actual, source.values + 1, check_names=False)
    assert source.reads > 0


def test_real_public_run_disables_cache_after_scope_identity_failure(monkeypatch):
    source = CountingSource()
    observed = []

    class ObservingBackend(PandasBackend):
        def execute(self, plan, ctx):
            observed.append(ctx.cache)
            return super().execute(plan, ctx)

    def invalid_identity(*args, **kwargs):
        raise ValueError("synthetic exact identity unavailable")

    monkeypatch.setattr("factor_engine.storage.data_scope.compute_execution_cache_scope", invalid_identity)
    engine = FactorEngine(backend=ObservingBackend(), data_source=source, cache=object(), run_mode="research")
    result = engine.run(Factor(name="f", expr=col("close") + 3))["result"]
    assert observed and all(cache is None for cache in observed)
    pd.testing.assert_series_equal(result, source.values + 3, check_names=False)
