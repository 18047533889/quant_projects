"""Small actual run_many tests for final backend routing, not just kernels."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from factor_engine.api.cleaned_ops import make_cleaned_call_factory as F
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.resource_broker import ResourceBroker
from tests.helpers import InMemorySeriesSource


def _coherent_broker() -> ResourceBroker:
    """Keep this numeric contract test independent of live host pressure."""
    broker = ResourceBroker(
        hard_memory_limit=8 * 1024**3,
        cpu_slots=4,
        min_host_reserve_gb=0,
        min_host_reserve_fraction=0,
    )
    hard = broker.hard_memory_limit
    snapshot = replace(
        broker.snapshot(),
        hard_memory_limit=hard,
        cgroup_memory_current=0,
        host_mem_available=hard,
        process_rss=0,
        worker_rss=0,
        process_family_rss=0,
        process_family_pss=0,
        system_cpu_util=0.0,
        our_cpu_util=0.0,
        external_cpu_util=0.0,
        host_mem_available_known=True,
    )
    broker._refresh = lambda force=False: snapshot
    return broker

@pytest.mark.parametrize("backend", ["pandas", "polars_long", "auto"])
def test_run_many_normalize_extrema_and_shared_subexpression(backend):
    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", periods=4), ["A", "B", "C"]],
        names=["timestamp", "instrument"],
    )
    values = pd.Series(np.tile([-1e308, 0., 1e308], 4), index=index)
    source = InMemorySeriesSource(data={"x": values})
    # Exact bounded fixture scope supplies the same shape evidence required
    # from real sources; auto must not invent row counts for unknown sources.
    source.instrument_filter = ("A", "B", "C")
    source.start_date = index.levels[0].min().tz_localize("UTC")
    source.end_date = index.levels[0].max().tz_localize("UTC")
    source.schema = {"x": "float64"}
    engine = FactorEngine(backend=build_backend(backend), data_source=source, run_mode="research")
    normalized = F("normalize")(col("x"))
    engine.resource_broker = _coherent_broker()
    factors = [
        Factor(name="normalized", expr=normalized),
        Factor(name="normalized_abs", expr=F("abs")(normalized)),
    ]
    out = engine.run_many(factors)
    expected = pd.Series(np.tile([0., .5, 1.], 4), index=index)
    for name in ("normalized", "normalized_abs"):
        pd.testing.assert_series_equal(out["results"][name], expected,
                                       check_names=False, check_dtype=False)
    assert out["dag"].shared_nodes
