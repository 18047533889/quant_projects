import pandas as pd
import polars as pl
from factor_engine.backend.context import ExecutionContext
from factor_engine.backend.polars_backend import PolarsBackend
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.storage.sources.wave_prefetched_source import WavePrefetchedSourceAdapter


class NoReopen:
    def load_column(self, name):
        raise AssertionError("must use the admitted wave, not reopen its source")
    def scan_polars_long(self, columns):
        raise AssertionError("must use the admitted wave, not rescan its source")


def panel():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-05", periods=2), ["A"]],
        names=["timestamp", "instrument"])
    return pd.Series([7.0, 8.0], index=index, name="low")


def test_polars_pandas_wave_uses_bound_value_without_native_rescan():
    expected = panel()
    source = WavePrefetchedSourceAdapter(NoReopen())
    source.publish_columns(1, ["low"], {"low": expected})
    ctx = ExecutionContext(data_source=source)
    actual = PolarsBackend().execute(PlanNode("column", attrs={"name": "low"}), ctx)
    pd.testing.assert_series_equal(actual.sort_index(), expected, check_names=False)
    assert not (ctx.runtime_stats or {}).get("polars_expr", False)


def test_polars_native_wave_still_uses_expression_fastpath():
    expected = panel()
    source = WavePrefetchedSourceAdapter(NoReopen())
    frame = pl.DataFrame({"ts": expected.index.get_level_values(0),
                          "inst": ["A", "A"], "low": [7.0, 8.0]})
    source.publish_native(1, ["low"], frame)
    ctx = ExecutionContext(data_source=source)
    actual = PolarsBackend().execute(PlanNode("column", attrs={"name": "low"}), ctx)
    pd.testing.assert_series_equal(actual.sort_index(), expected, check_names=False)
    assert ctx.runtime_stats["polars_expr"] is True
