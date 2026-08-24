"""cleaned_operators 接入 factor_engine 的冒烟测试。"""

import pytest

pd = pytest.importorskip("pandas")

from factor_engine.api import rank, ts_mean
from factor_engine.api.columns import col
from factor_engine.api.dsl_parser import parse_expr
from factor_engine.api.factor import Factor
from factor_engine.backend.pandas_backend import PandasBackend
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def _panel_idx():
    return pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            ["A", "B"],
        ],
        names=["timestamp", "instrument"],
    )


def test_cleaned_ops_rank_ts_mean():
    idx = _panel_idx()
    close = pd.Series(
        [10.0, 20.0, 11.0, 21.0, 12.0, 18.0, 13.0, 19.0],
        index=idx,
    )
    source = InMemorySeriesSource(data={"close": close})
    factor = Factor(name="mom_2_rank", expr=rank(ts_mean(col("close"), 2)))
    out = FactorEngine(backend=PandasBackend(), data_source=source).run(factor)
    assert out["result"].index.names == ["timestamp", "instrument"]


def test_cleaned_only_op_sma():
    idx = _panel_idx()
    close = pd.Series(
        [10.0, 20.0, 11.0, 21.0, 12.0, 18.0, 13.0, 19.0],
        index=idx,
    )
    source = InMemorySeriesSource(data={"close": close})
    expr = parse_expr("rank(SMA(col('close'), 2))")
    factor = Factor(name="sma2_rank", expr=expr)
    out = FactorEngine(backend=PandasBackend(), data_source=source).run(factor)
    result = out["result"]
    assert isinstance(result, pd.Series)
    assert result.notna().any()


def test_cleaned_dsl_allowlist_includes_sma():
    from factor_engine.api.operator_registry import build_dsl_allowlist

    allow = build_dsl_allowlist()
    assert "SMA" in allow
    assert "ts_mean" in allow
