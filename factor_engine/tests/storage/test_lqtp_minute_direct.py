from types import MethodType, SimpleNamespace

import pandas as pd
import pytest
import numpy as np

from factor_engine.api.source_ref import SourceRefSpec, encode_source_ref
from factor_engine.backend.cleaned_bridge import _call_cleaned_operator
from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.storage.sources.data_access_source import MissingDataDependencyError
from factor_engine.storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource


def _source_with_child(series):
    source = object.__new__(LQTPLogicalDataSource)
    dependencies = []
    source._anchor_index = MethodType(
        lambda self: pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2025-01-02"), "000001.SZ")],
            names=["timestamp", "instrument"],
        ),
        source,
    )
    child = SimpleNamespace(
        load_column=lambda field: series.rename(field),
        data_snapshot_id="minute-snapshot",
    )
    source._child = MethodType(lambda self, dataset: child, source)
    source._record_dependency = MethodType(
        lambda self, dataset, **kwargs: dependencies.append((dataset, kwargs)), source
    )
    return source, dependencies


def test_adjusted_minute_source_ref_preserves_session_axis():
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-01-02 01:31:00"), "000001.SZ"),
            (pd.Timestamp("2025-01-02 01:32:00"), "000001.SZ"),
        ],
        names=["timestamp", "instrument"],
    )
    minute = pd.Series([100, 200], index=index, name="Volume")
    source, dependencies = _source_with_child(minute)

    out = source._load_source_ref(SourceRefSpec("StockMinuteBarAdj", "Volume"))

    expected = minute.copy()
    expected.index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-01-02 09:31:00"), "000001.SZ"),
            (pd.Timestamp("2025-01-02 09:32:00"), "000001.SZ"),
        ],
        names=["timestamp", "instrument"],
    )
    pd.testing.assert_series_equal(out, expected)
    assert dependencies == [
        (
            "ashare_stock_minute_adj",
            {
                "kind": "minute_session",
                "snapshot_id": "minute-snapshot",
                "field": "Volume",
                "join_policy": "exact_session",
            },
        )
    ]


def test_adjusted_minute_batch_uses_session_route_without_daily_alignment():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-02 01:31:00"), "000001.SZ")],
        names=["timestamp", "instrument"],
    )
    minute = pd.Series([100], index=index, name="Volume")
    source, _ = _source_with_child(minute)
    spec = SourceRefSpec("StockMinuteBarAdj", "Volume")
    encoded = encode_source_ref(spec)

    out = source.load_source_refs_batch([encoded])[encoded]

    assert out.index[0] == (pd.Timestamp("2025-01-02 09:31:00"), "000001.SZ")
    assert len(out) == 1


def test_explicit_raw_daily_source_ref_does_not_reuse_adjusted_anchor():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-02"), "000001.SZ")],
        names=["timestamp", "instrument"],
    )
    raw = pd.Series([9.5], index=index, name="LowLimit")
    source, dependencies = _source_with_child(raw)
    source.inner = SimpleNamespace(
        load_column=lambda field: pytest.fail(
            f"explicit StockDailyBar.{field} reused the adjusted anchor"
        )
    )

    spec = SourceRefSpec("StockDailyBar", "LowLimit")
    encoded = encode_source_ref(spec)
    out = source.load_source_refs_batch([encoded])[encoded]

    pd.testing.assert_series_equal(out, raw)
    assert dependencies == [
        (
            "ashare_stock_daily",
            {
                "kind": "daily_exact",
                "snapshot_id": "minute-snapshot",
                "field": "LowLimit",
                "join_policy": "exact",
            },
        )
    ]


def test_adjusted_minute_transform_never_substitutes_raw_dataset():
    source, _ = _source_with_child(pd.Series(dtype=float))
    spec = SourceRefSpec(
        "StockMinuteBarAdj",
        "AdjAmount",
        transform="minute_range",
        transform_params=(("start", "09:31"), ("end", "10:00")),
    )
    with pytest.raises(MissingDataDependencyError, match="refusing to substitute raw"):
        source._load_source_ref(spec)


def test_engine_dispatch_supplies_declared_ashare_session_calendar():
    morning = pd.date_range("2025-01-02 09:31", "2025-01-02 11:30", freq="1min")
    afternoon = pd.date_range("2025-01-02 13:01", "2025-01-02 15:00", freq="1min")
    panel = pd.DataFrame(
        {"000001.SZ": np.arange(1.0, 241.0)}, index=morning.append(afternoon)
    )
    ensure_cleaned_loaded()
    operator = OperatorRegistry.get(
        "intraday_activity_duration_curvature", "pandas_numpy"
    )
    result = _call_cleaned_operator(
        "intraday_activity_duration_curvature", operator, [panel], {"buckets": 10}
    )
    assert np.isfinite(result.iloc[0, 0])
