from __future__ import annotations

import pandas as pd

from factor_engine.storage.datasource import DataSource


class _MemorySource(DataSource):
    def __init__(self):
        self.start_date = "2024-01-01"
        self.end_date = "2024-12-31"
        self.bar_freq = "1d"
        self.params = {
            "full_history_start": "2000-01-03",
            "session_open": "09:30",
            "session_close": "16:00",
            "timestamp_convention": "bar_end",
        }
        self.read_auto = True
        self._lazy_scan = True
        index = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=5, freq="B"), ["A"]],
            names=["timestamp", "instrument"],
        )
        self.series = pd.Series(range(5), index=index, dtype=float)

    @property
    def lazy_scan(self):
        return self._lazy_scan

    def load_column(self, name):
        return self.series


def test_generic_window_preserves_read_contracts_and_public_bounds():
    import factor_engine.runtime  # noqa: F401 - installs narrowing contract
    from factor_engine.storage.time_window import narrow_data_source_for_window

    source = _MemorySource()
    narrowed = narrow_data_source_for_window(
        source,
        start_date="2024-01-03",
        end_date="2024-01-05",
        bar_freq="1d",
    )
    assert narrowed.params == source.params
    assert narrowed.read_auto is True
    assert narrowed._lazy_scan is True
    assert str(narrowed.start_date).startswith("2024-01-03")
    assert str(narrowed.end_date).startswith("2024-01-05")
    result = narrowed.load_column("close")
    dates = result.index.get_level_values("timestamp")
    assert dates.min() >= pd.Timestamp("2024-01-03")
    assert dates.max() <= pd.Timestamp("2024-01-05")


def test_logical_source_remains_logical_after_windowing():
    import factor_engine.runtime  # noqa: F401
    from factor_engine.storage.sources.lqtp_logical_source_v2 import LQTPLogicalDataSource
    from factor_engine.storage.time_window import narrow_data_source_for_window

    logical = LQTPLogicalDataSource(_MemorySource(), factor_freq="1d")
    logical._execution_id = "execution-1"
    narrowed = narrow_data_source_for_window(
        logical,
        start_date="2024-01-03",
        end_date="2024-01-05",
        bar_freq="1d",
    )
    assert isinstance(narrowed, LQTPLogicalDataSource)
    assert narrowed._execution_id == "execution-1"
    assert narrowed.factor_freq == "1d"
    assert narrowed.params["full_history_start"] == "2000-01-03"
    dates = narrowed.load_column("close").index.get_level_values("timestamp")
    assert dates.min() >= pd.Timestamp("2024-01-03")
