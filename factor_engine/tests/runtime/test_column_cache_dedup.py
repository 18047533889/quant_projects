"""input_dq + prefetch 列缓存去重测试。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from runtime.input_dq import assert_input_dq
from storage.datasource import DataSource


def _panel(values: list[float]) -> pd.Series:
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B"]],
        names=["timestamp", "instrument"],
    )
    return pd.Series(values, index=idx)


@dataclass
class _CountingColumnSource(DataSource):
    """模拟 DataAccessSource 的 _column_cache + prefetch_columns 行为。"""

    data: dict[str, pd.Series]
    load_columns_calls: int = field(default=0, init=False)
    _column_cache: dict[str, pd.Series] = field(default_factory=dict, init=False)

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        needed = [n for n in names if n not in self._column_cache]
        if needed:
            self.load_columns_calls += 1
            for n in needed:
                self._column_cache[n] = self.data[n]
        return {n: self._column_cache[n] for n in names}

    def load_column(self, name: str) -> pd.Series:
        if name in self._column_cache:
            return self._column_cache[name]
        return self.load_columns([name])[name]

    def prefetch_columns(self, names: list[str]) -> None:
        self.load_columns(names)


def test_input_dq_then_prefetch_single_load_columns_batch():
    panel = _panel([1.0, 2.0, 3.0, 4.0])
    src = _CountingColumnSource(data={"close": panel, "open": panel + 0.5})

    assert_input_dq(src, ["close", "open"], raise_on_fail=True)
    assert src.load_columns_calls == 1

    src.prefetch_columns(["close", "open"])
    assert src.load_columns_calls == 1


def test_prefetch_then_load_column_uses_cache():
    panel = _panel([1.0, 2.0, 3.0, 4.0])
    src = _CountingColumnSource(data={"close": panel})

    src.prefetch_columns(["close"])
    assert src.load_columns_calls == 1

    series = src.load_column("close")
    assert src.load_columns_calls == 1
    assert len(series) == 4
