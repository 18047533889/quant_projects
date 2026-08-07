from dataclasses import dataclass, field

import pytest

pd = pytest.importorskip("pandas")

from storage.composite_source import CompositeDataSource
from storage.datasource import DataSource


def _build_series(rows: list[tuple[str, str, float]]) -> "pd.Series":
    frame = pd.DataFrame(rows, columns=["timestamp", "instrument", "value"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    series = frame.set_index(["timestamp", "instrument"])["value"].sort_index()
    series.index = series.index.set_names(["timestamp", "instrument"])
    return series


@dataclass
class CountingSeriesSource(DataSource):
    data: dict[str, "pd.Series"]
    calls: dict[str, int] = field(default_factory=dict)

    def load_column(self, name: str):
        self.calls[name] = self.calls.get(name, 0) + 1
        return self.data[name]


def test_composite_source_aligns_auxiliary_columns_to_anchor_and_caches():
    price = CountingSeriesSource(
        {
            "close": _build_series(
                [
                    ("2024-01-02", "AAA", 10.0),
                    ("2024-01-02", "BBB", 20.0),
                    ("2024-01-03", "AAA", 12.0),
                    ("2024-01-03", "BBB", 18.0),
                ]
            )
        }
    )
    fundamental = CountingSeriesSource(
        {
            "price_to_earnings": _build_series(
                [
                    ("2024-01-01", "AAA", 2.0),
                    ("2024-01-01", "BBB", 5.0),
                    ("2024-01-03", "AAA", 3.0),
                    ("2024-01-03", "BBB", 6.0),
                ]
            )
        }
    )

    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        aliases={"pe": "fundamental.price_to_earnings"},
        joins={"fundamental": {"method": "asof_backward"}},
    )

    pe = source.load_column("pe")

    assert list(pe.index) == list(price.data["close"].index)
    assert pe.loc[(pd.Timestamp("2024-01-02"), "AAA")] == pytest.approx(2.0)
    assert pe.loc[(pd.Timestamp("2024-01-02"), "BBB")] == pytest.approx(5.0)
    assert pe.loc[(pd.Timestamp("2024-01-03"), "AAA")] == pytest.approx(3.0)
    assert pe.loc[(pd.Timestamp("2024-01-03"), "BBB")] == pytest.approx(6.0)
    assert price.calls == {"close": 1}
    assert fundamental.calls == {"price_to_earnings": 1}

    close = source.load_column("close")
    again = source.load_column("fundamental.price_to_earnings")

    assert close.equals(price.data["close"])
    assert again.equals(pe)
    assert price.calls == {"close": 1}
    assert fundamental.calls == {"price_to_earnings": 1}


@dataclass
class _BatchingSeriesSource(DataSource):
    """带 load_columns 计数的子源（模拟 DataAccessSource 的批量读）。"""

    data: dict[str, "pd.Series"]
    load_columns_calls: list[list[str]] = field(default_factory=list)

    def load_column(self, name: str):
        self.load_columns_calls.append([name])
        return self.data[name]

    def load_columns(self, names: list[str]):
        self.load_columns_calls.append(list(names))
        return {n: self.data[n] for n in names}


def test_composite_batches_same_source_columns_into_one_load():
    price = CountingSeriesSource(
        {"close": _build_series([("2024-01-02", "AAA", 10.0), ("2024-01-03", "AAA", 12.0)])}
    )
    fundamental = _BatchingSeriesSource(
        {
            "pe": _build_series([("2024-01-01", "AAA", 2.0), ("2024-01-03", "AAA", 3.0)]),
            "pb": _build_series([("2024-01-01", "AAA", 4.0), ("2024-01-03", "AAA", 5.0)]),
        }
    )
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": {"method": "asof_backward"}},
    )

    batch = source.load_columns(["fundamental.pe", "fundamental.pb"])

    assert set(batch) == {"fundamental.pe", "fundamental.pb"}
    # 同源多列 → 一次 load_columns（而非逐列两次）
    fundamental_calls = [c for c in fundamental.load_columns_calls if c != ["close"]]
    assert fundamental_calls == [["pe", "pb"]]


def test_composite_source_supports_exact_alignment():
    price = CountingSeriesSource(
        {
            "close": _build_series(
                [
                    ("2024-01-02", "AAA", 10.0),
                    ("2024-01-03", "AAA", 12.0),
                ]
            )
        }
    )
    fundamental = CountingSeriesSource(
        {
            "price_to_earnings": _build_series(
                [
                    ("2024-01-01", "AAA", 2.0),
                    ("2024-01-03", "AAA", 3.0),
                ]
            )
        }
    )

    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": "exact"},
    )

    pe = source.load_column("fundamental.price_to_earnings")

    assert pd.isna(pe.loc[(pd.Timestamp("2024-01-02"), "AAA")])
    assert pe.loc[(pd.Timestamp("2024-01-03"), "AAA")] == pytest.approx(3.0)


def test_composite_collect_join_reports():
    price = CountingSeriesSource(
        {
            "close": _build_series(
                [
                    ("2024-01-02", "AAA", 10.0),
                    ("2024-01-03", "AAA", 12.0),
                ]
            )
        }
    )
    fundamental = CountingSeriesSource(
        {
            "price_to_earnings": _build_series(
                [
                    ("2024-01-01", "AAA", 2.0),
                    ("2024-01-03", "AAA", 3.0),
                ]
            )
        }
    )
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fundamental},
        joins={"fundamental": "asof_backward"},
    )
    source.load_column("fundamental.price_to_earnings")
    reports = source.collect_join_reports()
    assert len(reports) == 1
    assert reports[0]["source"] == "fundamental"
    assert reports[0]["matched_rows"] == 2
    assert reports[0]["method"] == "asof_backward"